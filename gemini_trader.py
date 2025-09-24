import google.generativeai as genai
import os
import re
import time
import json
import logging
import pandas as pd
import pandas_ta as ta
import requests
from datetime import datetime, timedelta, UTC

# =================================================================================
# --- بخش تنظیمات اصلی ---
# =================================================================================

google_api_key = os.getenv("GOOGLE_API_KEY")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")

if not all([google_api_key, TWELVEDATA_API_KEY]):
    raise ValueError("لطفاً کلیدهای API را تنظیم کنید: GOOGLE_API_KEY, TWELVEDATA_API_KEY")

genai.configure(api_key=google_api_key)

HIGH_TIMEFRAME = "4h"
LOW_TIMEFRAME = "1h"
CANDLES_TO_FETCH = 200
CURRENCY_PAIRS_TO_ANALYZE = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
    "GBP/JPY", "EUR/JPY", "AUD/JPY", "NZD/USD", "USD/CAD"
]
CACHE_FILE = "signal_cache.json"
CACHE_DURATION_HOURS = 2
LOG_FILE = "trading_log.log"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()])

# =================================================================================
# --- توابع اصلی سیستم ---
# =================================================================================

def normalize_pair_format(pair_string):
    if not isinstance(pair_string, str): return ""
    pair_string = pair_string.upper().strip()
    if "/" in pair_string: return pair_string
    if len(pair_string) == 6: return f"{pair_string[:3]}/{pair_string[3:]}"
    logging.warning(f"فرمت جفت ارز '{pair_string}' ناشناخته است.")
    return pair_string

def load_cache():
    if not os.path.exists(CACHE_FILE): return {}
    try:
        with open(CACHE_FILE, 'r') as f: return json.load(f)
    except (json.JSONDecodeError, IOError): return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f: json.dump(cache, f, indent=4)

def is_pair_on_cooldown(pair, cache):
    if pair not in cache: return False
    last_signal_time = datetime.fromisoformat(cache[pair])
    if datetime.now(UTC) - last_signal_time < timedelta(hours=CACHE_DURATION_HOURS):
        logging.info(f"جفت ارز {pair} در حافظه کش است. از تحلیل مجدد صرف‌نظر می‌شود.")
        return True
    return False

def get_market_data(symbol, interval, outputsize):
    logging.info(f"دریافت {outputsize} کندل {interval} برای {symbol}...")
    url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={TWELVEDATA_API_KEY}'
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if 'values' in data and len(data['values']) > 0:
            df = pd.DataFrame(data['values'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            df = df.sort_values('datetime', ascending=True).reset_index(drop=True)
            return df
    except Exception as e:
        logging.error(f"خطا در دریافت دیتای بازار برای {symbol}: {e}")
    return None

# ✨ FIX: Corrected candlestick function names ✨
def apply_full_technical_indicators(df):
    if df is None or df.empty: return None
    logging.info("محاسبه اندیکاتورهای جامع و الگوهای کندل استیک...")
    try:
        if len(df) < 50: return pd.DataFrame()
        # Standard indicators
        df.ta.ema(length=21, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        if 'volume' in df.columns:
            df.ta.sma(close=df['volume'], length=20, prefix="VOLUME", append=True)
        df['sup'] = df['low'].rolling(window=10, min_periods=3).min()
        df['res'] = df['high'].rolling(window=10, min_periods=3).max()
        
        # Correct candlestick pattern recognition function calls
        df.ta.doji(append=True)
        df.ta.hammer(append=True)
        df.ta.engulfing(append=True)
        
        df.dropna(inplace=True)
        if df.empty: logging.warning("دیتافریم پس از محاسبه اندیکاتورها تهی شد.")
        return df
    except Exception as e:
        logging.error(f"خطا در هنگام محاسبه اندیکاتورهای تکنیکال: {e}")
        return pd.DataFrame()

# ✨ FIX: Updated column names for candlestick patterns ✨
def gather_technical_briefing(htf_df, ltf_df):
    if htf_df.empty or ltf_df.empty:
        return "داده‌های تکنیکال برای تهیه گزارش کافی نیست."

    last_htf = htf_df.iloc[-1]
    last_ltf = ltf_df.iloc[-1]

    adx = last_htf.get('ADX_14', 'N/A')
    market_regime = "UNCLEAR"
    if isinstance(adx, float):
        if adx > 25: market_regime = "TRENDING"
        elif adx < 20: market_regime = "RANGING"
    
    htf_trend = "UNCLEAR"
    if last_htf.get('EMA_21') > last_htf.get('EMA_50'): htf_trend = "UPTREND"
    elif last_htf.get('EMA_21') < last_htf.get('EMA_50'): htf_trend = "DOWNTREND"

    atr_avg = ltf_df['ATRr_14'].rolling(window=50).mean().iloc[-1]
    volatility_state = "High" if last_ltf.get('ATRr_14') > atr_avg * 1.5 else \
                       "Low" if last_ltf.get('ATRr_14') < atr_avg * 0.7 else "Normal"

    # Correctly check for the columns created by the library
    recent_candles = ltf_df.tail(3)
    patterns = []
    if 'DOJI_10_0.1' in recent_candles.columns and recent_candles['DOJI_10_0.1'].sum() > 0: patterns.append("Doji (Indecision)")
    if 'CDL_HAMMER' in recent_candles.columns and recent_candles['CDL_HAMMER'].sum() > 0: patterns.append("Hammer (Bullish Reversal)")
    if 'CDL_ENGULFING' in recent_candles.columns and recent_candles['CDL_ENGULFING'].sum() != 0: patterns.append("Engulfing Pattern")
    candlestick_summary = ", ".join(patterns) if patterns else "No significant pattern"

    briefing = f"""
- **Overall Trend ({HIGH_TIMEFRAME}):** {htf_trend}
- **Market Regime (HTF ADX):** {market_regime} ({adx:.2f})
- **Price vs. LTF EMAs:** {'Above' if last_ltf.get('close') > last_ltf.get('EMA_50') else 'Below'} key moving averages.
- **Momentum (LTF MACD):** {'Bullish' if last_ltf.get('MACD_12_26_9') > last_ltf.get('MACDs_12_26_9') else 'Bearish'}
- **Strength (LTF RSI):** {last_ltf.get('RSI_14'):.2f}
- **Volatility (LTF ATR):** {volatility_state} (Value: {last_ltf.get('ATRr_14'):.5f})
- **Recent Candlestick Pattern (LTF):** {candlestick_summary}
- **Key Support (HTF):** {last_htf.get('sup'):.5f}
- **Key Resistance (HTF):** {last_htf.get('res'):.5f}
- **Proximity to Key Levels:** Price is closer to {'Support' if abs(last_ltf.get('close') - last_htf.get('sup')) < abs(last_ltf.get('close') - last_htf.get('res')) else 'Resistance'}.
"""
    return briefing.strip()

def get_ai_trade_decision(symbol, technical_briefing):
    # This function's logic remains unchanged as it was already robust.
    base_currency, quote_currency = symbol.split('/')
    prompt = f"""
    You are a seasoned Portfolio Manager at a multi-billion dollar hedge fund. Your quantitative analysis (Quant) team has just handed you the following technical briefing for **{symbol}**. Your job is to make the final, critical decision to trade or not to trade.

    **Quant Team's Technical Briefing:**
    ```
    {technical_briefing}
    ```

    **Your Decision-Making Framework:**

    1.  **Synthesize Technicals:** Review the quant briefing. Is there a coherent, compelling technical story? (e.g., "The data shows a clear uptrend pulling back to a key support level with bullish candlestick confirmation.")

    2.  **Apply Fundamental Overlay:** Conduct a quick but thorough analysis of the fundamental picture for '{base_currency}' and '{quote_currency}'. Check for high-impact news in the next 8 hours. What is the prevailing market sentiment? Does the fundamental narrative align with the technical story, or does it create a dangerous contradiction?

    3.  **Assess Risk & Price Action:** Look beyond the indicators. Is the price action choppy or clean? Is the volatility too high for a safe entry, or too low for a profitable move? What is the single biggest risk to this trade idea?

    4.  **Formulate Final Trade Plan:**
        -   If, and only if, you find a strong **CONFLUENCE** across Technicals, Fundamentals, and Price Action, **CREATE** a trade signal. Your reasoning must clearly state how these factors align.
        -   If there is any significant doubt or contradiction, you MUST protect capital and respond ONLY with **"NO_SIGNAL"**.

    **Strict Output Format:**
    - If no trade: `NO_SIGNAL`
    - If a trade is identified:
    PAIR: {symbol}
    TYPE: [BUY_STOP, SELL_STOP, BUY_LIMIT, or SELL_LIMIT]
    ENTRY: [Precise Entry Price]
    SL: [Precise Stop Loss Price]
    TP: [Precise Take Profit Price]
    CONFIDENCE: [Your final confidence score from 1-10]
    REASON: [Your expert reason for the trade, synthesizing all factors]
    """
    logging.info(f"ارسال پرونده تحلیلی {symbol} به مدیر پورتفولیو (AI) برای تصمیم‌گیری نهایی...")
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt.strip(), request_options={'timeout': 180})
        
        if "NO_SIGNAL" in response.text.upper():
            logging.info(f"AI پس از بررسی کامل، هیچ فرصت مناسبی برای {symbol} پیدا نکرد.")
            return None
        
        logging.info(f"AI یک سیگنال معاملاتی با تحلیل جامع برای {symbol} ایجاد کرد.")
        return response.text
    except Exception as e:
        logging.error(f"خطا در تصمیم‌گیری نهایی AI: {e}")
        return None

def main():
    logging.info("================== شروع اجرای اسکریپت (مدل مدیر پورتفولیو) ==================")
    signal_cache = load_cache()
    final_signals = []

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, help="Specify a single currency pair to analyze")
    args, _ = parser.parse_known_args()
    pairs_to_run = [args.pair] if args.pair else CURRENCY_PAIRS_TO_ANALYZE

    for pair_raw in pairs_to_run:
        pair = normalize_pair_format(pair_raw)
        if not pair or is_pair_on_cooldown(pair, signal_cache):
            time.sleep(1); continue

        logging.info(f"--- شروع تحلیل جامع برای: {pair} ---")
        
        htf_df = get_market_data(pair, HIGH_TIMEFRAME, CANDLES_TO_FETCH)
        ltf_df = get_market_data(pair, LOW_TIMEFRAME, CANDLES_TO_FETCH)

        htf_df_analyzed = apply_full_technical_indicators(htf_df)
        ltf_df_analyzed = apply_full_technical_indicators(ltf_df)

        if htf_df_analyzed is None or ltf_df_analyzed is None or htf_df_analyzed.empty or ltf_df_analyzed.empty:
            logging.warning(f"دیتای کافی برای تحلیل {pair} پس از پردازش وجود ندارد.")
            continue
            
        technical_briefing = gather_technical_briefing(htf_df_analyzed, ltf_df_analyzed)
        logging.info(f"پرونده تحلیلی برای {pair} تهیه شد:\n{technical_briefing}")
        
        final_response = get_ai_trade_decision(pair, technical_briefing)
        
        if final_response:
            final_response_with_exp = final_response.strip() + "\nExpiration: 6 hours"
            final_signals.append(final_response_with_exp)
            signal_cache[pair] = datetime.now(UTC).isoformat()

        logging.info(f"... تحلیل {pair} تمام شد. تاخیر ۱۰ ثانیه‌ای ...")
        time.sleep(10)

    if final_signals:
        output_content = "Portfolio Manager Grade Signals (Quant-Briefed AI)\n" + "="*60 + "\n\n"
        output_content += "\n---\n".join(final_signals)
        with open("trade_signal.txt", "w", encoding="utf-8") as f:
            f.write(output_content)
        logging.info(f"عملیات موفقیت‌آمیز بود. {len(final_signals)} سیگنال نهایی در trade_signal.txt ذخیره شد.")
    else:
        logging.info("در این اجرا، هوش مصنوعی هیچ سیگنال قابل معامله‌ای ایجاد نکرد.")

    save_cache(signal_cache)
    logging.info("================== پایان اجرای اسکریپت ==================")

if __name__ == "__main__":
    main()
