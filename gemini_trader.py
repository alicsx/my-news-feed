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

# کلیدهای API را از متغیرهای محیطی یا GitHub Secrets بخوان
google_api_key = os.getenv("GOOGLE_API_KEY")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")

if not all([google_api_key, TWELVEDATA_API_KEY]):
    raise ValueError("لطفاً کلیدهای API را تنظیم کنید: GOOGLE_API_KEY, TWELVEDATA_API_KEY")

genai.configure(api_key=google_api_key)

# --- تنظیمات استراتژی ---
HIGH_TIMEFRAME = "4h"
LOW_TIMEFRAME = "1h"
CANDLES_TO_FETCH = 200

CURRENCY_PAIRS_TO_ANALYZE = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
    "GBP/JPY", "EUR/JPY", "AUD/JPY", "NZD/USD", "USD/CAD"
]

# --- تنظیمات فنی ---
CACHE_FILE = "signal_cache.json"
CACHE_DURATION_HOURS = 2 # زمان کش را کاهش می‌دهیم تا تحلیل‌ها به‌روزتر باشند
LOG_FILE = "trading_log.log"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()])

# =================================================================================
# --- توابع اصلی سیستم ---
# =================================================================================

# توابع normalize_pair_format, load_cache, save_cache, is_pair_on_cooldown, get_market_data, apply_full_technical_indicators
# بدون تغییر باقی می‌مانند و برای کامل بودن کد اینجا آورده شده‌اند.

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
        logging.info(f"جفت ارز {pair} در حافظه کش است (cooldown). از تحلیل مجدد صرف‌نظر می‌شود.")
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
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            df = df.sort_values('datetime', ascending=True).reset_index(drop=True)
            return df
    except Exception as e:
        logging.error(f"خطا در دریافت دیتای بازار برای {symbol}: {e}")
    return None

def apply_full_technical_indicators(df):
    if df is None or df.empty: return None
    logging.info("محاسبه اندیکاتورهای جامع...")
    try:
        if len(df) < 50: return pd.DataFrame()
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
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logging.error(f"خطا در هنگام محاسبه اندیکاتورهای تکنیکال: {e}")
        return pd.DataFrame()

# ✨ NEW: Function to prepare a comprehensive technical briefing ✨
def gather_technical_briefing(htf_df, ltf_df):
    """یک خلاصه و پرونده تحلیلی کامل از وضعیت تکنیکال بازار تهیه می‌کند."""
    if htf_df.empty or ltf_df.empty:
        return "داده‌های تکنیکال برای تهیه گزارش کافی نیست."

    last_htf = htf_df.iloc[-1]
    last_ltf = ltf_df.iloc[-1]

    # تعیین وضعیت بازار
    adx = last_htf.get('ADX_14', 'N/A')
    market_regime = "UNCLEAR"
    if isinstance(adx, float):
        if adx > 25: market_regime = "TRENDING"
        elif adx < 20: market_regime = "RANGING"
    
    # تعیین روند
    htf_trend = "UNCLEAR"
    if last_htf.get('EMA_21') > last_htf.get('EMA_50'): htf_trend = "UPTREND"
    elif last_htf.get('EMA_21') < last_htf.get('EMA_50'): htf_trend = "DOWNTREND"
        
    briefing = f"""
- **High-Timeframe Trend ({HIGH_TIMEFRAME}):** {htf_trend}
- **Market Regime (HTF ADX):** {market_regime} ({adx:.2f})
- **Low-Timeframe Momentum ({LOW_TIMEFRAME} MACD):** {'Bullish' if last_ltf.get('MACD_12_26_9') > last_ltf.get('MACDs_12_26_9') else 'Bearish'}
- **Low-Timeframe RSI ({LOW_TIMEFRAME}):** {last_ltf.get('RSI_14'):.2f}
- **Key Support (HTF):** {last_htf.get('sup'):.5f}
- **Key Resistance (HTF):** {last_htf.get('res'):.5f}
- **Current Volatility (LTF ATR):** {last_ltf.get('ATRr_14'):.5f}
- **Volume Status (LTF):** {'Above Average' if 'volume' in last_ltf and 'VOLUME_SMA_20' in last_ltf and last_ltf['volume'] > last_ltf['VOLUME_SMA_20'] else 'Normal or Below Average'}
"""
    return briefing.strip()

# ✨ NEW: The ultimate AI prompt that receives the full briefing ✨
def get_ai_trade_decision(symbol, technical_briefing):
    """
    پرونده تحلیلی را به هوش مصنوعی ارسال کرده و از او می‌خواهد تا بهترین سیگنال ممکن را ایجاد کند.
    """
    base_currency, quote_currency = symbol.split('/')
    prompt = f"""
    You are the Head Analyst and primary decision-maker at a proprietary trading firm. Your quantitative assistant has prepared the following technical briefing for **{symbol}**.

    Your task is to synthesize this quantitative data with your own deep, qualitative knowledge (including fundamental analysis, news events, and price action nuances) to decide if a high-probability trade exists.

    **Quantitative Technical Briefing from Assistant:**
    ```
    {technical_briefing}
    ```

    **Your Mandate:**
    1.  **Analyze the Full Picture:** Review the provided technical data. Does it present a clear picture?
    2.  **Add Your Qualitative Layer:** Search for upcoming high-impact news for '{base_currency}' and '{quote_currency}'. How does the current market sentiment and fundamental outlook align with the technical briefing?
    3.  **Make the Final Call:**
        -   If you identify a clear, high-probability opportunity where technicals and fundamentals align, **CREATE** a trade signal with precise parameters. Your reasoning should reflect the confluence of factors.
        -   If the data is conflicting, the outlook is uncertain, or no quality setup exists, you MUST respond ONLY with the word **"NO_SIGNAL"**.

    **Strict Output Format:**
    - If no trade: `NO_SIGNAL`
    - If a trade is identified:
    PAIR: {symbol}
    TYPE: [BUY_STOP, SELL_STOP, BUY_LIMIT, or SELL_LIMIT]
    ENTRY: [Precise Entry Price]
    SL: [Precise Stop Loss Price]
    TP: [Precise Take Profit Price]
    CONFIDENCE: [Your final confidence score from 1-10]
    REASON: [Your concise, expert reason for the trade, combining technical and fundamental insights]
    """
    logging.info(f"ارسال پرونده تحلیلی {symbol} به AI برای تصمیم‌گیری نهایی...")
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt.strip(), request_options={'timeout': 180})
        
        if "NO_SIGNAL" in response.text.upper():
            logging.info(f"هوش مصنوعی پس از بررسی پرونده، هیچ فرصت مناسبی برای {symbol} پیدا نکرد.")
            return None
        
        logging.info(f"هوش مصنوعی یک سیگنال معاملاتی برای {symbol} ایجاد کرد.")
        return response.text
    except Exception as e:
        logging.error(f"خطا در تصمیم‌گیری نهایی AI: {e}")
        return None

# =================================================================================
# --- حلقه اصلی برنامه ---
# =================================================================================

def main():
    """تابع اصلی برای اجرای کامل فرآیند تولید سیگنال."""
    logging.info("================== شروع اجرای اسکریپت (مدل دستیار کوانتومی) ==================")
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
            
        # ✨ NEW: The script's main job is now to create the briefing ✨
        technical_briefing = gather_technical_briefing(htf_df_analyzed, ltf_df_analyzed)
        logging.info(f"پرونده تحلیلی برای {pair} تهیه شد:\n{technical_briefing}")
        
        # Send the briefing to the AI for the final decision
        final_response = get_ai_trade_decision(pair, technical_briefing)
        
        if final_response:
            final_response_with_exp = final_response.strip() + "\nExpiration: 6 hours"
            final_signals.append(final_response_with_exp)
            signal_cache[pair] = datetime.now(UTC).isoformat()

        logging.info(f"... تحلیل {pair} تمام شد. تاخیر ۱۰ ثانیه‌ای ...")
        time.sleep(10)

    # --- ذخیره سیگنال‌های نهایی و حافظه ---
    if final_signals:
        output_content = "AI-Generated Signals from Quantitative Briefings\n" + "="*60 + "\n\n"
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
