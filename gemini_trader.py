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
# --- بخش تنظیمات اصلی و استراتژی ---
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
ADX_TREND_THRESHOLD = 25
ADX_RANGE_THRESHOLD = 20

CURRENCY_PAIRS_TO_ANALYZE = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
    "GBP/JPY", "EUR/JPY", "AUD/JPY", "NZD/USD", "USD/CAD"
]

# --- تنظیمات فنی ---
CACHE_FILE = "signal_cache.json"
CACHE_DURATION_HOURS = 4
LOG_FILE = "trading_log.log"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()])

# =================================================================================
# --- توابع اصلی سیستم ---
# =================================================================================

def normalize_pair_format(pair_string):
    """جفت ارز ورودی را به فرمت استاندارد 'BASE/QUOTE' تبدیل می‌کند."""
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
        logging.info(f"جفت ارز {pair} در حافظه کش است (cooldown).")
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
        if len(df) < 50:
            logging.warning(f"داده‌های ورودی برای محاسبه اندیکاتورها کافی نیست. تعداد ردیف‌ها: {len(df)}")
            return pd.DataFrame()
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
        logging.info(f"شکل دیتافریم قبل از dropna: {df.shape}")
        df.dropna(inplace=True)
        logging.info(f"شکل دیتافریم بعد از dropna: {df.shape}")
        if df.empty: logging.warning("دیتافریم پس از محاسبه اندیکاتورها تهی شد.")
        return df
    except Exception as e:
        logging.error(f"خطا در هنگام محاسبه اندیکاتورهای تکنیکال: {e}")
        return pd.DataFrame()

def detect_market_regime(df):
    if df is None or df.empty or 'ADX_14' not in df.columns: return "UNCLEAR"
    last_adx = df.iloc[-1]['ADX_14']
    logging.info(f"تشخیص شرایط بازار... ADX فعلی: {last_adx:.2f}")
    if last_adx > ADX_TREND_THRESHOLD: return "TRENDING"
    if last_adx < ADX_RANGE_THRESHOLD: return "RANGING"
    return "UNCLEAR"

def find_trade_candidate(htf_df, ltf_df):
    market_regime = detect_market_regime(htf_df)
    if market_regime == "UNCLEAR": return None, None, None, None
    
    logging.info(f"اجرای استراتژی مناسب برای شرایط بازار '{market_regime}'...")
    last_ltf = ltf_df.iloc[-1]
    htf_support = htf_df.iloc[-1]['sup']
    htf_resistance = htf_df.iloc[-1]['res']
    
    volume_confirmed = 'VOLUME_SMA_20' in last_ltf and 'volume' in last_ltf and last_ltf['volume'] > last_ltf['VOLUME_SMA_20']

    if market_regime == "TRENDING":
        if 'MACD_12_26_9' not in last_ltf or 'MACDs_12_26_9' not in last_ltf: return None, None, None, None
        htf_trend = "UPTREND" if htf_df.iloc[-1]['EMA_21'] > htf_df.iloc[-1]['EMA_50'] else "DOWNTREND"
        if htf_trend == "UPTREND" and last_ltf['EMA_21'] > last_ltf['EMA_50'] and last_ltf['MACD_12_26_9'] > last_ltf['MACDs_12_26_9'] and volume_confirmed:
            return "BUY_TREND", market_regime, last_ltf, (htf_support, htf_resistance)
        if htf_trend == "DOWNTREND" and last_ltf['EMA_21'] < last_ltf['EMA_50'] and last_ltf['MACD_12_26_9'] < last_ltf['MACDs_12_26_9'] and volume_confirmed:
            return "SELL_TREND", market_regime, last_ltf, (htf_support, htf_resistance)

    elif market_regime == "RANGING":
        if 'BBL_20_2.0' not in last_ltf or 'BBU_20_2.0' not in last_ltf: return None, None, None, None
        if last_ltf['close'] <= last_ltf['BBL_20_2.0'] and last_ltf['RSI_14'] < 35:
            return "BUY_RANGE", market_regime, last_ltf, (htf_support, htf_resistance)
        if last_ltf['close'] >= last_ltf['BBU_20_2.0'] and last_ltf['RSI_14'] > 65:
            return "SELL_RANGE", market_regime, last_ltf, (htf_support, htf_resistance)
            
    return None, None, None, None

def get_ai_confluence_vetting(symbol, signal_type, market_regime, key_levels, ltf_df):
    """
    از هوش مصنوعی برای ارزیابی نهایی یک کاندیدای تکنیکال از سه جنبه (بنیادی، پرایس اکشن و تکنیکال) استفاده می‌کند.
    این تابع جامع، جایگزین تمام توابع قبلی AI شده است.
    """
    
    strategy_name = "Trend Following" if market_regime == "TRENDING" else "Mean Reversion"
    last_candle = ltf_df.iloc[-1]
    base_currency, quote_currency = symbol.split('/')

    prompt = f"""
    You are a multi-disciplinary Head Analyst at a trading desk. A quantitative system has proposed a trade. Your task is to perform the final vetting by checking for confluence across three key perspectives.

    **1. Technical Thesis (from the System):**
    - **Asset:** {symbol}
    - **Market Regime:** {market_regime}
    - **Strategy:** {strategy_name}
    - **Proposed Signal:** {signal_type}
    - **Quantitative Reason:** The system has identified this based on indicators like {'MACD/EMA/Volume' if market_regime == 'TRENDING' else 'Bollinger Bands/RSI'}.

    **Your Vetting Tasks:**

    **A. Fundamental & News Analysis:**
    - Check for any high-impact economic news for '{base_currency}' or '{quote_currency}' in the next 8 hours.
    - Briefly summarize the news and the market's sentiment/forecast.
    - **Crucially, does this fundamental outlook SUPPORT or CONTRADICT the technical thesis?**

    **B. Price Action Analysis:**
    - Examine the last few candles. Do you see confirming price action patterns (e.g., pin bars, engulfing patterns, consolidation before a breakout)?
    - **Does the recent price action SUPPORT or CONTRADICT the technical thesis?**

    **C. Final Decision:**
    - A trade is only valid if there is **CONFLUENCE** (agreement) across the Technical, Fundamental, and Price Action perspectives.
    - If all three align, **CONFIRM** the trade and provide optimal parameters. Set SL based on recent price structure (like a swing low/high) and ATR.
    - If there is a clear contradiction from either the Fundamental or Price Action view, **REJECT** the trade and state the reason for the contradiction.

    **Strict Output Format:**
    - If REJECTED: `REJECT: [State the contradiction, e.g., "Technical breakout contradicted by upcoming bearish news."]`
    - If CONFIRMED:
    PAIR: {symbol}
    TYPE: [Order Type]
    ENTRY: [Entry Price]
    SL: [Stop Loss Price]
    TP: [Take Profit Price]
    CONFIDENCE: [Score 1-10]
    REASON: [Concise reason highlighting the confluence, e.g., "Technical breakout aligns with bullish sentiment pre-CPI and is confirmed by price action."]
    """
    logging.info(f"ارسال سیگنال {symbol} به AI برای ارزیابی نهایی هم‌افزایی...")
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt.strip(), request_options={'timeout': 180})
        
        if "REJECT" in response.text.upper():
            logging.warning(f"هوش مصنوعی سیگنال {symbol} را به دلیل عدم هم‌افزایی رد کرد. دلیل: {response.text}")
            return None
        
        logging.info(f"هوش مصنوعی سیگنال {symbol} را بر اساس هم‌افزایی سه جانبه تأیید کرد.")
        return response.text
    except Exception as e:
        logging.error(f"خطا در ارزیابی نهایی AI: {e}")
        return None

# =================================================================================
# --- حلقه اصلی برنامه ---
# =================================================================================

def main():
    """تابع اصلی برای اجرای کامل فرآیند تولید سیگنال."""
    logging.info("================== شروع اجرای اسکریپت تحلیل هم‌افزایی ==================")
    signal_cache = load_cache()
    final_signals = []

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, help="Specify a single currency pair to analyze, e.g., EUR/USD")
    args, _ = parser.parse_known_args()

    pairs_to_run = [args.pair] if args.pair else CURRENCY_PAIRS_TO_ANALYZE

    for pair_raw in pairs_to_run:
        pair = normalize_pair_format(pair_raw)
        if not pair: continue

        logging.info(f"--- شروع تحلیل جامع برای: {pair} ---")
        if is_pair_on_cooldown(pair, signal_cache):
            time.sleep(2); continue

        htf_df = get_market_data(pair, HIGH_TIMEFRAME, CANDLES_TO_FETCH)
        htf_df_analyzed = apply_full_technical_indicators(htf_df)
        
        ltf_df = get_market_data(pair, LOW_TIMEFRAME, CANDLES_TO_FETCH)
        ltf_df_analyzed = apply_full_technical_indicators(ltf_df)

        if htf_df_analyzed is None or ltf_df_analyzed is None or htf_df_analyzed.empty or ltf_df_analyzed.empty:
            logging.warning(f"دیتای کافی برای تحلیل {pair} پس از پردازش وجود ندارد.")
            continue
            
        signal_type, market_regime, _, key_levels = find_trade_candidate(htf_df_analyzed, ltf_df_analyzed)

        if signal_type:
            final_response = get_ai_confluence_vetting(pair, signal_type, market_regime, key_levels, ltf_df_analyzed)
            
            if final_response:
                # اضافه کردن زمان انقضا به صورت دستی چون در پرامپت اصلی نیست
                final_response_with_exp = final_response.strip() + "\nExpiration: 6 hours"
                final_signals.append(final_response_with_exp)
                signal_cache[pair] = datetime.now(UTC).isoformat()
        else:
            logging.info(f"هیچ کاندیدای معامله‌ای بر اساس استراتژی‌های فعلی برای {pair} یافت نشد.")

        logging.info(f"... تحلیل {pair} تمام شد. تاخیر ۱۰ ثانیه‌ای ...")
        time.sleep(10)

    if final_signals:
        output_content = "Confluence-Based Signals (Technical + Fundamental + Price Action)\n" + "="*60 + "\n\n"
        output_content += "\n---\n".join(final_signals)
        with open("trade_signal.txt", "w", encoding="utf-8") as f:
            f.write(output_content)
        logging.info(f"عملیات موفقیت‌آمیز بود. {len(final_signals)} سیگنال نهایی در trade_signal.txt ذخیره شد.")
    else:
        logging.info("در این اجرا، هیچ سیگنال مورد تأییدی یافت نشد.")

    save_cache(signal_cache)
    logging.info("================== پایان اجرای اسکریپت ==================")

if __name__ == "__main__":
    main()
