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

# FIX: Only two API keys are now needed
google_api_key = os.getenv("GOOGLE_API_KEY")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")

if not all([google_api_key, TWELVEDATA_API_KEY]):
    raise ValueError("لطفاً کلیدهای API را تنظیم کنید: GOOGLE_API_KEY, TWELVEDATA_API_KEY")

genai.configure(api_key=google_api_key)

# --- All other settings remain the same ---
HIGH_TIMEFRAME = "4h"
LOW_TIMEFRAME = "1h"
CANDLES_TO_FETCH = 200
ADX_TREND_THRESHOLD = 25
ADX_RANGE_THRESHOLD = 20
AI_DEEP_ANALYSIS_CONFIDENCE_THRESHOLD = 9
CURRENCY_PAIRS_TO_ANALYZE = ["EUR/USD"]
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

# ✨ NEW: Function to check for news using the Gemini AI itself ✨
def check_news_with_ai(pair):
    """Uses Gemini to check for high-impact economic news."""
    logging.info(f"استفاده از هوش مصنوعی برای بررسی تقویم اقتصادی برای {pair}...")
    try:
        base_currency, quote_currency = pair.split('/')
        
        # A highly specific prompt to force a reliable YES/NO answer
        prompt = f"""
        You are a financial data verification system. Your only task is to check for high-impact economic news.
        Search reliable economic calendars (like Forex Factory, DailyFX) for the next 4 hours from now.
        Are there any 'high' impact news events scheduled for the currencies '{base_currency}' or '{quote_currency}' in this timeframe?

        Respond ONLY with the word "YES" if there is at least one high-impact event.
        Respond ONLY with the word "NO" if there are no high-impact events.
        Do not provide any other text, explanation, or formatting.
        """
        
        model = genai.GenerativeModel('gemini-2.5-flash') # Using a fast model for this simple task
        response = model.generate_content(prompt.strip(), request_options={'timeout': 120})
        
        response_text = response.text.strip().upper()
        
        if "YES" in response_text:
            logging.warning(f"هوش مصنوعی یک خبر مهم برای {pair} در ساعات آینده شناسایی کرد. تحلیل متوقف شد.")
            return True
        elif "NO" in response_text:
            logging.info(f"هوش مصنوعی خبر مهمی برای {pair} در پیش‌بینی نکرد.")
            return False
        else:
            # If the AI gives an unexpected response, assume there might be news to be safe.
            logging.warning(f"پاسخ غیرمنتظره‌ای از AI برای بررسی اخبار دریافت شد: '{response.text}'. برای احتیاط، تحلیل متوقف می‌شود.")
            return True

    except Exception as e:
        logging.error(f"خطا در هنگام بررسی اخبار با هوش مصنوعی: {e}")
        # To be safe in case of an error, assume there IS news.
        return True

def get_market_data(symbol, interval, outputsize):
    # This function remains unchanged
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
    # This function remains unchanged
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
        
        if df.empty:
            logging.warning("دیتافریم پس از محاسبه اندیکاتورها و حذف مقادیر خالی، تهی شد.")

    except Exception as e:
        logging.error(f"خطا در هنگام محاسبه اندیکاتورهای تکنیکال: {e}")
        return pd.DataFrame()
    return df


# --- The rest of the functions (detect_market_regime, find_trade_candidate, AI functions) remain unchanged ---
# They are included here for completeness.

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
        if 'MACD_12_26_9' not in last_ltf or 'MACDs_12_26_9' not in last_ltf:
            logging.warning("ستون‌های MACD برای تحلیل روند یافت نشد.")
            return None, None, None, None
        
        htf_trend = "UPTREND" if htf_df.iloc[-1]['EMA_21'] > htf_df.iloc[-1]['EMA_50'] else "DOWNTREND"
        if htf_trend == "UPTREND" and last_ltf['EMA_21'] > last_ltf['EMA_50'] and last_ltf['MACD_12_26_9'] > last_ltf['MACDs_12_26_9'] and volume_confirmed:
            return "BUY_TREND", market_regime, last_ltf, (htf_support, htf_resistance)
        if htf_trend == "DOWNTREND" and last_ltf['EMA_21'] < last_ltf['EMA_50'] and last_ltf['MACD_12_26_9'] < last_ltf['MACDs_12_26_9'] and volume_confirmed:
            return "SELL_TREND", market_regime, last_ltf, (htf_support, htf_resistance)

    elif market_regime == "RANGING":
        if 'BBL_20_2.0' not in last_ltf or 'BBU_20_2.0' not in last_ltf:
            logging.warning("ستون‌های Bollinger Bands برای تحلیل رنج یافت نشد.")
            return None, None, None, None
            
        if last_ltf['close'] <= last_ltf['BBL_20_2.0'] and last_ltf['RSI_14'] < 35:
            return "BUY_RANGE", market_regime, last_ltf, (htf_support, htf_resistance)
        if last_ltf['close'] >= last_ltf['BBU_20_2.0'] and last_ltf['RSI_14'] > 65:
            return "SELL_RANGE", market_regime, last_ltf, (htf_support, htf_resistance)
            
    return None, None, None, None

def get_ai_initial_confirmation(symbol, signal_type, market_regime, key_levels, ltf_df):
    strategy_name = "Trend Following (Volume Confirmed)" if market_regime == "TRENDING" else "Mean Reversion"
    last_candle = ltf_df.iloc[-1]
    prompt = f"""
    As a primary analyst, validate this trade signal proposed by a quantitative system.
    - Asset: {symbol}, Market Regime: {market_regime}, Strategy: {strategy_name}, Signal: {signal_type}
    - Key HTF Support: {key_levels[0]:.5f}, Key HTF Resistance: {key_levels[1]:.5f}
    - Last Close: {last_candle['close']}
    Task: Provide a quick validation. If you CONFIRM, give parameters. If REJECT, respond only with "REJECT: [reason]".
    
    Strict Output Format (on CONFIRMATION only):
    PAIR: {symbol}
    TYPE: [Order Type]
    ENTRY: [Entry Price]
    SL: [Stop Loss Price]
    TP: [Take Profit Price]
    CONFIDENCE: [Score from 1-10]
    REASON: [Concise reason]
    """
    logging.info(f"ارسال سیگنال {symbol} به AI برای تأییدیه اولیه...")
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt.strip(), request_options={'timeout': 180})
        if "REJECT" in response.text.upper():
            logging.warning(f"هوش مصنوعی سیگنال اولیه {symbol} را رد کرد.")
            return None, 0
        
        confidence_match = re.search(r"CONFIDENCE:\s*(\d+)", response.text)
        confidence = int(confidence_match.group(1)) if confidence_match else 0
        logging.info(f"تأییدیه اولیه برای {symbol} با امتیاز {confidence} دریافت شد.")
        return response.text, confidence
    except Exception as e:
        logging.error(f"خطا در تأییدیه اولیه AI: {e}")
        return None, 0

def get_ai_deep_analysis(initial_response_text):
    logging.info("سیگنال با امتیاز بالا یافت شد! ارسال برای تحلیل عمیق و روایت‌سازی...")
    prompt = f"""
    You are the Head of Trading. A junior analyst has confirmed the following high-probability trade. Your job is to provide the final narrative and refine the parameters.
    
    **Junior Analyst's Confirmed Signal:**
    {initial_response_text}

    **Your Task:**
    1.  **Craft a Trade Narrative:** In 3 bullet points, explain the core thesis.
    2.  **Refine Parameters:** Based on your senior expertise, provide the final, optimized SL and TP.
    3.  **Combine and Finalize:** Re-write the full signal, embedding your narrative within it.

    **Strict Final Output Format:**
    [Copy the initial PAIR, TYPE, ENTRY, SL, TP, CONFIDENCE, REASON lines exactly]
    ---
    **SENIOR ANALYST NARRATIVE:**
    * **Technical Thesis:** [Your analysis of the pattern]
    * **Fundamental View:** [Your view on fundamentals/sentiment]
    * **Primary Risk:** [The main risk to the trade]
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt.strip(), request_options={'timeout': 180})
        logging.info("تحلیل عمیق و روایت با موفقیت دریافت شد.")
        return response.text
    except Exception as e:
        logging.error(f"خطا در تحلیل عمیق AI: {e}")
        return initial_response_text


# =================================================================================
# --- حلقه اصلی برنامه ---
# =================================================================================

def main():
    """تابع اصلی برای اجرای کامل فرآیند تولید سیگنال."""
    logging.info("================== شروع اجرای اسکریپت تحلیلگر ارشد ==================")
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
        # FIX: Call the new AI-based news check function
        if is_pair_on_cooldown(pair, signal_cache) or check_news_with_ai(pair):
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
            initial_response, confidence = get_ai_initial_confirmation(pair, signal_type, market_regime, key_levels, ltf_df_analyzed)
            if initial_response:
                final_response = initial_response
                if confidence >= AI_DEEP_ANALYSIS_CONFIDENCE_THRESHOLD:
                    final_response = get_ai_deep_analysis(initial_response)
                
                final_signals.append(final_response.strip())
                signal_cache[pair] = datetime.now(UTC).isoformat()
        else:
            logging.info(f"هیچ کاندیدای معامله‌ای بر اساس استراتژی‌های فعلی برای {pair} یافت نشد.")

        logging.info(f"... تحلیل {pair} تمام شد. تاخیر ۱۰ ثانیه‌ای ...")
        time.sleep(10)

    if final_signals:
        output_content = "Senior Analyst Grade Signals (Adaptive + 2-Step AI)\n" + "="*60 + "\n\n"
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
