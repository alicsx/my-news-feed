import google.generativeai as genai
import os
import re
import time
import json
import logging
import pandas as pd
import pandas_ta as ta
import requests
from datetime import datetime, timedelta

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
AI_DEEP_ANALYSIS_CONFIDENCE_THRESHOLD = 9 # حداقل امتیاز برای ارسال به مرحله دوم تحلیل AI

CURRENCY_PAIRS_TO_ANALYZE = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
    "GBP/JPY", "EUR/JPY", "AUD/JPY", "NZD/USD", "USD/CAD"
]

# --- تنظیمات فنی ---
CACHE_FILE = "signal_cache.json"
CACHE_DURATION_HOURS = 4
LOG_FILE = "trading_log.log"

# --- راه‌اندازی سیستم لاگ‌برداری ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()])

# =================================================================================
# --- توابع اصلی سیستم ---
# =================================================================================

# توابع load_cache, save_cache, is_pair_on_cooldown, has_high_impact_news بدون تغییر
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
    if datetime.utcnow() - last_signal_time < timedelta(hours=CACHE_DURATION_HOURS):
        logging.info(f"جفت ارز {pair} در حافظه کش است (cooldown).")
        return True
    return False

def has_high_impact_news(pair, api_key):
    logging.info(f"بررسی تقویم اقتصادی Twelve Data برای {pair}...")
    try:
        base_currency, quote_currency = pair.split('/')
        country_map = {'USD': 'United States', 'EUR': 'Euro Zone', 'GBP': 'United Kingdom', 'JPY': 'Japan', 'CHF': 'Switzerland', 'AUD': 'Australia', 'NZD': 'New Zealand', 'CAD': 'Canada'}
        countries_to_check = [country for c, country in country_map.items() if c in [base_currency, quote_currency]]
        if not countries_to_check: return False
        
        today = datetime.utcnow().strftime('%Y-%m-%d')
        end_date = (datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%d')
        country_params = ",".join(countries_to_check)
        url = f"https://api.twelvedata.com/economic_calendar?country={country_params}&start_date={today}&end_date={end_date}&apikey={api_key}"
        res = requests.get(url, timeout=20).json()

        if 'events' not in res or not res['events']: return False
            
        for event in res['events']:
            event_time = datetime.fromisoformat(event['date'].replace("Z", "+00:00"))
            time_until_event = event_time - datetime.now(event_time.tzinfo)

            if timedelta(minutes=-30) < time_until_event < timedelta(hours=4) and event.get('importance') == 'High':
                logging.warning(f"خبر مهم '{event['event']}' برای {pair} در راه است! تحلیل متوقف شد.")
                return True
        return False
    except Exception as e:
        logging.error(f"خطا در بررسی تقویم اقتصادی Twelve Data: {e}")
        return False

def get_market_data(symbol, interval, outputsize):
    logging.info(f"دریافت {outputsize} کندل {interval} برای {symbol}...")
    # توجه: Twelve Data در پلن رایگان، ممکن است حجم معاملات را برنگرداند
    url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={TWELVEDATA_API_KEY}'
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if 'values' in data and len(data['values']) > 0:
            df = pd.DataFrame(data['values'])
            df = df.apply(pd.to_numeric, errors='ignore')
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime', ascending=True).reset_index(drop=True)
            return df
    except Exception as e:
        logging.error(f"خطا در دریافت دیتای بازار برای {symbol}: {e}")
    return None

def apply_full_technical_indicators(df):
    """مجموعه کامل اندیکاتورها شامل حجم و سطوح کلیدی را محاسبه می‌کند."""
    if df is None or df.empty: return None
    logging.info("محاسبه اندیکاتورهای جامع (EMA, MACD, RSI, ATR, ADX, BBands, Volume)...")
    df.ta.ema(length=21, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    if 'volume' in df.columns:
        df.ta.sma(close=df['volume'], length=20, prefix="VOLUME", append=True)
    
    # شناسایی سطوح حمایت و مقاومت کلیدی
    df['sup'] = df['low'].rolling(window=10, min_periods=3).min()
    df['res'] = df['high'].rolling(window=10, min_periods=3).max()
    
    df.dropna(inplace=True)
    return df

def detect_market_regime(df):
    if df is None or 'ADX_14' not in df.columns: return "UNCLEAR"
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
    
    volume_confirmed = 'VOLUME_SMA_20' in last_ltf and last_ltf['volume'] > last_ltf['VOLUME_SMA_20']

    if market_regime == "TRENDING":
        htf_trend = "UPTREND" if htf_df.iloc[-1]['EMA_21'] > htf_df.iloc[-1]['EMA_50'] else "DOWNTREND"
        if htf_trend == "UPTREND" and last_ltf['EMA_21'] > last_ltf['EMA_50'] and volume_confirmed:
            return "BUY_TREND", market_regime, last_ltf, (htf_support, htf_resistance)
        if htf_trend == "DOWNTREND" and last_ltf['EMA_21'] < last_ltf['EMA_50'] and volume_confirmed:
            return "SELL_TREND", market_regime, last_ltf, (htf_support, htf_resistance)

    elif market_regime == "RANGING":
        if last_ltf['close'] <= last_ltf['BBL_20_2.0'] and last_ltf['RSI_14'] < 35:
            return "BUY_RANGE", market_regime, last_ltf, (htf_support, htf_resistance)
        if last_ltf['close'] >= last_ltf['BBU_20_2.0'] and last_ltf['RSI_14'] > 65:
            return "SELL_RANGE", market_regime, last_ltf, (htf_support, htf_resistance)
            
    return None, None, None, None

def get_ai_initial_confirmation(symbol, signal_type, market_regime, key_levels, ltf_df):
    """مرحله اول: دریافت تأییدیه اولیه از هوش مصنوعی."""
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
    # ... (کد ارتباط با Gemini مانند قبل)
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(prompt.strip(), request_options={'timeout': 180})
        if "REJECT" in response.text.upper(): return None, 0
        
        confidence_match = re.search(r"CONFIDENCE:\s*(\d+)", response.text)
        confidence = int(confidence_match.group(1)) if confidence_match else 0
        logging.info(f"تأییدیه اولیه برای {symbol} با امتیاز {confidence} دریافت شد.")
        return response.text, confidence
    except Exception as e:
        logging.error(f"خطا در تأییدیه اولیه AI: {e}")
        return None, 0


def get_ai_deep_analysis(initial_response_text):
    """مرحله دوم: برای سیگنال‌های با امتیاز بالا، تحلیل عمیق و روایت‌سازی درخواست می‌شود."""
    logging.info("سیگنال با امتیاز بالا یافت شد! ارسال برای تحلیل عمیق و روایت‌سازی...")
    prompt = f"""
    You are the Head of Trading. A junior analyst has confirmed the following high-probability trade. Your job is to provide the final narrative and refine the parameters.
    
    **Junior Analyst's Confirmed Signal:**
    {initial_response_text}

    **Your Task:**
    1.  **Craft a Trade Narrative:** In 3 bullet points, explain the core thesis. What is the main technical pattern? Is there a fundamental driver? What is the primary risk to this trade?
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
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(prompt.strip(), request_options={'timeout': 180})
        logging.info("تحلیل عمیق و روایت با موفقیت دریافت شد.")
        return response.text
    except Exception as e:
        logging.error(f"خطا در تحلیل عمیق AI: {e}")
        return initial_response_text # در صورت خطا، همان پاسخ اولیه را برگردان

# =================================================================================
# --- حلقه اصلی برنامه ---
# =================================================================================

def main():
    """تابع اصلی برای اجرای کامل فرآیند تولید سیگنال."""
    logging.info("================== شروع اجرای اسکریپت تحلیلگر ارشد ==================")
    signal_cache = load_cache()
    final_signals = []

    for pair in CURRENCY_PAIRS_TO_ANALYZE:
        logging.info(f"--- شروع تحلیل جامع برای: {pair} ---")
        if is_pair_on_cooldown(pair, signal_cache) or has_high_impact_news(pair, TWELVEDATA_API_KEY):
            time.sleep(2); continue

        htf_df = get_market_data(pair, HIGH_TIMEFRAME, CANDLES_TO_FETCH)
        htf_df_analyzed = apply_full_technical_indicators(htf_df)
        
        ltf_df = get_market_data(pair, LOW_TIMEFRAME, CANDLES_TO_FETCH)
        ltf_df_analyzed = apply_full_technical_indicators(ltf_df)

        if htf_df_analyzed is None or ltf_df_analyzed is None: continue
            
        signal_type, market_regime, _, key_levels = find_trade_candidate(htf_df_analyzed, ltf_df_analyzed)

        if signal_type:
            initial_response, confidence = get_ai_initial_confirmation(pair, signal_type, market_regime, key_levels, ltf_df_analyzed)
            if initial_response:
                final_response = initial_response
                if confidence >= AI_DEEP_ANALYSIS_CONFIDENCE_THRESHOLD:
                    final_response = get_ai_deep_analysis(initial_response)
                
                # اضافه کردن زمان انقضا به پاسخ نهایی
                final_response_with_exp = final_response.strip() + "\nExpiration: 6 hours"
                final_signals.append(final_response_with_exp)
                signal_cache[pair] = datetime.utcnow().isoformat()
        else:
            logging.info(f"هیچ کاندیدای معامله‌ای بر اساس استراتژی‌های فعلی برای {pair} یافت نشد.")

        logging.info(f"... تحلیل {pair} تمام شد. تاخیر ۱۰ ثانیه‌ای ...")
        time.sleep(10)

    # ذخیره سیگنال‌های نهایی و حافظه
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
