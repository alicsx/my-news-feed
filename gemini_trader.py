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
        df.dropna(inplace=True)
        if df.empty: logging.warning("دیتافریم پس از محاسبه اندیکاتورها تهی شد.")
        return df
    except Exception as e:
        logging.error(f"خطا در هنگام محاسبه اندیکاتورهای تکنیکال: {e}")
        return pd.DataFrame()

# ✨ NEW: Function to score different trade setups ✨
def score_trade_setups(htf_df, ltf_df):
    """به تمام موقعیت‌های بالقوه (روند و رنج) امتیاز می‌دهد."""
    scores = {}
    if htf_df is None or ltf_df is None or htf_df.empty or ltf_df.empty:
        return scores
        
    last_htf = htf_df.iloc[-1]
    last_ltf = ltf_df.iloc[-1]
    
    # امتیازدهی به استراتژی روند
    if all(k in last_htf and k in last_ltf for k in ['ADX_14', 'EMA_21', 'EMA_50', 'MACD_12_26_9', 'MACDs_12_26_9']):
        score = 0
        if last_htf['ADX_14'] > 20: score += last_htf['ADX_14'] / 10 # امتیاز بر اساس قدرت روند
        # روند صعودی
        if last_htf['EMA_21'] > last_htf['EMA_50'] and last_ltf['EMA_21'] > last_ltf['EMA_50'] and last_ltf['MACD_12_26_9'] > last_ltf['MACDs_12_26_9']:
            score += 5
            scores['BUY_TREND'] = {'score': score, 'last_candle': last_ltf, 'key_levels': (last_htf['sup'], last_htf['res'])}
        # روند نزولی
        if last_htf['EMA_21'] < last_htf['EMA_50'] and last_ltf['EMA_21'] < last_ltf['EMA_50'] and last_ltf['MACD_12_26_9'] < last_ltf['MACDs_12_26_9']:
            score += 5
            scores['SELL_TREND'] = {'score': score, 'last_candle': last_ltf, 'key_levels': (last_htf['sup'], last_htf['res'])}

    # امتیازدهی به استراتژی رنج
    if all(k in last_htf and k in last_ltf for k in ['ADX_14', 'BBL_20_2.0', 'BBU_20_2.0', 'RSI_14']):
        score = 0
        if last_htf['ADX_14'] < 25: score += (25 - last_htf['ADX_14']) # امتیاز بر اساس ضعف روند
        # خرید در کف کانال
        if last_ltf['close'] <= last_ltf['BBL_20_2.0'] and last_ltf['RSI_14'] < 35:
            score += (35 - last_ltf['RSI_14']) / 5 # امتیاز بر اساس شدت اشباع فروش
            scores['BUY_RANGE'] = {'score': score, 'last_candle': last_ltf, 'key_levels': (last_htf['sup'], last_htf['res'])}
        # فروش در سقف کانال
        if last_ltf['close'] >= last_ltf['BBU_20_2.0'] and last_ltf['RSI_14'] > 65:
            score += (last_ltf['RSI_14'] - 65) / 5 # امتیاز بر اساس شدت اشباع خرید
            scores['SELL_RANGE'] = {'score': score, 'last_candle': last_ltf, 'key_levels': (last_htf['sup'], last_htf['res'])}
            
    return scores

def get_ai_final_vetting(best_candidate):
    """از هوش مصنوعی برای ارزیابی نهایی 'بهترین کاندیدای' شناسایی شده استفاده می‌کند."""
    
    # استخراج اطلاعات از بهترین کاندیدا
    symbol = best_candidate['pair']
    signal_type = best_candidate['signal_type']
    quant_score = best_candidate['score']
    market_regime = "TRENDING" if "TREND" in signal_type else "RANGING"
    key_levels = best_candidate['data']['key_levels']
    
    base_currency, quote_currency = symbol.split('/')

    prompt = f"""
    You are the Head of Trading. A quantitative model has scanned the entire market and identified the single most promising trade setup right now. Your task is to perform the final, decisive analysis.

    **The System's Best Candidate:**
    - **Asset:** {symbol}
    - **Proposed Signal:** {signal_type}
    - **Underlying Strategy:** {market_regime}
    - **Quantitative Score:** {quant_score:.2f} (This score reflects the strength of the technical indicators.)

    **Your Multi-Faceted Vetting Task:**

    1.  **Fundamental & News Context:**
        - Quickly check for high-impact news for '{base_currency}' or '{quote_currency}' in the next 8 hours.
        - Does the fundamental outlook (sentiment, upcoming news) align with, contradict, or is it neutral to the proposed trade direction?

    2.  **Price Action Context:**
        - Look at the recent price action. Do you see patterns that strongly confirm the signal (e.g., clear rejection wicks for ranging, or a clean breakout for trending)? Or are there warning signs (e.g., indecision candles)?

    3.  **Final Verdict & Execution Plan:**
        - Based on the **confluence** of all three factors (Quant Score, Fundamentals, Price Action), make your final call.
        - If you **CONFIRM**, provide the final, precise parameters.
        - If you **REJECT**, you must state the primary reason for overriding the system's top pick.

    **Strict Output Format:**
    - If REJECTED: `REJECT: [Your reason, e.g., "Quant score is high, but upcoming CPI news creates too much uncertainty."]`
    - If CONFIRMED:
    PAIR: {symbol}
    TYPE: [Order Type]
    ENTRY: [Entry Price]
    SL: [Stop Loss Price]
    TP: [Take Profit Price]
    CONFIDENCE: [Your final confidence score from 1-10]
    REASON: [Your concise reason highlighting the confluence]
    """
    logging.info(f"ارسال 'بهترین کاندیدا' ({symbol} - {signal_type}) با امتیاز {quant_score:.2f} به AI برای ارزیابی نهایی...")
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt.strip(), request_options={'timeout': 180})
        
        if "REJECT" in response.text.upper():
            logging.warning(f"هوش مصنوعی بهترین کاندیدا ({symbol}) را رد کرد. دلیل: {response.text}")
            return None
        
        logging.info(f"هوش مصنوعی بهترین کاندیدا ({symbol}) را به عنوان سیگنال نهایی تأیید کرد.")
        return response.text
    except Exception as e:
        logging.error(f"خطا در ارزیابی نهایی AI: {e}")
        return None

# =================================================================================
# --- حلقه اصلی برنامه ---
# =================================================================================

def main():
    """تابع اصلی برای اجرای کامل فرآیند تولید سیگنال."""
    logging.info("================== شروع اجرای اسکریپت (مدل بهترین کاندیدا) ==================")
    signal_cache = load_cache()
    all_candidates = []

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
            
        # ✨ NEW: Score all potential setups for the current pair ✨
        setups = score_trade_setups(htf_df_analyzed, ltf_df_analyzed)
        if setups:
            for signal_type, data in setups.items():
                all_candidates.append({
                    'pair': pair,
                    'signal_type': signal_type,
                    'score': data['score'],
                    'data': data
                })
                logging.info(f"کاندیدای یافت شده: {pair} - {signal_type} - امتیاز: {data['score']:.2f}")

        logging.info(f"... تحلیل {pair} تمام شد. تاخیر ۱۰ ثانیه‌ای ...")
        time.sleep(10)
    
    # ✨ NEW: Find the single best candidate across all pairs and setups ✨
    if all_candidates:
        best_candidate = max(all_candidates, key=lambda x: x['score'])
        logging.info(f"\n--- بهترین کاندیدای کلی انتخاب شد: {best_candidate['pair']} با سیگنال {best_candidate['signal_type']} و امتیاز {best_candidate['score']:.2f} ---")
        
        # Send the single best candidate for final AI vetting
        final_signal = get_ai_final_vetting(best_candidate)
        
        if final_signal:
            final_signal_with_exp = final_signal.strip() + "\nExpiration: 6 hours"
            output_content = "Best Candidate Signal (Quant-Ranked + AI-Vetted)\n" + "="*60 + "\n\n"
            output_content += final_signal_with_exp
            with open("trade_signal.txt", "w", encoding="utf-8") as f:
                f.write(output_content)
            logging.info(f"عملیات موفقیت‌آمیز بود. سیگنال نهایی در trade_signal.txt ذخیره شد.")
            # Update cache for the pair that generated the signal
            signal_cache[best_candidate['pair']] = datetime.now(UTC).isoformat()
        else:
            logging.info("هوش مصنوعی بهترین کاندیدای انتخاب شده را رد کرد. هیچ سیگنالی صادر نشد.")
            
    else:
        logging.info("در این اجرا، هیچ کاندیدای معامله‌ای در هیچ جفت ارزی یافت نشد.")

    save_cache(signal_cache)
    logging.info("================== پایان اجرای اسکریپت ==================")

if __name__ == "__main__":
    main()
