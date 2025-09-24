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

# لطفاً کلیدهای API خود را به عنوان متغیر محیطی در سیستم خود تنظیم کنید.
# در ویندوز: set GOOGLE_API_KEY=your_key
# در لینوکس/مک: export GOOGLE_API_KEY=your_key
google_api_key = os.getenv("GOOGLE_API_KEY")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")

if not all([google_api_key, TWELVEDATA_API_KEY]):
    raise ValueError("لطفاً کلیدهای API را به عنوان متغیر محیطی تنظیم کنید: GOOGLE_API_KEY, TWELVEDATA_API_KEY")

# فعال‌سازی ابزار جستجوی گوگل برای مدل
# این بخش حیاتی است و به مدل اجازه می‌دهد از گوگل استفاده کند
try:
    tools = [google.generativeai.Tool(
        google_search=google.generativeai.GoogleSearch()
    )]
except ImportError:
    logging.warning("SDK شما از google_search پشتیبانی نمی‌کند. لطفا آپدیت کنید. AI بدون دسترسی به اخبار زنده کار خواهد کرد.")
    tools = []


genai.configure(api_key=google_api_key)

# تنظیمات اصلی سیستم تحلیل
HIGH_TIMEFRAME = "4h"
LOW_TIMEFRAME = "1h"
CANDLES_TO_FETCH = 300
CURRENCY_PAIRS_TO_ANALYZE = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
    "GBP/JPY", "EUR/JPY", "AUD/JPY", "NZD/USD", "USD/CAD",
    "EUR/GBP", "AUD/NZD", "EUR/AUD", "GBP/CHF", "CAD/JPY"
]
CACHE_FILE = "signal_cache.json"
CACHE_DURATION_HOURS = 4
LOG_FILE = "trading_log.log"
# برای استفاده از ابزار جستجو، یکی از مدل‌های جدیدتر توصیه می‌شود
AI_MODEL_NAME = 'gemini-1.5-pro-latest'

# راه‌اندازی سیستم لاگ‌گیری
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()])

# =================================================================================
# --- توابع کمکی و ابزارها ---
# =================================================================================

def normalize_pair_format(pair_string):
    """فرمت جفت ارز را استاندارد می‌کند (مثال: eurusd -> EUR/USD)"""
    if not isinstance(pair_string, str): return ""
    pair_string = pair_string.upper().strip().replace(" ", "")
    if "/" in pair_string: return pair_string
    if len(pair_string) == 6: return f"{pair_string[:3]}/{pair_string[3:]}"
    logging.warning(f"فرمت جفت ارز '{pair_string}' ناشناخته است.")
    return pair_string

def load_cache():
    """بارگذاری کش سیگنال‌ها از فایل جیسون"""
    if not os.path.exists(CACHE_FILE): return {}
    try:
        with open(CACHE_FILE, 'r') as f: return json.load(f)
    except (json.JSONDecodeError, IOError): return {}

def save_cache(cache):
    """ذخیره کش سیگنال‌ها در فایل جیسون"""
    with open(CACHE_FILE, 'w') as f: json.dump(cache, f, indent=4)

def is_pair_on_cooldown(pair, cache):
    """بررسی می‌کند آیا برای یک جفت ارز به تازگی سیگنال صادر شده است یا خیر"""
    if pair not in cache: return False
    last_signal_time = datetime.fromisoformat(cache[pair])
    if datetime.now(UTC) - last_signal_time < timedelta(hours=CACHE_DURATION_HOURS):
        logging.info(f"جفت ارز {pair} در دوره استراحت (cooldown) قرار دارد. از تحلیل مجدد صرف‌نظر می‌شود.")
        return True
    return False

# =================================================================================
# --- توابع اصلی سیستم ---
# =================================================================================

def get_market_data(symbol, interval, outputsize, retries=3):
    """دریافت دیتا از TwelveData API با قابلیت تلاش مجدد."""
    logging.info(f"دریافت {outputsize} کندل {interval} برای {symbol}...")
    url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={TWELVEDATA_API_KEY}'
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            data = response.json()
            if 'values' in data and len(data['values']) > 0:
                df = pd.DataFrame(data['values'])
                df = df.iloc[::-1].reset_index(drop=True)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
                return df
        except requests.exceptions.RequestException as e:
            logging.error(f"خطا در دریافت دیتا برای {symbol} (تلاش {attempt + 1}/{retries}): {e}")
            time.sleep(5)
    return None

def apply_full_technical_indicators(df):
    """محاسبه مجموعه‌ای جامع از اندیکاتورهای تکنیکال و الگوهای کندل استیک."""
    if df is None or df.empty or len(df) < 50:
        logging.warning("دیتافریم برای محاسبه اندیکاتورها بسیار کوچک یا خالی است.")
        return None
        
    logging.info("محاسبه اندیکاتورهای جامع و الگوهای کندل استیک...")
    try:
        df.ta.ema(length=21, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.ichimoku(append=True)
        df['sup'] = df['low'].rolling(window=20, min_periods=5).min().shift(1)
        df['res'] = df['high'].rolling(window=20, min_periods=5).max().shift(1)
        df.ta.cdl_pattern(name="all", append=True)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logging.error(f"خطا در هنگام محاسبه اندیکاتورهای تکنیکال: {e}")
        return None

def find_last_candlestick_pattern(df_row):
    """آخرین الگوی کندل استیک معنادار را از بین تمام الگوهای شناسایی‌شده پیدا می‌کند."""
    cdl_cols = [col for col in df_row.index if col.startswith('CDL_')]
    for col in reversed(cdl_cols):
        if df_row[col] != 0:
            pattern_name = col.replace("CDL_", "")
            direction = "Bullish" if df_row[col] > 0 else "Bearish"
            return f"{pattern_name} ({direction})"
    return "No significant pattern"

def gather_technical_briefing(symbol, htf_df, ltf_df):
    """تهیه یک گزارش جامع و ساختاریافته از وضعیت تکنیکال بازار برای ارسال به AI."""
    if htf_df.empty or ltf_df.empty: return "داده‌های تکنیکال برای تهیه گزارش کافی نیست."
    last_htf, last_ltf = htf_df.iloc[-1], ltf_df.iloc[-1]
    htf_trend = "Uptrend" if last_htf['EMA_21'] > last_htf['EMA_50'] > last_htf['EMA_200'] else "Downtrend" if last_htf['EMA_21'] < last_htf['EMA_50'] < last_htf['EMA_200'] else "Sideways"
    market_regime = "Strong Trend" if last_htf.get('ADX_14', 0) > 25 else "Weak/Ranging Trend"
    price = last_ltf['close']
    kumo_a, kumo_b = last_ltf.get('ISA_9', price), last_ltf.get('ISB_26', price)
    ichimoku_status = "Bullish (Above Kumo)" if price > kumo_a and price > kumo_b else "Bearish (Below Kumo)" if price < kumo_a and price < kumo_b else "Inside Kumo (Unclear)"
    last_pattern = find_last_candlestick_pattern(ltf_df.iloc[-2])

    briefing = f"""### Technical Briefing for {symbol}
- **HTF ({HIGH_TIMEFRAME}) Context:** Trend is **{htf_trend}** ({market_regime}, ADX: {last_htf.get('ADX_14', 0):.2f}). HTF Support: **{last_htf.get('sup', 0):.5f}**, HTF Resistance: **{last_htf.get('res', 0):.5f}**.
- **LTF ({LOW_TIMEFRAME}) Analysis:** Current Price: {price:.5f}. LTF Support: **{last_ltf.get('sup', 0):.5f}**, LTF Resistance: **{last_ltf.get('res', 0):.5f}**.
- **Key Indicators:** Ichimoku is **{ichimoku_status}**. MACD is {'Bullish' if last_ltf.get('MACDh_12_26_9') > 0 else 'Bearish'}. RSI is {last_ltf.get('RSI_14'):.2f}.
- **Price Action:** Last significant candle pattern was **{last_pattern}**.
- **Volatility (ATR):** **{last_ltf.get('ATRr_14', 0):.5f}**. Use this to validate SL distance."""
    return briefing.strip()

def get_ai_trade_decision(symbol, technical_briefing):
    """ارسال گزارش به AI برای تحلیل سه‌گانه: تکنیکال، اخبار زنده و مدیریت ریسک."""
    base_currency, quote_currency = symbol.split('/')
    prompt = f"""You are a "Live Market Strategist" at a hedge fund. Your primary job is to find high-probability trades by synthesizing three pillars of analysis: **1. Static Technical Data, 2. Live Fundamental News, and 3. Robust Risk Management.**

**Your Mandatory 3-Step Workflow:**

**Step 1: Analyze the Provided Technical Briefing.**
- First, deeply analyze the technical data provided below for **{symbol}**. What is the primary story the chart is telling you?
    {technical_briefing}

 
**Step 2: Conduct Live Fundamental & Sentiment Analysis (CRITICAL).**
- **You MUST use your Google Search tool.** Search for the latest (last 24-48 hours) high-impact news, central bank statements, key economic data releases (inflation, jobs, GDP), and overall market sentiment for both `{base_currency}` and `{quote_currency}`.
- Summarize your findings. Is the recent news flow creating a headwind or a tailwind for the technical setup? Is sentiment bullish, bearish, or mixed?

**Step 3: Synthesize and Formulate a Trade Plan.**
- **Confluence Check:** Is there a powerful alignment between the technical picture (from Step 1) and the live news sentiment (from Step 2)?
- **Risk Management:** Define a precise Stop Loss just beyond a key support/resistance level identified in the briefing. Validate that this SL is at a reasonable distance (e.g., >1.5x ATR) to avoid noise. Then, define a Take Profit at the next logical key level, ensuring at least a 1:1.5 Risk:Reward ratio.
- **Final Decision:**
    - If there is strong confluence across all three pillars, construct the trade plan in the required JSON format.
    - If there is a major conflict (e.g., bullish chart but very bearish live news, or poor R:R), you MUST return `NO_SIGNAL`.

**STRICT OUTPUT FORMAT (Provide ONLY the JSON or the words NO_SIGNAL):**
- If no trade: `NO_SIGNAL`
- If trade is found:
```json
{{
  "PAIR": "{symbol}",
  "TYPE": "[BUY_LIMIT, SELL_LIMIT, BUY_STOP, SELL_STOP]",
  "ENTRY": "[Precise Entry Price]",
  "SL": "[Precise Stop Loss Price, justified by a key S/R level]",
  "TP": "[Precise Take Profit Price, targeting the next key level]",
  "CONFIDENCE": "[Score from 1-10, based on the strength of confluence]",
  "REASONING": {{
    "TECHNICAL_VIEW": "[Your summary of the technical analysis.]",
    "LIVE_NEWS_SENTIMENT": "[Your summary of the live news search and how it supports the trade.]",
    "RISK_REWARD_ANALYSIS": "[Confirmation of SL/TP logic and favorable R:R ratio.]"
  }}
}}
```"""
    logging.info(f"ارسال پرونده تحلیلی {symbol} به استراتژیست زنده (AI) برای تحلیل سه‌گانه...")
    try:
        model = genai.GenerativeModel(AI_MODEL_NAME, tools=tools)
        response = model.generate_content(prompt.strip(), request_options={'timeout': 300})
        
        text_response = response.text.strip()
        
        if text_response.upper() == "NO_SIGNAL":
            logging.info(f"AI پس از تحلیل زنده، هیچ فرصت مناسبی برای {symbol} پیدا نکرد.")
            return None
            
        # Regex to find JSON block, even with surrounding text
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text_response, re.DOTALL)
        if not json_match:
            json_match = re.search(r'(\{.*?\})', text_response, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)
            logging.info(f"AI یک سیگنال معاملاتی با تحلیل جامع برای {symbol} ایجاد کرد.")
            return json_str

        logging.warning(f"پاسخ AI برای {symbol} در فرمت مورد انتظار نبود: {text_response}")
        return None

    except Exception as e:
        logging.error(f"خطا در تصمیم‌گیری نهایی AI برای {symbol}: {e}")
        return None

def main():
    logging.info("================== شروع اجرای اسکریپت (مدل استراتژیست زنده v3) ==================")
    signal_cache = load_cache()
    final_signals_json = []

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, help="یک جفت ارز مشخص را برای تحلیل وارد کنید")
    args, _ = parser.parse_known_args()
    pairs_to_run = [args.pair] if args.pair else CURRENCY_PAIRS_TO_ANALYZE

    for pair_raw in pairs_to_run:
        pair = normalize_pair_format(pair_raw)
        if not pair or is_pair_on_cooldown(pair, signal_cache):
            time.sleep(1)
            continue

        logging.info(f"--- شروع تحلیل جامع برای: {pair} ---")
        
        htf_df = get_market_data(pair, HIGH_TIMEFRAME, CANDLES_TO_FETCH)
        ltf_df = get_market_data(pair, LOW_TIMEFRAME, CANDLES_TO_FETCH)

        if htf_df is None or ltf_df is None: 
            logging.warning(f"دیتای بازار برای {pair} دریافت نشد. به جفت ارز بعدی می‌رویم.")
            continue

        htf_df_analyzed = apply_full_technical_indicators(htf_df)
        ltf_df_analyzed = apply_full_technical_indicators(ltf_df)

        if not isinstance(htf_df_analyzed, pd.DataFrame) or htf_df_analyzed.empty or not isinstance(ltf_df_analyzed, pd.DataFrame) or ltf_df_analyzed.empty:
            logging.warning(f"دیتای کافی برای تحلیل {pair} پس از پردازش اندیکاتورها وجود ندارد.")
            continue
            
        technical_briefing = gather_technical_briefing(pair, htf_df_analyzed, ltf_df_analyzed)
        logging.info(f"پرونده تحلیلی برای {pair} تهیه شد:\n{technical_briefing}")
        
        final_response_json_str = get_ai_trade_decision(pair, technical_briefing)
        
        if final_response_json_str:
            try:
                signal_data = json.loads(final_response_json_str)
                final_signals_json.append(signal_data)
                signal_cache[pair] = datetime.now(UTC).isoformat()
            except json.JSONDecodeError as e:
                logging.error(f"خطا در پارس کردن پاسخ JSON از AI برای {pair}: {e}\nپاسخ دریافتی: {final_response_json_str}")

        logging.info(f"... تحلیل {pair} تمام شد. تاخیر ۱۰ ثانیه‌ای برای مدیریت نرخ API ...")
        time.sleep(10)

    if final_signals_json:
        with open("trade_signals_live.json", "w", encoding="utf-8") as f:
            json.dump(final_signals_json, f, indent=2, ensure_ascii=False)
        logging.info(f"عملیات موفقیت‌آمیز بود. {len(final_signals_json)} سیگنال نهایی در trade_signals_live.json ذخیره شد.")
    else:
        logging.info("در این اجرا، هوش مصنوعی هیچ سیگنال قابل معامله‌ای با همگرایی بالا بین عوامل پیدا نکرد.")

    save_cache(signal_cache)
    logging.info("================== پایان اجرای اسکریپت ==================")

if __name__ == "__main__":
    main()
