import google.generativeai as genai
import os
import re
import time
from collections import defaultdict
import requests

# --- بخش تنظیمات ---

# کلید API گوگل از GitHub Secrets خوانده می‌شود
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("کلید API گوگل در GOOGLE_API_KEY یافت نشد.")
genai.configure(api_key=google_api_key)

# ✨ تغییر ۱: کلید API Finnhub از GitHub Secrets خوانده می‌شود ✨
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
if not FINNHUB_API_KEY:
    raise ValueError("کلید API Finnhub در FINNHUB_API_KEY یافت نشد.")

CURRENCY_PAIRS_TO_ANALYZE = [
    "EUR/USD", "GBP/USD", "USD/CHF", "EUR/JPY",
    "AUD/JPY", "GBP/JPY", "EUR/AUD", "NZD/CAD"
]

# --- ✨ تغییر ۲: تابع دریافت قیمت با Finnhub جایگزین شد ✨ ---
def get_finnhub_price(currency_pair, api_key):
    """قیمت لحظه‌ای را با استفاده از API قدرتمند Finnhub دریافت می‌کند."""
    # Finnhub از فرمت OANDA:EUR_USD یا فقط EURUSD استفاده می‌کند. ما / را حذف می‌کنیم.
    symbol = f"OANDA:{currency_pair.replace('/', '_')}"
    url = f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}'
    
    try:
        print(f"\nدر حال دریافت قیمت لحظه‌ای برای {currency_pair} از Finnhub...")
        response = requests.get(url, timeout=20)
        response.raise_for_status() # برای خطاهای HTTP
        
        data = response.json()
        
        # در Finnhub، قیمت فعلی با کلید 'c' مشخص می‌شود
        if 'c' in data and data['c'] != 0:
            price = data['c']
            print(f"قیمت دریافت شد: {price}")
            return float(price)
        else:
            # اگر قیمت صفر باشد یا کلید وجود نداشته باشد، یعنی مشکلی هست
            print(f"پاسخ غیرمنتظره یا قیمت نامعتبر از Finnhub برای {symbol}: {data}")
            return None
            
    except Exception as e:
        print(f"خطا در دریافت قیمت از Finnhub برای {symbol}: {e}")
        return None

# --- تابع ساخت پرامپت (بدون تغییر) ---
def create_single_pair_prompt(currency_pair, current_price):
    """پرامپت اصلی را با استفاده از قیمت لحظه‌ای واقعی ایجاد می‌کند."""
    user_prompt = f"""
    **قیمت لحظه‌ای و دقیق {currency_pair} هم اکنون {current_price} است.**

    تحلیل خود را مستقیماً بر اساس این قیمت شروع کن. با در نظر گرفتن این قیمت به عنوان نقطه شروع، نمودار را با انواع اندیکاتورهای معتبر و ترفندهای تکنیکال تحلیل کن. همچنین اخبار اکنون و اتفاقات آینده که ممکن است روی آن تاثیر بگذارد و تحلیل فاندامنتال آن را بررسی کرده و هوشمندانه بهترین نقطه ورود به همراه TP و SL را مشخص کن.

    بگو چون میخواهم استاپ اردر بگذارم، زمان انقضایش را چند ساعت بگذارم و این تحلیل تا چند ساعت معتبر است.

    مناسب برای اردر هم برای فروش و هم برای خرید پیشنهاد بده. میخواهم اردر کوتاه مدت باشد و در عرض چند ساعت نتیجه بدهد. در واقع کوتاه مدت ترین اردری که تحلیل قوی و احتمال موفقیت بالایی دارد را انتخاب کن.

    تحلیل فاندامنتال و اخبار را از اینترنت دریافت کن اما برای تحلیل تکنیکال، قیمت شروع را عددی که به تو دادم ({current_price}) در نظر بگیر.
    یادت باشه تحلیل تکنیکال چارت های این ارزو با بررسی دقیق تاریخچه آن که از منابع آنلاین دریافت میکنی با دقت و قدرت و با ترفند های مختلف و معتبر ترید و اندیکاتور های قوی انجام دهی
   و در انتها برآیند تمام تحلیل های فاندامنتال و تکنیکال را در قالب سیگنال بهم بدی
    ---
    **دستورالعمل‌های خروجی:**
    برای هر سیگنال پیشنهادی (خرید و فروش)، یک "امتیاز اطمینان" (Confidence Score) از 1 تا 10 و یک "دلیل" (Reason) بسیار کوتاه ارائه بده.
    خروجی را "دقیقا و فقط" با فرمت زیر برای هر دو سیگنال ارائه بده. بین دو سیگنال از "---" استفاده کن. هیچ متن اضافه دیگری ننویس.
    یعنی تمام پاسخت همین قالب زیر باشه و هیچ متنی حتی سلام یا هر چیز دیگه ای نباشه و همچنین تماما انگلیسی باشه
    PAIR: {currency_pair}
    TYPE: [نوع اردر مثل BUY_STOP]
    ENTRY: [قیمت ورود]
    SL: [حد ضرر]
    TP: [حد سود]
    Expiration: [زمانی که اگر اردر در طول آن فعال نشده بود حذف شود]
    CONFIDENCE: [امتیاز عددی بین 1 تا 10]
    REASON: [یک دلیل بسیار کوتاه و یک خطی]
    ---
    PAIR: {currency_pair}
    TYPE: [نوع اردر برای پوزیشن مخالف]
    ENTRY: [قیمت ورود]
    SL: [حد ضرر]
    TP: [حد سود]
    Expiration: [زمانی که اگر اردر در طول آن فعال نشده بود حذف شود]
    CONFIDENCE: [امتیاز عددی بین 1 تا 10]
    REASON: [یک دلیل بسیار کوتاه و یک خطی]
    """
    return user_prompt.strip()

# --- تابع دریافت سیگنال از Gemini (بدون تغییر) ---
def get_signal_for_pair(pair, current_price):
    """برای یک جفت ارز و قیمت مشخص، سیگنال را از Gemini دریافت می‌کند."""
    try:
        print(f"در حال ارسال درخواست تحلیل برای {pair} با قیمت {current_price} به Gemini...")
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = create_single_pair_prompt(pair, current_price)
        response = model.generate_content(prompt, request_options={'timeout': 150})
        print(f"پاسخ تحلیلی برای {pair} با موفقیت دریافت شد.")
        return response.text
    except Exception as e:
        print(f"خطایی در ارتباط با Gemini برای {pair} رخ داد: {e}")
        return None

# --- توابع پارس کردن و فیلتر کردن (بدون تغییر) ---
def parse_signals(raw_text):
    signals = []
    signal_blocks = raw_text.strip().split('---')
    for block in signal_blocks:
        if not block.strip() or "PAIR:" not in block.upper(): continue
        try:
            signal = {
                'pair': re.search(r"PAIR:\s*(.*)", block, re.IGNORECASE).group(1).strip(),
                'type': re.search(r"TYPE:\s*(.*)", block, re.IGNORECASE).group(1).strip(),
                'entry': float(re.search(r"ENTRY:\s*(.*)", block, re.IGNORECASE).group(1).strip()),
                'sl': float(re.search(r"SL:\s*(.*)", block, re.IGNORECASE).group(1).strip()),
                'tp': float(re.search(r"TP:\s*(.*)", block, re.IGNORECASE).group(1).strip()),
                'confidence': int(re.search(r"CONFIDENCE:\s*(.*)", block, re.IGNORECASE).group(1).strip()),
                'reason': re.search(r"REASON:\s*(.*)", block, re.IGNORECASE).group(1).strip(),
                'raw': block.strip()
            }
            signals.append(signal)
        except (AttributeError, ValueError) as e:
            print(f"خطا در پارس کردن بلوک سیگنال. بلوک نادیده گرفته شد. Error: {e}")
    return signals

def filter_and_rank_signals(signals, max_signals=10, max_per_currency=2):
    signals.sort(key=lambda x: x['confidence'], reverse=True)
    final_signals, currency_counts = [], defaultdict(int)
    for signal in signals:
        if len(final_signals) >= max_signals: break
        try:
            base, quote = signal['pair'].split('/')
            if currency_counts[base] < max_per_currency and currency_counts[quote] < max_per_currency:
                final_signals.append(signal)
                currency_counts[base] += 1
                currency_counts[quote] += 1
        except ValueError: continue
    return final_signals

def format_for_file(signals, title):
    output = f"{title}\n" + "=" * 40 + "\n\n"
    if signals:
        for i, signal in enumerate(signals, 1):
            output += f"# Rank {i} | Confidence: {signal['confidence']}/10\n{signal['raw']}\n---\n"
    else:
        output += "هیچ سیگنالی برای نمایش یافت نشد.\n"
    return output

# --- منطق اصلی برنامه ---
if __name__ == "__main__":
    all_raw_responses = []
    
    for pair in CURRENCY_PAIRS_TO_ANALYZE:
        # ۱. ابتدا قیمت لحظه‌ای را از Finnhub بگیر
        price = get_finnhub_price(pair, FINNHUB_API_KEY)
        
        # ۲. اگر قیمت با موفقیت دریافت شد، آن را برای تحلیل بفرست
        if price:
            response = get_signal_for_pair(pair, price)
            if response:
                all_raw_responses.append(response)
        else:
            print(f"تحلیل برای {pair} انجام نشد چون قیمت لحظه‌ای دریافت نگردید.")
            
        # ✨ تغییر ۳: تاخیر کمتر به لطف محدودیت بالاتر Finnhub ✨
        # (Finnhub رایگان: 60 درخواست در دقیقه)
        print("...ایجاد تاخیر 2 ثانیه‌ای برای مدیریت محدودیت API...")
        time.sleep(15) 

    if all_raw_responses:
        full_raw_text = "\n---\n".join(all_raw_responses)
        all_signals = parse_signals(full_raw_text)
        
        if all_signals:
            print(f"\nمجموعا {len(all_signals)} سیگنال با موفقیت پارس شد.")
            top_signals = filter_and_rank_signals(all_signals)
            
            title_top = f"Top {len(top_signals)} Trade Signals (Ranked & Filtered)"
            file_content_top = format_for_file(top_signals, title_top)
            with open("trade_signal.txt", "w", encoding="utf-8") as file:
                file.write(file_content_top)
            print("فایل 'trade_signal.txt' با سیگنال‌های برتر به‌روز شد.")
            
        else:
            print("هیچ سیگنال معتبری برای پردازش یافت نشد.")
    else:
        print("هیچ پاسخی از Gemini دریافت نشد.")
