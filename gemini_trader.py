import google.generativeai as genai
import os
import re
import time
from collections import defaultdict

# --- بخش تنظیمات ---

# خواندن کلید API از متغیرهای محیطی
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("کلید API گوگل در GOOGLE_API_KEY یافت نشد.")
genai.configure(api_key=api_key)

# لیست جفت ارزهای پیشنهادی (می‌توانید آن را تغییر دهید)
CURRENCY_PAIRS_TO_ANALYZE = [
    # Majors
    "EUR/USD", "USD/JPY", "GBP/USD", "USD/CHF",
    "AUD/USD", "USD/CAD", "NZD/USD",
    
    # Key Yen Crosses
    "EUR/JPY", "GBP/JPY", "AUD/JPY",
    
    # Key Euro Crosses
    "EUR/GBP", "EUR/AUD",
    
    # Key Pound Crosses
    "GBP/CHF", "GBP/AUD"
]


# --- بخش پرامپت اصلی شما (با حداقل تغییرات) ---

def create_single_pair_prompt(currency_pair):
    """پرامپت اصلی و دقیق کاربر را برای تحلیل یک جفت ارز ایجاد می‌کند."""
    # متن پرامپت شما در اینجا قرار گرفته است
    user_prompt = f"""
    با دقت نمودار ارز {currency_pair} را بررسی کن. با تحلیل تکنیکال انواع اندیکاتورهای معتبر (به چند عدد محدود اکتفا نکن) و همچنین بقیه ترفندهای تکنیکال، و همچنین بررسی دقیق اخبار اکنون و اتفاقاتی که ممکن است در روزهای آینده روی آن تاثیر بگذارد و تحلیل فاندامنتال آن، هوشمندانه بهترین نقطه ورود و همچنین تعیین tp و sl را مشخص کن.

    بگو چون میخواهم استاپ اردر بگذارم، زمان انقضایش را چند ساعت بگذارم و این تحلیل تا چند ساعت معتبر است.

    مناسب برای اردر هم برای فروش و هم برای خرید پیشنهاد بده. میخواهم اردر کوتاه مدت باشد و نیاز به چند روز برای به سرانجام رسیدن نداشته باشد و در عرض چند ساعت نتیجه بدهد. در واقع کوتاه مدت ترین اردری که تحلیل قوی و احتمال موفقیت بالایی را نسبت به زمانش دارد انتخاب کن.

    تمام داده ها و اطلاعات مورد نیاز را از اینترنت و سایت های مختلف دریافت کن و مطمئن و با توجه به وضعیت بازار همین لحظه بگو. قیمت را از چند منبع دریافت کن و با داده های آنلاین مقایسه کن که بهترین داده ها را برای تحلیل انتخاب کنی. خوب فکر کن و سریع جواب نده.

    ---
    **دستورالعمل‌های خروجی:**
    برای هر سیگنال پیشنهادی (خرید و فروش)، یک "امتیاز اطمینان" (Confidence Score) از 1 تا 10 و یک "دلیل" (Reason) بسیار کوتاه ارائه بده.
    خروجی را "دقیقا و فقط" با فرمت زیر برای هر دو سیگنال ارائه بده. بین دو سیگنال از "---" استفاده کن. هیچ متن اضافه دیگری ننویس.

    PAIR: {currency_pair}
    TYPE: [نوع اردر مثل BUY_STOP]
    ENTRY: [قیمت ورود]
    SL: [حد ضرر]
    TP: [حد سود]
    CONFIDENCE: [امتیاز عددی بین 1 تا 10]
    REASON: [یک دلیل بسیار کوتاه و یک خطی]
    ---
    PAIR: {currency_pair}
    TYPE: [نوع اردر برای پوزیشن مخالف]
    ENTRY: [قیمت ورود]
    SL: [حد ضرر]
    TP: [حد سود]
    CONFIDENCE: [امتیاز عددی بین 1 تا 10]
    REASON: [یک دلیل بسیار کوتاه و یک خطی]
    """
    return user_prompt.strip()

# --- بخش پردازش و منطق (بدون تغییر) ---

def get_signal_for_pair(pair):
    """برای یک جفت ارز مشخص، سیگنال را از Gemini دریافت می‌کند."""
    try:
        print(f"در حال ارسال درخواست اختصاصی برای: {pair}...")
        model = genai.GenerativeModel('gemini-pro')
        prompt = create_single_pair_prompt(pair)
        response = model.generate_content(prompt, request_options={'timeout': 150})
        print(f"پاسخ برای {pair} با موفقیت دریافت شد.")
        return response.text
    except Exception as e:
        print(f"خطایی در ارتباط با Gemini برای {pair} رخ داد: {e}")
        return None

def parse_signals(raw_text):
    """متن خام پاسخ‌ها را به لیستی از دیکشنری‌های سیگنال تبدیل می‌کند."""
    signals = []
    signal_blocks = raw_text.strip().split('---')
    
    for block in signal_blocks:
        if not block.strip() or "PAIR:" not in block.upper():
            continue
        
        signal = {}
        try:
            signal['pair'] = re.search(r"PAIR:\s*(.*)", block, re.IGNORECASE).group(1).strip()
            signal['type'] = re.search(r"TYPE:\s*(.*)", block, re.IGNORECASE).group(1).strip()
            signal['entry'] = float(re.search(r"ENTRY:\s*(.*)", block, re.IGNORECASE).group(1).strip())
            signal['sl'] = float(re.search(r"SL:\s*(.*)", block, re.IGNORECASE).group(1).strip())
            signal['tp'] = float(re.search(r"TP:\s*(.*)", block, re.IGNORECASE).group(1).strip())
            signal['confidence'] = int(re.search(r"CONFIDENCE:\s*(.*)", block, re.IGNORECASE).group(1).strip())
            signal['reason'] = re.search(r"REASON:\s*(.*)", block, re.IGNORECASE).group(1).strip()
            signal['raw'] = block.strip()
            signals.append(signal)
        except (AttributeError, ValueError) as e:
            print(f"خطا در پارس کردن بلوک سیگنال. بلوک نادیده گرفته شد. Error: {e}")
            
    return signals

def filter_and_rank_signals(signals, max_signals=10, max_per_currency=2):
    """سیگنال‌ها را رتبه‌بندی، فیلتر و بر اساس قوانین تنوع‌بخشی انتخاب می‌کند."""
    signals.sort(key=lambda x: x['confidence'], reverse=True)
    
    final_signals = []
    currency_counts = defaultdict(int)
    
    for signal in signals:
        if len(final_signals) >= max_signals:
            break
        
        try:
            base_currency, quote_currency = signal['pair'].split('/')
            
            if currency_counts[base_currency] < max_per_currency and currency_counts[quote_currency] < max_per_currency:
                final_signals.append(signal)
                currency_counts[base_currency] += 1
                currency_counts[quote_currency] += 1
        except ValueError:
            print(f"فرمت جفت ارز '{signal['pair']}' نامعتبر است و نادیده گرفته شد.")

    return final_signals

def format_for_file(signals):
    """سیگنال‌های نهایی را برای نوشتن در فایل فرمت‌بندی می‌کند."""
    output = f"Top {len(signals)} Trade Signals (Ranked & Filtered)\n"
    output += "=" * 40 + "\n\n"
    
    for i, signal in enumerate(signals, 1):
        output += f"# Rank {i} | Confidence: {signal['confidence']}/10\n"
        output += signal['raw'] + "\n"
        output += "---\n"
        
    return output

# --- بخش اجرایی اصلی (بدون تغییر) ---

if __name__ == "__main__":
    all_raw_responses = []
    
    for pair in CURRENCY_PAIRS_TO_ANALYZE:
        response = get_signal_for_pair(pair)
        if response:
            all_raw_responses.append(response)
        time.sleep(90)

    if all_raw_responses:
        full_raw_text = "\n---\n".join(all_raw_responses)
        
        all_signals = parse_signals(full_raw_text)
        
        if all_signals:
            print(f"\nمجموعا {len(all_signals)} سیگنال با موفقیت پارس شد.")
            
            top_signals = filter_and_rank_signals(all_signals)
            print(f"تعداد {len(top_signals)} سیگنال برتر پس از فیلتر انتخاب شد.")
            
            file_content = format_for_file(top_signals)
            
            with open("trade_signal.txt", "w", encoding="utf-8") as file:
                file.write(file_content)
            print("فایل 'trade_signal.txt' با سیگنال‌های برتر به‌روز شد.")
        else:
            print("هیچ سیگنال معتبری برای پردازش یافت نشد.")
    else:
        print("هیچ پاسخی از Gemini دریافت نشد.")
