import requests
from datetime import datetime

# ما از منبع اولیه و قابل اعتماد faireconomy استفاده می‌کنیم
NEWS_SOURCE_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
IMPORTANT_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF"]

def fetch_and_save_news():
    try:
        response = requests.get(NEWS_SOURCE_URL, timeout=15)
        response.raise_for_status()
        news_data = response.json()

        event_times = []
        for event in news_data:
            if event.get("impact") == "High" and event.get("country") in IMPORTANT_CURRENCIES:
                dt_object = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
                formatted_time = dt_object.strftime('%Y.%m.%d %H:%M')
                event_times.append(formatted_time)

        unique_times = sorted(list(set(event_times)))
        
        # نتیجه را در یک فایل متنی به نام news_calendar.txt ذخیره کن
        with open("news_calendar.txt", "w") as f:
            f.write(",".join(unique_times))
        
        print("News calendar successfully generated.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    fetch_and_save_news()
