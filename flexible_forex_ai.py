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
import aiohttp
import asyncio
from typing import Dict, List, Optional, Tuple
import concurrent.futures

# =================================================================================
# --- بخش تنظیمات اصلی پیشرفته ---
# =================================================================================

# کلیدهای API
google_api_key = os.getenv("GOOGLE_API_KEY")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
CLOUDFLARE_AI_API_KEY = os.getenv("CLOUDFLARE_AI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not all([google_api_key, TWELVEDATA_API_KEY]):
    raise ValueError("لطفاً کلیدهای API را تنظیم کنید: GOOGLE_API_KEY, TWELVEDATA_API_KEY")

# تنظیمات اصلی سیستم
HIGH_TIMEFRAME = "4h"
LOW_TIMEFRAME = "1h"
CANDLES_TO_FETCH = 300
CURRENCY_PAIRS_TO_ANALYZE = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
    "GBP/JPY", "EUR/JPY", "AUD/JPY", "NZD/USD", "USD/CAD"
]

CACHE_FILE = "signal_cache.json"
USAGE_TRACKER_FILE = "api_usage_tracker.json"
LOG_FILE = "trading_log.log"

# مدل‌های AI به‌روز شده با تنوع بیشتر
GEMINI_MODEL = 'gemini-2.5-flash'

# مدل‌های Cloudflare با تنوع بیشتر
CLOUDFLARE_MODELS = [
    "@cf/meta/llama-4-scout-17b-16e-instruct",  # Llama 4
    "@cf/google/gemma-3-12b-it",               # Gemma 3
    "@cf/mistralai/mistral-small-3.1-24b-instruct",  # Mistral
    "@cf/qwen/qwq-32b",                        # Qwen
    "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b"  # DeepSeek
]

# مدل‌های Groq با تنوع بیشتر
GROQ_MODELS = [
    "llama-3.3-70b-versatile",                 # Llama 3.3
    "qwen/qwen3-32b",                          # Qwen 3
    "meta-llama/llama-4-maverick-17b-128e-instruct",  # Llama 4 Maverick
    "llama-3.1-8b-instant",                    # Llama 3.1
    "mixtral-8x7b-32768"                       # Mixtral
]

# محدودیت‌های روزانه API
API_DAILY_LIMITS = {
    "google_gemini": 1500,
    "cloudflare": 10000,
    "groq": 10000
}

# راه‌اندازی سیستم لاگ‌گیری پیشرفته
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# =================================================================================
# --- کلاس مدیریت مصرف API هوشمند با انتخاب متنوع مدل‌ها ---
# =================================================================================

class SmartAPIManager:
    def __init__(self, usage_file: str):
        self.usage_file = usage_file
        self.usage_data = self.load_usage_data()
        self.available_models = self.initialize_available_models()
        self.failed_models = set()

    def load_usage_data(self) -> Dict:
        """بارگذاری داده‌های مصرف API"""
        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return self.check_and_reset_daily_usage(data)
            return self.initialize_usage_data()
        except Exception as e:
            logging.error(f"خطا در بارگذاری داده‌های مصرف API: {e}")
            return self.initialize_usage_data()

    def initialize_usage_data(self) -> Dict:
        """مقداردهی اولیه داده‌های مصرف"""
        today = datetime.now(UTC).date().isoformat()
        return {
            "last_reset_date": today,
            "providers": {
                "google_gemini": {"used_today": 0, "limit": API_DAILY_LIMITS["google_gemini"]},
                "cloudflare": {"used_today": 0, "limit": API_DAILY_LIMITS["cloudflare"]},
                "groq": {"used_today": 0, "limit": API_DAILY_LIMITS["groq"]}
            }
        }

    def check_and_reset_daily_usage(self, data: Dict) -> Dict:
        """بررسی و ریست مصرف روزانه"""
        today = datetime.now(UTC).date().isoformat()
        last_reset = data.get("last_reset_date", "")
        
        if last_reset != today:
            for provider in data["providers"]:
                data["providers"][provider]["used_today"] = 0
            data["last_reset_date"] = today
            self.save_usage_data(data)
            logging.info("✅ مصرف روزانه APIها ریست شد")
        return data

    def save_usage_data(self, data: Dict = None):
        """ذخیره داده‌های مصرف"""
        if data is None:
            data = self.usage_data
        try:
            with open(self.usage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logging.error(f"خطا در ذخیره داده‌های مصرف API: {e}")

    def initialize_available_models(self) -> Dict:
        """مقداردهی اولیه مدل‌های موجود"""
        return {
            "google_gemini": [GEMINI_MODEL],
            "cloudflare": CLOUDFLARE_MODELS.copy(),
            "groq": GROQ_MODELS.copy()
        }

    def can_use_provider(self, provider: str) -> bool:
        """بررسی امکان استفاده از provider"""
        if provider not in self.usage_data["providers"]:
            return False
        provider_data = self.usage_data["providers"][provider]
        remaining = provider_data["limit"] - provider_data["used_today"]
        return remaining > 0

    def get_available_models_count(self, provider: str) -> int:
        """تعداد مدل‌های موجود برای یک provider"""
        if not self.can_use_provider(provider):
            return 0
        provider_data = self.usage_data["providers"][provider]
        remaining = provider_data["limit"] - provider_data["used_today"]
        available_models = len(self.available_models[provider])
        return min(remaining, available_models)

    def mark_model_failed(self, provider: str, model_name: str):
        """علامت‌گذاری مدل شکست خورده"""
        self.failed_models.add((provider, model_name))
        logging.warning(f"❌ مدل {provider}/{model_name} به لیست شکست‌خورده‌ها اضافه شد")

    def is_model_failed(self, provider: str, model_name: str) -> bool:
        """بررسی آیا مدل شکست خورده است"""
        return (provider, model_name) in self.failed_models

    def select_diverse_models(self, target_total: int = 5, min_required: int = 3) -> List[Tuple[str, str]]:
        """انتخاب مدل‌های متنوع از providerهای مختلف"""
        selected_models = []

        # محاسبه ظرفیت هر provider
        provider_capacity = {}
        for provider in ["google_gemini", "cloudflare", "groq"]:
            provider_capacity[provider] = self.get_available_models_count(provider)

        logging.info(f"📊 ظرفیت providerها: Gemini={provider_capacity['google_gemini']}, "
                    f"Cloudflare={provider_capacity['cloudflare']}, Groq={provider_capacity['groq']}")

        # استراتژی: انتخاب متنوع از همه providerها
        total_available = sum(provider_capacity.values())
        if total_available == 0:
            logging.error("❌ هیچ providerی در دسترس نیست")
            return selected_models

        # توزیع متعادل بین providerها
        providers_order = ["google_gemini", "cloudflare", "groq"]
        round_robin_index = 0
        remaining_target = min(target_total, total_available)

        while remaining_target > 0 and any(provider_capacity[p] > 0 for p in providers_order):
            current_provider = providers_order[round_robin_index % len(providers_order)]
            
            if provider_capacity[current_provider] > 0:
                # انتخاب اولین مدل موجود از این provider که شکست نخورده باشد
                for model_name in self.available_models[current_provider]:
                    if (current_provider, model_name) not in selected_models and not self.is_model_failed(current_provider, model_name):
                        selected_models.append((current_provider, model_name))
                        provider_capacity[current_provider] -= 1
                        remaining_target -= 1
                        break
            
            round_robin_index += 1
            
            # اگر بعد از یک دور کامل چیزی اضافه نشد، break کن
            if round_robin_index > len(providers_order) * 2:  # حداکثر 2 دور چرخش
                break

        # اگر به حداقل نرسیدیم، Fallback
        if len(selected_models) < min_required:
            logging.warning(f"⚠️ فقط {len(selected_models)} مدل انتخاب شد. فعال کردن Fallback...")
            # جستجو برای مدل‌های اضافی
            additional_models = []
            for provider in providers_order:
                if self.can_use_provider(provider):
                    for model_name in self.available_models[provider]:
                        if (provider, model_name) not in selected_models and not self.is_model_failed(provider, model_name):
                            additional_models.append((provider, model_name))
                            if len(additional_models) >= (min_required - len(selected_models)):
                                break
                    if len(selected_models) + len(additional_models) >= min_required:
                        break
            
            selected_models.extend(additional_models)

        logging.info(f"🎯 {len(selected_models)} مدل متنوع انتخاب شد: {selected_models}")
        return selected_models

    def record_api_usage(self, provider: str, count: int = 1):
        """ثبت استفاده از API"""
        if provider in self.usage_data["providers"]:
            self.usage_data["providers"][provider]["used_today"] += count
            self.save_usage_data()

    def get_usage_summary(self) -> str:
        """خلاصه وضعیت مصرف"""
        summary = "📊 خلاصه مصرف API:\n"
        for provider, data in self.usage_data["providers"].items():
            remaining = data["limit"] - data["used_today"]
            summary += f"  {provider}: {data['used_today']}/{data['limit']} ({remaining} باقیمانده)\n"
        return summary

# =================================================================================
# --- کلاس تحلیل تکنیکال پیشرفته ---
# =================================================================================

class AdvancedTechnicalAnalyzer:
    def __init__(self):
        self.indicators_config = {
            'trend': ['ema_21', 'ema_50', 'ema_200', 'adx_14'],
            'momentum': ['rsi_14', 'stoch_14_3_3', 'macd'],
            'volatility': ['bb_20_2', 'atr_14'],
            'ichimoku': True,
            'support_resistance': True,
            'candle_patterns': True
        }

    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or len(df) < 100:
            return None

        try:
            # اندیکاتورهای روند
            df.ta.ema(length=21, append=True)
            df.ta.ema(length=50, append=True)
            df.ta.ema(length=200, append=True)
            df.ta.adx(length=14, append=True)

            # اندیکاتورهای مومنتوم
            df.ta.rsi(length=14, append=True)
            df.ta.stoch(append=True)
            df.ta.macd(append=True)

            # اندیکاتورهای نوسان
            df.ta.bbands(length=20, std=2, append=True)
            df.ta.atr(length=14, append=True)

            # ایچیموکو
            df.ta.ichimoku(append=True)

            # سطوح حمایت و مقاومت
            df['sup_1'] = df['low'].rolling(20).min().shift(1)
            df['res_1'] = df['high'].rolling(20).max().shift(1)
            df['sup_2'] = df['low'].rolling(50).min().shift(1)
            df['res_2'] = df['high'].rolling(50).max().shift(1)

            df.dropna(inplace=True)
            return df

        except Exception as e:
            logging.error(f"خطا در محاسبه اندیکاتورها: {e}")
            return None

    def generate_technical_analysis(self, symbol: str, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> Dict:
        if htf_df.empty or ltf_df.empty:
            return None

        last_htf = htf_df.iloc[-1]
        last_ltf = ltf_df.iloc[-1]

        # تحلیل روند
        htf_trend = self._analyze_trend(last_htf)
        ltf_trend = self._analyze_trend(last_ltf)

        # تحلیل مومنتوم
        momentum = self._analyze_momentum(last_ltf)

        # تحلیل سطوح کلیدی
        key_levels = self._analyze_key_levels(htf_df, ltf_df, last_ltf['close'])

        return {
            'symbol': symbol,
            'htf_trend': htf_trend,
            'ltf_trend': ltf_trend,
            'momentum': momentum,
            'key_levels': key_levels,
            'volatility': last_ltf.get('ATRr_14', 0),
            'timestamp': datetime.now(UTC).isoformat()
        }

    def _analyze_trend(self, data: pd.Series) -> Dict:
        ema_21 = data.get('EMA_21', 0)
        ema_50 = data.get('EMA_50', 0)
        ema_200 = data.get('EMA_200', 0)
        adx = data.get('ADX_14', 0)

        trend_direction = "صعودی" if ema_21 > ema_50 > ema_200 else "نزولی" if ema_21 < ema_50 < ema_200 else "خنثی"
        trend_strength = "قوی" if adx > 25 else "ضعیف" if adx < 20 else "متوسط"

        return {
            'direction': trend_direction,
            'strength': trend_strength,
            'adx': adx
        }

    def _analyze_momentum(self, data: pd.Series) -> Dict:
        rsi = data.get('RSI_14', 50)
        macd_hist = data.get('MACDh_12_26_9', 0)
        stoch_k = data.get('STOCHk_14_3_3', 50)

        rsi_signal = "اشباع خرید" if rsi > 70 else "اشباع فروش" if rsi < 30 else "خنثی"
        macd_signal = "صعودی" if macd_hist > 0 else "نزولی"
        stoch_signal = "اشباع خرید" if stoch_k > 80 else "اشباع فروش" if stoch_k < 20 else "خنثی"

        return {
            'rsi': {'value': rsi, 'signal': rsi_signal},
            'macd': {'signal': macd_signal, 'histogram': macd_hist},
            'stochastic': {'value': stoch_k, 'signal': stoch_signal}
        }

    def _analyze_key_levels(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame, current_price: float) -> Dict:
        bb_upper = ltf_df.get('BBU_20_2.0', pd.Series([0])).iloc[-1]
        bb_lower = ltf_df.get('BBL_20_2.0', pd.Series([0])).iloc[-1]
        support_1 = ltf_df.get('sup_1', pd.Series([0])).iloc[-1]
        resistance_1 = ltf_df.get('res_1', pd.Series([0])).iloc[-1]
        support_2 = ltf_df.get('sup_2', pd.Series([0])).iloc[-1]
        resistance_2 = ltf_df.get('res_2', pd.Series([0])).iloc[-1]

        return {
            'support_1': support_1,
            'resistance_1': resistance_1,
            'support_2': support_2,
            'resistance_2': resistance_2,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower
        }

# =================================================================================
# --- کلاس مدیریت AI با پرامپت بهبود یافته ---
# =================================================================================

class EnhancedAIManager:
    def __init__(self, gemini_api_key: str, cloudflare_api_key: str, groq_api_key: str, api_manager: SmartAPIManager):
        self.gemini_api_key = gemini_api_key
        self.cloudflare_api_key = cloudflare_api_key
        self.groq_api_key = groq_api_key
        self.api_manager = api_manager
        genai.configure(api_key=gemini_api_key)

    async def get_enhanced_ai_analysis(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        """تحلیل پیشرفته با حداقل ۳ مدل AI"""
        selected_models = self.api_manager.select_diverse_models(target_total=5, min_required=3)
        
        if len(selected_models) < 3:
            logging.error(f"❌ نمی‌توان حداقل ۳ مدل AI برای تحلیل {symbol} پیدا کرد")
            return None

        logging.info(f"🎯 استفاده از {len(selected_models)} مدل AI برای {symbol}")

        tasks = []
        for provider, model_name in selected_models:
            task = self._get_single_analysis(symbol, technical_analysis, provider, model_name)
            tasks.append(task)

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            valid_results = []
            failed_count = 0
            
            for i, (provider, model_name) in enumerate(selected_models):
                result = results[i]
                if isinstance(result, Exception):
                    logging.error(f"خطا در {provider}/{model_name} برای {symbol}: {result}")
                    self.api_manager.mark_model_failed(provider, model_name)
                    failed_count += 1
                    self.api_manager.record_api_usage(provider)
                elif result is not None:
                    valid_results.append(result)
                    self.api_manager.record_api_usage(provider)
                else:
                    self.api_manager.record_api_usage(provider)

            logging.info(f"📊 نتایج: {len(valid_results)} موفق, {failed_count} شکست")
            return self._combine_signals(symbol, valid_results, len(selected_models))

        except Exception as e:
            logging.error(f"خطا در تحلیل AI برای {symbol}: {e}")
            return None

    async def _get_single_analysis(self, symbol: str, technical_analysis: Dict, provider: str, model_name: str) -> Optional[Dict]:
        """تحلیل با یک مدل خاص"""
        try:
            prompt = self._create_enhanced_prompt(symbol, technical_analysis)
            
            if provider == "google_gemini":
                return await self._get_gemini_analysis(symbol, prompt, model_name)
            elif provider == "cloudflare":
                return await self._get_cloudflare_analysis(symbol, prompt, model_name)
            elif provider == "groq":
                return await self._get_groq_analysis(symbol, prompt, model_name)
            else:
                return None
                
        except Exception as e:
            logging.warning(f"خطا در تحلیل {provider}/{model_name} برای {symbol}: {e}")
            return None

    def _create_enhanced_prompt(self, symbol: str, technical_analysis: Dict) -> str:
        """ایجاد پرامپت بهبود یافته"""
        current_price = 1.0850  # مقدار نمونه - در عمل باید از داده‌های واقعی استفاده شود
        
        return f"""
IMPORTANT: You MUST return ONLY valid JSON format. No other text, no explanations.

As a professional forex trading analyst, analyze this currency pair and provide trading signals:

SYMBOL: {symbol}
CURRENT PRICE: ~{current_price}

TECHNICAL ANALYSIS:
- HTF Trend (4H): {technical_analysis['htf_trend']['direction']} ({technical_analysis['htf_trend']['strength']})
- LTF Trend (1H): {technical_analysis['ltf_trend']['direction']}
- RSI: {technical_analysis['momentum']['rsi']['value']:.1f} ({technical_analysis['momentum']['rsi']['signal']})
- MACD: {technical_analysis['momentum']['macd']['signal']}
- Key Support: {technical_analysis['key_levels']['support_1']:.5f}
- Key Resistance: {technical_analysis['key_levels']['resistance_1']:.5f}

CALCULATE REALISTIC LEVELS based on current price ~{current_price} and technical structure.

RETURN ONLY THIS EXACT JSON FORMAT:
{{
  "SYMBOL": "{symbol}",
  "ACTION": "BUY",
  "CONFIDENCE": 7,
  "ENTRY_ZONE": "{current_price - 0.0010:.5f}-{current_price + 0.0005:.5f}",
  "STOP_LOSS": "{current_price - 0.0020:.5f}",
  "TAKE_PROFIT": "{current_price + 0.0030:.5f}",
  "RISK_REWARD_RATIO": "1.5",
  "ANALYSIS": "تحلیل فنی بر اساس روند صعودی و مومنتوم مثبت",
  "EXPIRATION_H": 4,
  "TRADE_RATIONALE": "سیگنال بر اساس همگرایی اندیکاتورها"
}}

Your response must be ONLY the JSON object.
"""

    async def _get_gemini_analysis(self, symbol: str, prompt: str, model_name: str) -> Optional[Dict]:
        """تحلیل با Gemini"""
        try:
            model = genai.GenerativeModel(model_name)
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                request_options={'timeout': 60}
            )
            return self._parse_ai_response(response.text, symbol, f"Gemini-{model_name}")
        except Exception as e:
            logging.warning(f"خطا در تحلیل Gemini برای {symbol}: {e}")
            return None

    async def _get_cloudflare_analysis(self, symbol: str, prompt: str, model_name: str) -> Optional[Dict]:
        """تحلیل با Cloudflare AI"""
        if not self.cloudflare_api_key:
            return None

        try:
            headers = {
                "Authorization": f"Bearer {self.cloudflare_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a forex trading expert. Return ONLY valid JSON format."
                    },
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }

            account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "your_account_id")
            url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_name}"

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=60) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = ""
                        if "result" in data and "response" in data["result"]:
                            content = data["result"]["response"]
                        elif "response" in data:
                            content = data["response"]
                            
                        if content:
                            return self._parse_ai_response(content, symbol, f"Cloudflare-{model_name}")
            return None
            
        except Exception as e:
            logging.warning(f"خطا در تحلیل Cloudflare/{model_name} برای {symbol}: {e}")
            return None

    async def _get_groq_analysis(self, symbol: str, prompt: str, model_name: str) -> Optional[Dict]:
        """تحلیل با Groq API"""
        if not self.groq_api_key:
            return None

        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a forex trading expert. Return ONLY valid JSON format."
                    },
                    {"role": "user", "content": prompt}
                ],
                "model": model_name,
                "temperature": 0.1,
                "max_tokens": 500,
                "stream": False
            }

            url = "https://api.groq.com/openai/v1/chat/completions"

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=60) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "choices" in data and len(data["choices"]) > 0:
                            content = data["choices"][0]["message"]["content"]
                            return self._parse_ai_response(content, symbol, f"Groq-{model_name}")
            return None
            
        except Exception as e:
            logging.warning(f"خطا در تحلیل Groq/{model_name} برای {symbol}: {e}")
            return None

    def _parse_ai_response(self, response: str, symbol: str, ai_name: str) -> Optional[Dict]:
        """پارس کردن پاسخ AI با قابلیت بهبود یافته"""
        try:
            cleaned_response = response.strip()
            
            # حذف markdown و کد بلاک
            cleaned_response = re.sub(r'```json\s*', '', cleaned_response)
            cleaned_response = re.sub(r'```\s*', '', cleaned_response)
            
            # یافتن JSON
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                signal_data = json.loads(json_str)
                
                if self._validate_signal_data(signal_data, symbol):
                    signal_data['ai_model'] = ai_name
                    signal_data['timestamp'] = datetime.now(UTC).isoformat()
                    
                    # اعتبارسنجی مقادیر عددی
                    signal_data = self._validate_numeric_values(signal_data, symbol)
                    logging.info(f"✅ {ai_name} سیگنال برای {symbol}: {signal_data.get('ACTION', 'HOLD')}")
                    return signal_data

            logging.warning(f"❌ پاسخ {ai_name} برای {symbol} فاقد فرمت JSON معتبر")
            return None
            
        except json.JSONDecodeError as e:
            logging.error(f"خطای JSON در پاسخ {ai_name} برای {symbol}: {e}")
            return None
        except Exception as e:
            logging.error(f"خطا در پارس کردن پاسخ {ai_name} برای {symbol}: {e}")
            return None

    def _validate_signal_data(self, signal_data: Dict, symbol: str) -> bool:
        """اعتبارسنجی داده‌های سیگنال"""
        required_fields = ['SYMBOL', 'ACTION', 'CONFIDENCE']
        
        for field in required_fields:
            if field not in signal_data:
                logging.warning(f"فیلد ضروری {field} در سیگنال {symbol} وجود ندارد")
                return False

        action = signal_data['ACTION'].upper()
        if action not in ['BUY', 'SELL', 'HOLD']:
            logging.warning(f"ACTION نامعتبر برای {symbol}: {action}")
            return False

        try:
            confidence = float(signal_data['CONFIDENCE'])
            if not (1 <= confidence <= 10):
                logging.warning(f"CONFIDENCE خارج از محدوده برای {symbol}: {confidence}")
                return False
        except (ValueError, TypeError):
            logging.warning(f"CONFIDENCE نامعتبر برای {symbol}: {signal_data['CONFIDENCE']}")
            return False

        return True

    def _validate_numeric_values(self, signal_data: Dict, symbol: str) -> Dict:
        """اعتبارسنجی و اصلاح مقادیر عددی"""
        numeric_fields = ['ENTRY_ZONE', 'STOP_LOSS', 'TAKE_PROFIT', 'EXPIRATION_H', 'RISK_REWARD_RATIO']
        
        for field in numeric_fields:
            if field in signal_data:
                value = signal_data[field]
                if value is None or value == "null" or str(value).strip() == "":
                    # مقدار پیش‌فرض برای فیلدهای ضروری
                    if field == 'EXPIRATION_H':
                        signal_data[field] = 4
                    elif field == 'RISK_REWARD_RATIO':
                        signal_data[field] = "1.5"
                    else:
                        signal_data[field] = "N/A"
        
        return signal_data

    def _combine_signals(self, symbol: str, valid_results: List[Dict], total_models: int) -> Optional[Dict]:
        """ترکیب نتایج سیگنال‌ها"""
        if not valid_results:
            return None

        # شمارش سیگنال‌ها
        action_counts = {}
        for result in valid_results:
            action = result['ACTION'].upper()
            action_counts[action] = action_counts.get(action, 0) + 1

        total_valid = len(valid_results)
        max_agreement = max(action_counts.values())

        # تعیین نوع توافق
        if max_agreement >= 3:
            agreement_type = 'STRONG_CONSENSUS'
        elif max_agreement == 2:
            agreement_type = 'MEDIUM_CONSENSUS'
        else:
            agreement_type = 'WEAK_CONSENSUS'

        # انتخاب سیگنال اکثریت
        majority_action = max(action_counts, key=action_counts.get)
        agreeing_results = [r for r in valid_results if r['ACTION'].upper() == majority_action]

        # ترکیب سیگنال‌های موافق
        combined = {
            'SYMBOL': symbol,
            'ACTION': majority_action,
            'AGREEMENT_LEVEL': max_agreement,
            'AGREEMENT_TYPE': agreement_type,
            'VALID_MODELS': total_valid,
            'TOTAL_MODELS': total_models,
            'timestamp': datetime.now(UTC).isoformat()
        }

        # میانگین اعتماد
        if agreeing_results:
            confidences = [float(r.get('CONFIDENCE', 5)) for r in agreeing_results]
            combined['CONFIDENCE'] = round(sum(confidences) / len(confidences), 1)

        # استفاده از مقادیر از اولین سیگنال معتبر
        if agreeing_results:
            first_valid = agreeing_results[0]
            for field in ['ENTRY_ZONE', 'STOP_LOSS', 'TAKE_PROFIT', 'RISK_REWARD_RATIO', 'EXPIRATION_H', 'ANALYSIS']:
                if field in first_valid and first_valid[field] not in [None, "null", ""]:
                    combined[field] = first_valid[field]
                else:
                    # مقدار پیش‌فرض برای فیلدهای ضروری
                    if field == 'ANALYSIS':
                        combined[field] = f"سیگنال {majority_action} بر اساس توافق {max_agreement} از {total_models} مدل AI"
                    elif field == 'EXPIRATION_H':
                        combined[field] = 4
                    elif field == 'RISK_REWARD_RATIO':
                        combined[field] = "1.5"

        return combined

# =================================================================================
# --- کلاس اصلی تحلیلگر فارکس ---
# =================================================================================

class ImprovedForexAnalyzer:
    def __init__(self):
        self.api_manager = SmartAPIManager(USAGE_TRACKER_FILE)
        self.technical_analyzer = AdvancedTechnicalAnalyzer()
        self.ai_manager = EnhancedAIManager(google_api_key, CLOUDFLARE_AI_API_KEY, GROQ_API_KEY, self.api_manager)

    async def analyze_pair(self, pair: str) -> Optional[Dict]:
        """تحلیل کامل یک جفت ارز"""
        logging.info(f"🔍 شروع تحلیل برای {pair}")
        try:
            logging.info(self.api_manager.get_usage_summary())

            # دریافت داده‌های بازار
            htf_df = await self.get_market_data(pair, HIGH_TIMEFRAME)
            ltf_df = await self.get_market_data(pair, LOW_TIMEFRAME)
            
            if htf_df is None or ltf_df is None:
                logging.warning(f"⚠️ دریافت داده‌های بازار برای {pair} ناموفق بود")
                return None

            # تحلیل تکنیکال
            htf_df_processed = self.technical_analyzer.calculate_advanced_indicators(htf_df)
            ltf_df_processed = self.technical_analyzer.calculate_advanced_indicators(ltf_df)
            
            if htf_df_processed is None or ltf_df_processed is None:
                logging.warning(f"⚠️ تحلیل تکنیکال برای {pair} ناموفق بود")
                return None

            technical_analysis = self.technical_analyzer.generate_technical_analysis(
                pair, htf_df_processed, ltf_df_processed
            )
            
            if not technical_analysis:
                logging.warning(f"⚠️ تولید تحلیل تکنیکال برای {pair} ناموفق بود")
                return None

            # تحلیل AI
            ai_analysis = await self.ai_manager.get_enhanced_ai_analysis(pair, technical_analysis)
            
            if ai_analysis and ai_analysis.get('ACTION') != 'HOLD':
                logging.info(f"✅ سیگنال برای {pair}: {ai_analysis['ACTION']} "
                            f"(توافق: {ai_analysis.get('AGREEMENT_LEVEL', 0)}/{ai_analysis.get('TOTAL_MODELS', 0)})")
                return ai_analysis
            
            logging.info(f"🔍 هیچ سیگنال معاملاتی برای {pair}")
            return None
            
        except Exception as e:
            logging.error(f"❌ خطا در تحلیل {pair}: {e}")
            return None

    async def get_market_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """دریافت داده‌های بازار"""
        try:
            url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={CANDLES_TO_FETCH}&apikey={TWELVEDATA_API_KEY}'
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'values' in data and data['values']:
                            df = pd.DataFrame(data['values'])
                            df = df.iloc[::-1].reset_index(drop=True)
                            
                            for col in ['open', 'high', 'low', 'close']:
                                if col in df.columns:
                                    df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                            df = df.dropna()
                            return df
            return None
            
        except Exception as e:
            logging.error(f"خطا در دریافت داده‌های بازار برای {symbol}: {e}")
            return None

    async def analyze_all_pairs(self, pairs: List[str]) -> List[Dict]:
        """تحلیل همه جفت ارزها"""
        logging.info(f"🚀 شروع تحلیل برای {len(pairs)} جفت ارز")
        
        tasks = [self.analyze_pair(pair) for pair in pairs]
        results = await asyncio.gather(*tasks)
        
        valid_signals = [r for r in results if r is not None]
        logging.info(f"📊 تحلیل کامل شد. {len(valid_signals)} سیگنال معتبر")
        return valid_signals

    def save_signals(self, signals: List[Dict]):
        """ذخیره سیگنال‌ها در فایل‌های مختلف"""
        if not signals:
            logging.info("📝 هیچ سیگنالی برای ذخیره وجود ندارد")
            return

        strong_signals = []
        medium_signals = []
        weak_signals = []

        for signal in signals:
            agreement_type = signal.get('AGREEMENT_TYPE', '')
            if agreement_type == 'STRONG_CONSENSUS':
                strong_signals.append(signal)
            elif agreement_type == 'MEDIUM_CONSENSUS':
                medium_signals.append(signal)
            else:
                weak_signals.append(signal)

        # ذخیره با مدیریت خطا
        try:
            if strong_signals:
                with open("strong_consensus_signals.json", 'w', encoding='utf-8') as f:
                    json.dump(strong_signals, f, indent=2, ensure_ascii=False)
                logging.info(f"💾 {len(strong_signals)} سیگنال قوی ذخیره شد")
            
            if medium_signals:
                with open("medium_consensus_signals.json", 'w', encoding='utf-8') as f:
                    json.dump(medium_signals, f, indent=2, ensure_ascii=False)
                logging.info(f"💾 {len(medium_signals)} سیگنال متوسط ذخیره شد")
            
            if weak_signals:
                with open("weak_consensus_signals.json", 'w', encoding='utf-8') as f:
                    json.dump(weak_signals, f, indent=2, ensure_ascii=False)
                logging.info(f"💾 {len(weak_signals)} سیگنال ضعیف ذخیره شد")
                
        except Exception as e:
            logging.error(f"❌ خطا در ذخیره‌سازی سیگنال‌ها: {e}")

# =================================================================================
# --- تابع اصلی ---
# =================================================================================

async def main():
    """تابع اصلی اجرای برنامه"""
    logging.info("🎯 شروع سیستم تحلیل فارکس (Improved AI Engine)")
    
    import argparse
    parser = argparse.ArgumentParser(description='سیستم تحلیل فارکس با AI')
    parser.add_argument("--pair", type=str, help="تحلیل جفت ارز مشخص")
    parser.add_argument("--all", action="store_true", help="تحلیل همه جفت ارزها")
    parser.add_argument("--pairs", type=str, help="تحلیل جفت ارزهای مشخص شده")
    
    args = parser.parse_args()

    if args.pair:
        pairs_to_analyze = [args.pair]
    elif args.pairs:
        pairs_to_analyze = [p.strip() for p in args.pairs.split(',')]
    elif args.all:
        pairs_to_analyze = CURRENCY_PAIRS_TO_ANALYZE
    else:
        pairs_to_analyze = CURRENCY_PAIRS_TO_ANALYZE[:2]
        logging.info(f"🔍 استفاده از 2 جفت ارز اصلی")

    logging.info(f"🎯 جفت ارزهای مورد تحلیل: {', '.join(pairs_to_analyze)}")

    analyzer = ImprovedForexAnalyzer()
    signals = await analyzer.analyze_all_pairs(pairs_to_analyze)

    # ذخیره سیگنال‌ها
    analyzer.save_signals(signals)

    # نمایش نتایج
    logging.info("📈 خلاصه نتایج:")
    
    strong_count = len([s for s in signals if s.get('AGREEMENT_TYPE') == 'STRONG_CONSENSUS'])
    medium_count = len([s for s in signals if s.get('AGREEMENT_TYPE') == 'MEDIUM_CONSENSUS'])
    weak_count = len([s for s in signals if s.get('AGREEMENT_TYPE') == 'WEAK_CONSENSUS'])
    
    logging.info(f"🎯 سیگنال‌های قوی: {strong_count}")
    logging.info(f"📊 سیگنال‌های متوسط: {medium_count}")
    logging.info(f"📈 سیگنال‌های ضعیف: {weak_count}")
    
    for signal in signals:
        action_icon = "🟢" if signal['ACTION'] == 'BUY' else "🔴" if signal['ACTION'] == 'SELL' else "⚪"
        logging.info(f"  {action_icon} {signal['SYMBOL']}: {signal['ACTION']} (اعتماد: {signal.get('CONFIDENCE', 0)}/10)")

    # نمایش وضعیت نهایی API
    analyzer.api_manager.save_usage_data()
    logging.info(analyzer.api_manager.get_usage_summary())
    
    logging.info("🏁 پایان اجرای سیستم")

if __name__ == "__main__":
    asyncio.run(main())
