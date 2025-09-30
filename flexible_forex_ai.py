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

# مدل‌های AI به‌روز شده
GEMINI_MODEL = 'gemini-2.5-flash'

# مدل‌های Cloudflare به‌روز شده
CLOUDFLARE_MODELS = [
    "@cf/meta/llama-4-scout-17b-16e-instruct",  # جدیدترین
    "@cf/google/gemma-3-12b-it",                # جدید
    "@cf/mistralai/mistral-small-3.1-24b-instruct", # جدید 
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast", # جدید
    "@cf/meta/llama-3.1-8b-instruct-fast"      # جدی
]

# مدل‌های Groq به‌روز شده (جدیدترین‌ها)
GROQ_MODELS = [
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",        # جدید
    "openai/gpt-oss-120b",
    "moonshotai/kimi-k2-instruct-0905"
    "llama-3.1-8b-instant"  , # جدید
    "llama-3.3-70b-versatile",                          # جدیدترین
    "meta-llama/llama-4-maverick-17b-128e-instruct"   # جدید 
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
# --- کلاس مدیریت مصرف API هوشمند با قابلیت Fallback ---
# =================================================================================

class SmartAPIManager:
    def __init__(self, usage_file: str):
        self.usage_file = usage_file
        self.usage_data = self.load_usage_data()
        self.available_models = self.initialize_available_models()
        self.failed_models = set()  # مدل‌های شکست خورده
        
    def load_usage_data(self) -> Dict:
        """بارگذاری داده‌های مصرف API"""
        if not os.path.exists(self.usage_file):
            return self.initialize_usage_data()
        
        try:
            with open(self.usage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return self.check_and_reset_daily_usage(data)
        except (json.JSONDecodeError, IOError) as e:
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
        except IOError as e:
            logging.error(f"خطا در ذخیره داده‌های مصرف API: {e}")
    
    def initialize_available_models(self) -> Dict:
        """مقداردهی اولیه مدل‌های موجود"""
        return {
            "google_gemini": [GEMINI_MODEL],
            "cloudflare": CLOUDFLARE_MODELS.copy(),
            "groq": GROQ_MODELS.copy()
        }
    
    def can_use_provider(self, provider: str, models_needed: int = 1) -> bool:
        """بررسی امکان استفاده از provider"""
        if provider not in self.usage_data["providers"]:
            return False
            
        provider_data = self.usage_data["providers"][provider]
        remaining = provider_data["limit"] - provider_data["used_today"]
        
        return remaining >= models_needed
    
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
    
    def select_optimal_models_with_fallback(self, target_total: int = 5, min_required: int = 3) -> List[Tuple[str, str]]:
        """انتخاب بهینه مدل‌ها با قابلیت Fallback"""
        selected_models = []
        
        # محاسبه ظرفیت هر provider
        provider_capacity = {}
        for provider in ["google_gemini", "cloudflare", "groq"]:
            provider_capacity[provider] = self.get_available_models_count(provider)
        
        logging.info(f"📊 ظرفیت providerها: Gemini={provider_capacity['google_gemini']}, "
                    f"Cloudflare={provider_capacity['cloudflare']}, Groq={provider_capacity['groq']}")
        
        # استراتژی انتخاب: اولویت با providerهایی است که ظرفیت بیشتری دارند
        total_available = sum(provider_capacity.values())
        
        if total_available == 0:
            logging.error("❌ هیچ providerی در دسترس نیست")
            return selected_models
        
        # توزیع مدل‌ها بین providerها
        remaining_target = min(target_total, total_available)
        
        # همیشه از Gemini استفاده کن (اگر موجود باشد)
        if provider_capacity["google_gemini"] > 0:
            gemini_model = self.available_models["google_gemini"][0]
            if not self.is_model_failed("google_gemini", gemini_model):
                selected_models.append(("google_gemini", gemini_model))
                remaining_target -= 1
                provider_capacity["google_gemini"] -= 1
        
        # توزیع باقی‌مانده بین Cloudflare و Groq
        providers_order = ["cloudflare", "groq"]
        
        while remaining_target > 0 and any(provider_capacity[p] > 0 for p in providers_order):
            # انتخاب provider با بیشترین ظرفیت باقیمانده
            best_provider = None
            max_capacity = 0
            
            for provider in providers_order:
                if provider_capacity[provider] > max_capacity:
                    max_capacity = provider_capacity[provider]
                    best_provider = provider
            
            if best_provider:
                # انتخاب جدیدترین مدل از این provider که شکست نخورده باشد
                available_in_provider = len(self.available_models[best_provider])
                for i in range(available_in_provider):
                    model_index = available_in_provider - 1 - i  # از جدید به قدیم
                    if model_index < provider_capacity[best_provider]:
                        candidate_model = self.available_models[best_provider][model_index]
                        if not self.is_model_failed(best_provider, candidate_model):
                            selected_models.append((best_provider, candidate_model))
                            provider_capacity[best_provider] -= 1
                            remaining_target -= 1
                            break
        
        # اگر به حداقل نرسیدیم، Fallback به مدل‌های دیگر
        if len(selected_models) < min_required:
            logging.warning(f"⚠️ فقط {len(selected_models)} مدل انتخاب شد. فعال کردن Fallback...")
            
            # جستجو برای مدل‌های اضافی از تمام providerها
            additional_models = []
            for provider in ["google_gemini", "cloudflare", "groq"]:
                available_count = self.get_available_models_count(provider)
                for i in range(min(available_count, 2)):  # حداکثر ۲ مدل از هر provider
                    model_name = self.available_models[provider][i]
                    if (provider, model_name) not in selected_models and not self.is_model_failed(provider, model_name):
                        additional_models.append((provider, model_name))
            
            # اضافه کردن مدل‌های اضافی تا رسیدن به حداقل
            needed_additional = min_required - len(selected_models)
            selected_models.extend(additional_models[:needed_additional])
        
        logging.info(f"🎯 {len(selected_models)} مدل انتخاب شد: {selected_models}")
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
            
            # الگوهای کندل استیک
            if self.indicators_config['candle_patterns']:
                popular_patterns = ['doji', 'hammer', 'engulfing', 'harami', 'morningstar', 'eveningstar']
                for pattern in popular_patterns:
                    try:
                        df.ta.cdl_pattern(name=pattern, append=True)
                    except Exception as e:
                        continue
            
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
        
        # تحلیل الگوهای کندل استیک
        candle_analysis = self._analyze_candle_patterns(ltf_df)
        
        # تحلیل قدرت روند
        trend_strength = self._analyze_trend_strength(htf_df, ltf_df)
        
        return {
            'symbol': symbol,
            'htf_trend': htf_trend,
            'ltf_trend': ltf_trend,
            'momentum': momentum,
            'key_levels': key_levels,
            'candle_patterns': candle_analysis,
            'trend_strength': trend_strength,
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
            'adx': adx,
            'ema_alignment': f"EMA21: {ema_21:.5f}, EMA50: {ema_50:.5f}, EMA200: {ema_200:.5f}"
        }

    def _analyze_momentum(self, data: pd.Series) -> Dict:
        rsi = data.get('RSI_14', 50)
        macd_hist = data.get('MACDh_12_26_9', 0)
        stoch_k = data.get('STOCHk_14_3_3', 50)
        stoch_d = data.get('STOCHd_14_3_3', 50)
        
        rsi_signal = "اشباع خرید" if rsi > 70 else "اشباع فروش" if rsi < 30 else "خنثی"
        macd_signal = "صعودی" if macd_hist > 0 else "نزولی"
        stoch_signal = "اشباع خرید" if stoch_k > 80 else "اشباع فروش" if stoch_k < 20 else "خنثی"
        stoch_cross = "طلایی" if stoch_k > stoch_d and stoch_d < 20 else "مرده" if stoch_k < stoch_d and stoch_d > 80 else "خنثی"
        
        return {
            'rsi': {'value': rsi, 'signal': rsi_signal},
            'macd': {'signal': macd_signal, 'histogram': macd_hist},
            'stochastic': {
                'k': stoch_k, 
                'd': stoch_d, 
                'signal': stoch_signal,
                'cross': stoch_cross
            }
        }

    def _analyze_key_levels(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame, current_price: float) -> Dict:
        bb_upper = ltf_df.get('BBU_20_2.0', pd.Series([0])).iloc[-1]
        bb_lower = ltf_df.get('BBL_20_2.0', pd.Series([0])).iloc[-1]
        bb_middle = ltf_df.get('BBM_20_2.0', pd.Series([0])).iloc[-1]
        
        support_1 = ltf_df.get('sup_1', pd.Series([0])).iloc[-1]
        resistance_1 = ltf_df.get('res_1', pd.Series([0])).iloc[-1]
        support_2 = ltf_df.get('sup_2', pd.Series([0])).iloc[-1]
        resistance_2 = ltf_df.get('res_2', pd.Series([0])).iloc[-1]
        
        return {
            'dynamic': {
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_middle': bb_middle,
                'bb_position': self._get_bb_position(current_price, bb_upper, bb_lower)
            },
            'static': {
                'support_1': support_1,
                'resistance_1': resistance_1,
                'support_2': support_2,
                'resistance_2': resistance_2
            },
            'current_price_position': self._get_price_position(current_price, support_1, resistance_1)
        }

    def _get_bb_position(self, price: float, bb_upper: float, bb_lower: float) -> str:
        if bb_upper == bb_lower:
            return "وسط باند"
        
        bb_width = bb_upper - bb_lower
        position = (price - bb_lower) / bb_width
        
        if position < 0.1:
            return "نزدیک باند پایین"
        elif position > 0.9:
            return "نزدیک باند بالا"
        elif position < 0.3:
            return "قسمت پایین باند"
        elif position > 0.7:
            return "قسمت بالای باند"
        else:
            return "وسط باند"

    def _get_price_position(self, price: float, support: float, resistance: float) -> str:
        if resistance == support or resistance <= support:
            return "در محدوده خنثی"
        
        range_size = resistance - support
        position = (price - support) / range_size
        
        if position < 0.2:
            return "خیلی نزدیک حمایت"
        elif position < 0.4:
            return "نزدیک حمایت"
        elif position > 0.8:
            return "خیلی نزدیک مقاومت"
        elif position > 0.6:
            return "نزدیک مقاومت"
        else:
            return "در میانه رنج"

    def _analyze_candle_patterns(self, df: pd.DataFrame) -> Dict:
        if len(df) < 3:
            return {'patterns': [], 'current_candle': {}, 'recent_patterns': []}
            
        last_candle = df.iloc[-1]
        patterns = []
        
        candle_indicators = [col for col in df.columns if col.startswith('CDL_')]
        for indicator in candle_indicators:
            if abs(last_candle.get(indicator, 0)) > 0:
                pattern_name = indicator.replace('CDL_', '')
                direction = "صعودی" if last_candle[indicator] > 0 else "نزولی"
                patterns.append(f"{pattern_name} ({direction})")
        
        current_candle = self._analyze_single_candle(df.iloc[-1])
        
        return {
            'patterns': patterns,
            'current_candle': current_candle,
            'recent_patterns': patterns[-3:] if patterns else []
        }

    def _analyze_single_candle(self, candle: pd.Series) -> Dict:
        open_price = candle.get('open', 0)
        close = candle.get('close', 0)
        high = candle.get('high', 0)
        low = candle.get('low', 0)
        
        body_size = abs(close - open_price)
        total_range = high - low
        
        if total_range == 0:
            return {"type": "تعریف نشده", "direction": "خنثی", "body_ratio": 0, "strength": "ضعیف"}
            
        body_ratio = body_size / total_range
        
        if body_ratio < 0.3:
            candle_type = "دوجی/فرفره"
        elif body_ratio > 0.7:
            candle_type = "ماروبوزو"
        else:
            candle_type = "عادی"
            
        direction = "صعودی" if close > open_price else "نزولی"
        
        return {
            'type': candle_type,
            'direction': direction,
            'body_ratio': body_ratio,
            'strength': "قوی" if body_ratio > 0.6 else "متوسط" if body_ratio > 0.3 else "ضعیف"
        }

    def _analyze_trend_strength(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> Dict:
        last_htf = htf_df.iloc[-1]
        last_ltf = ltf_df.iloc[-1]
        
        adx_htf = last_htf.get('ADX_14', 0)
        adx_ltf = last_ltf.get('ADX_14', 0)
        
        ema_alignment_htf = self._get_ema_alignment_score(last_htf)
        ema_alignment_ltf = self._get_ema_alignment_score(last_ltf)
        
        trend_strength_score = (
            (adx_htf / 50) * 0.3 +
            (adx_ltf / 50) * 0.2 +
            ema_alignment_htf * 0.3 +
            ema_alignment_ltf * 0.2
        )
        
        if trend_strength_score > 0.7:
            strength_level = "بسیار قوی"
        elif trend_strength_score > 0.5:
            strength_level = "قوی"
        elif trend_strength_score > 0.3:
            strength_level = "متوسط"
        else:
            strength_level = "ضعیف"
        
        return {
            'score': round(trend_strength_score, 2),
            'level': strength_level,
            'adx_htf': adx_htf,
            'adx_ltf': adx_ltf,
            'ema_alignment_htf': ema_alignment_htf,
            'ema_alignment_ltf': ema_alignment_ltf
        }

    def _get_ema_alignment_score(self, data: pd.Series) -> float:
        ema_21 = data.get('EMA_21', 0)
        ema_50 = data.get('EMA_50', 0)
        ema_200 = data.get('EMA_200', 0)
        
        if ema_21 > ema_50 > ema_200:
            return 1.0
        elif ema_21 < ema_50 < ema_200:
            return 1.0
        elif (ema_21 > ema_50 and ema_50 > ema_200) or (ema_21 < ema_50 and ema_50 < ema_200):
            return 0.7
        elif abs(ema_21 - ema_50) < abs(ema_50 - ema_200) * 0.1:
            return 0.3
        else:
            return 0.0

# =================================================================================
# --- کلاس مدیریت AI هوشمند با پرامپت پیشرفته ---
# =================================================================================

class AdvancedAIManager:
    def __init__(self, gemini_api_key: str, cloudflare_api_key: str, groq_api_key: str, api_manager: SmartAPIManager):
        self.gemini_api_key = gemini_api_key
        self.cloudflare_api_key = cloudflare_api_key
        self.groq_api_key = groq_api_key
        self.api_manager = api_manager
        
        genai.configure(api_key=gemini_api_key)
    
    async def get_enhanced_ai_analysis(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        """تحلیل پیشرفته با حداقل ۳ مدل AI"""
        # انتخاب مدل‌ها با قابلیت Fallback
        selected_models = self.api_manager.select_optimal_models_with_fallback(target_total=5, min_required=3)
        
        if len(selected_models) < 3:
            logging.error(f"❌ نمی‌توان حداقل ۳ مدل AI برای تحلیل {symbol} پیدا کرد")
            return None
        
        logging.info(f"🎯 استفاده از {len(selected_models)} مدل AI برای {symbol}")
        
        # ایجاد تسک‌های تحلیل
        tasks = []
        for provider, model_name in selected_models:
            task = self._get_enhanced_single_analysis(symbol, technical_analysis, provider, model_name)
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # پردازش نتایج
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
            
            # اگر کمتر از ۳ مدل موفق داشتیم، Fallback
            if len(valid_results) < 3:
                logging.warning(f"⚠️ فقط {len(valid_results)} مدل موفق بودند. فعال کردن Fallback...")
                additional_results = await self._get_fallback_analysis(symbol, technical_analysis, selected_models, 3 - len(valid_results))
                valid_results.extend(additional_results)
            
            return self._combine_and_classify_signals(symbol, valid_results, technical_analysis, len(selected_models))
            
        except Exception as e:
            logging.error(f"خطا در تحلیل AI برای {symbol}: {e}")
            return None
    
    async def _get_fallback_analysis(self, symbol: str, technical_analysis: Dict, 
                                   used_models: List[Tuple[str, str]], needed_count: int) -> List[Dict]:
        """دریافت تحلیل از مدل‌های جایگزین"""
        if needed_count <= 0:
            return []
        
        fallback_models = []
        used_set = set(used_models)
        
        # جستجو برای مدل‌های جایگزین
        for provider in ["google_gemini", "cloudflare", "groq"]:
            if self.api_manager.can_use_provider(provider):
                for model in self.api_manager.available_models[provider]:
                    if (provider, model) not in used_set and not self.api_manager.is_model_failed(provider, model):
                        fallback_models.append((provider, model))
                        if len(fallback_models) >= needed_count:
                            break
            if len(fallback_models) >= needed_count:
                break
        
        if not fallback_models:
            return []
        
        logging.info(f"🔄 استفاده از {len(fallback_models)} مدل Fallback: {fallback_models}")
        
        tasks = []
        for provider, model_name in fallback_models:
            task = self._get_enhanced_single_analysis(symbol, technical_analysis, provider, model_name)
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            valid_results = []
            for i, (provider, model_name) in enumerate(fallback_models):
                result = results[i]
                if isinstance(result, Exception):
                    logging.error(f"خطا در Fallback {provider}/{model_name}: {result}")
                    self.api_manager.mark_model_failed(provider, model_name)
                    self.api_manager.record_api_usage(provider)
                elif result is not None:
                    valid_results.append(result)
                    self.api_manager.record_api_usage(provider)
                else:
                    self.api_manager.record_api_usage(provider)
            
            return valid_results
            
        except Exception as e:
            logging.error(f"خطا در تحلیل Fallback: {e}")
            return []
    
    async def _get_enhanced_single_analysis(self, symbol: str, technical_analysis: Dict, 
                                          provider: str, model_name: str) -> Optional[Dict]:
        """تحلیل پیشرفته با یک مدل خاص"""
        try:
            fundamental_data = await self._get_fundamental_analysis(symbol)
            prompt = self._create_enhanced_analysis_prompt(symbol, technical_analysis, fundamental_data)
            
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
    
    async def _get_fundamental_analysis(self, symbol: str) -> Dict:
        """دریافت تحلیل فاندامنتال"""
        try:
            # استفاده از APIهای رایگان برای داده‌های فاندامنتال
            base_currency, quote_currency = symbol.split('/')
            
            fundamental_data = {
                'symbol': symbol,
                'base_currency': base_currency,
                'quote_currency': quote_currency,
                'interest_rate_diff': "تحلیل بر اساس آخرین داده‌های موجود",
                'economic_outlook': "رشد اقتصادی و تورم بر اساس آخرین گزارشات",
                'risk_sentiment': "تحلیل بر اساس شرایط بازار جهانی",
                'key_events': "رویدادهای اقتصادی مهم هفته جاری",
                'market_bias': "تمایل کلی بازار بر اساس عوامل فاندامنتال"
            }
            
            return fundamental_data
            
        except Exception as e:
            logging.warning(f"خطا در دریافت داده‌های فاندامنتال: {e}")
            return {
                'symbol': symbol,
                'error': 'عدم دسترسی به داده‌های فاندامنتال'
            }
    
    def _create_enhanced_analysis_prompt(self, symbol: str, technical_analysis: Dict, fundamental_data: Dict) -> str:
        """ایجاد پرامپت تحلیل پیشرفته با فاندامنتال"""
        return f"""
CRITICAL INSTRUCTION: YOU MUST RETURN ONLY VALID JSON FORMAT. NO OTHER TEXT, NO EXPLANATIONS, NO MARKDOWN.

As an expert forex analyst with 20+ years experience, analyze this currency pair using BOTH technical and fundamental analysis:

SYMBOL: {symbol}

TECHNICAL ANALYSIS DATA:
- HTF Trend (4H): {technical_analysis['htf_trend']['direction']} ({technical_analysis['htf_trend']['strength']})
- LTF Trend (1H): {technical_analysis['ltf_trend']['direction']} ({technical_analysis['ltf_trend']['strength']})
- Trend Strength: {technical_analysis['trend_strength']['level']} (Score: {technical_analysis['trend_strength']['score']})
- RSI: {technical_analysis['momentum']['rsi']['value']:.1f} ({technical_analysis['momentum']['rsi']['signal']})
- MACD: {technical_analysis['momentum']['macd']['signal']}
- Stochastic: {technical_analysis['momentum']['stochastic']['signal']} (Cross: {technical_analysis['momentum']['stochastic']['cross']})
- Price Position: {technical_analysis['key_levels']['current_price_position']}
- BB Position: {technical_analysis['key_levels']['dynamic']['bb_position']}
- Support 1: {technical_analysis['key_levels']['static']['support_1']:.5f}
- Resistance 1: {technical_analysis['key_levels']['static']['resistance_1']:.5f}
- Volatility (ATR): {technical_analysis['volatility']:.5f}

FUNDAMENTAL CONSIDERATIONS:
- Currency Pair: {fundamental_data['base_currency']}/{fundamental_data['quote_currency']}
- Interest Rate Outlook: {fundamental_data['interest_rate_diff']}
- Economic Conditions: {fundamental_data['economic_outlook']}
- Market Risk Sentiment: {fundamental_data['risk_sentiment']}
- Key Events: {fundamental_data['key_events']}
- Overall Market Bias: {fundamental_data['market_bias']}

ANALYSIS REQUIREMENTS:
1. Evaluate technical setup strength
2. Consider fundamental drivers
3. Assess risk-reward ratio
4. Determine optimal entry/exit levels
5. Consider market sentiment and positioning

RETURN ONLY THIS EXACT JSON FORMAT (NO OTHER TEXT):
{{
  "SYMBOL": "{symbol}",
  "ACTION": "BUY/SELL/HOLD",
  "CONFIDENCE": 1-10,
  "ENTRY_ZONE": "number",
  "STOP_LOSS": "number", 
  "TAKE_PROFIT": "number",
  "RISK_REWARD_RATIO": "number",
  "ANALYSIS": "brief_farsi_analysis_here",
  "EXPIRATION_H": "number",
  "TRADE_RATIONALE": "key_reasons_farsi"
}}

IMPORTANT: Your response must be ONLY the JSON object, no other text.
"""
    
    async def _get_gemini_analysis(self, symbol: str, prompt: str, model_name: str) -> Optional[Dict]:
        """تحلیل با Gemini"""
        try:
            model = genai.GenerativeModel(model_name)
            
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                request_options={'timeout': 120}
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
                        "content": "You are an expert forex trading analyst. Return ONLY valid JSON format, no other text."
                    },
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
            
            account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "your_account_id")
            url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_name}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=120) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "result" in data and "response" in data["result"]:
                            content = data["result"]["response"]
                            return self._parse_ai_response(content, symbol, f"Cloudflare-{model_name}")
                        elif "response" in data:
                            content = data["response"]
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
                        "content": "You are a professional forex trading expert. Return ONLY valid JSON format, no other text."
                    },
                    {"role": "user", "content": prompt}
                ],
                "model": model_name,
                "temperature": 0.1,
                "max_tokens": 1024,
                "stream": False
            }
            
            url = "https://api.groq.com/openai/v1/chat/completions"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=120) as response:
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
        """پارس کردن پاسخ AI"""
        try:
            cleaned_response = response.strip()
            
            # حذف تمام متن غیر از JSON
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                signal_data = json.loads(json_str)
                
                if self._validate_signal_data(signal_data, symbol):
                    signal_data['ai_model'] = ai_name
                    signal_data['timestamp'] = datetime.now(UTC).isoformat()
                    logging.info(f"✅ {ai_name} سیگنال برای {symbol}: {signal_data.get('ACTION', 'HOLD')} (اعتماد: {signal_data.get('CONFIDENCE', 0)})")
                    return signal_data
            
            logging.warning(f"❌ پاسخ {ai_name} برای {symbol} فاقد فرمت JSON بود")
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
                return False
        
        action = signal_data['ACTION'].upper()
        if action not in ['BUY', 'SELL', 'HOLD']:
            return False
        
        try:
            confidence = float(signal_data['CONFIDENCE'])
            if not (1 <= confidence <= 10):
                return False
        except (ValueError, TypeError):
            return False
        
        return True

    def _combine_and_classify_signals(self, symbol: str, valid_results: List[Dict], technical_analysis: Dict, total_models: int) -> Optional[Dict]:
        """ترکیب نتایج و طبقه‌بندی بر اساس توافق"""
        if not valid_results:
            return None
        
        # شمارش سیگنال‌های مختلف
        action_counts = {}
        for result in valid_results:
            action = result['ACTION'].upper()
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # تعیین سطح توافق
        total_valid_models = len(valid_results)
        max_agreement = max(action_counts.values())
        agreement_level = max_agreement
        
        # طبقه‌بندی
        if agreement_level >= 3:
            agreement_type = 'STRONG_CONSENSUS'
        elif agreement_level == 2:
            agreement_type = 'MEDIUM_CONSENSUS'  
        else:
            agreement_type = 'WEAK_CONSENSUS'
        
        # ترکیب سیگنال‌های موافق
        if agreement_level >= 2:
            majority_action = max(action_counts, key=action_counts.get)
            agreeing_results = [result for result in valid_results if result['ACTION'].upper() == majority_action]
            combined_signal = self._average_signals(symbol, agreeing_results, majority_action)
        else:
            # انتخاب مدل با بیشترین اعتماد
            highest_confidence_model = max(valid_results, key=lambda x: float(x.get('CONFIDENCE', 0)))
            combined_signal = highest_confidence_model
        
        combined_signal['AGREEMENT_LEVEL'] = agreement_level
        combined_signal['AGREEMENT_TYPE'] = agreement_type
        combined_signal['VALID_MODELS'] = total_valid_models
        combined_signal['TOTAL_MODELS'] = total_models
        
        return combined_signal

    def _average_signals(self, symbol: str, agreeing_results: List[Dict], majority_action: str) -> Dict:
        """میانگین‌گیری سیگنال‌ها"""
        if len(agreeing_results) == 1:
            return agreeing_results[0]
        
        averaged = {'SYMBOL': symbol, 'ACTION': majority_action}
        
        # میانگین CONFIDENCE
        confidences = [float(result.get('CONFIDENCE', 5)) for result in agreeing_results]
        averaged['CONFIDENCE'] = round(sum(confidences) / len(confidences), 1)
        
        return averaged

# =================================================================================
# --- کلاس اصلی تحلیلگر فارکس ---
# =================================================================================

class EnhancedForexAnalyzer:
    def __init__(self):
        self.api_manager = SmartAPIManager(USAGE_TRACKER_FILE)
        self.technical_analyzer = AdvancedTechnicalAnalyzer()
        self.ai_manager = AdvancedAIManager(google_api_key, CLOUDFLARE_AI_API_KEY, GROQ_API_KEY, self.api_manager)

    async def analyze_pair(self, pair: str) -> Optional[Dict]:
        """تحلیل کامل یک جفت ارز"""
        logging.info(f"🔍 شروع تحلیل پیشرفته برای {pair}")
        
        try:
            # نمایش وضعیت مصرف API
            logging.info(self.api_manager.get_usage_summary())
            
            # دریافت داده‌های بازار
            htf_df = await self.get_market_data_async(pair, HIGH_TIMEFRAME)
            ltf_df = await self.get_market_data_async(pair, LOW_TIMEFRAME)
            
            if htf_df is None or ltf_df is None:
                return None
            
            # تحلیل تکنیکال
            htf_df_processed = self.technical_analyzer.calculate_advanced_indicators(htf_df)
            ltf_df_processed = self.technical_analyzer.calculate_advanced_indicators(ltf_df)
            
            if htf_df_processed is None or ltf_df_processed is None:
                return None
            
            technical_analysis = self.technical_analyzer.generate_technical_analysis(pair, htf_df_processed, ltf_df_processed)
            
            if not technical_analysis:
                return None
            
            # تحلیل AI پیشرفته
            ai_analysis = await self.ai_manager.get_enhanced_ai_analysis(pair, technical_analysis)
            
            if ai_analysis and ai_analysis.get('ACTION') != 'HOLD':
                logging.info(f"✅ سیگنال معاملاتی برای {pair}: {ai_analysis['ACTION']} (توافق: {ai_analysis.get('AGREEMENT_LEVEL', 0)}/{ai_analysis.get('TOTAL_MODELS', 0)})")
                return ai_analysis
            else:
                logging.info(f"🔍 هیچ سیگنال معاملاتی برای {pair} شناسایی نشد")
                return None
                
        except Exception as e:
            logging.error(f"خطا در تحلیل {pair}: {e}")
            return None

    async def get_market_data_async(self, symbol: str, interval: str, retries: int = 3) -> Optional[pd.DataFrame]:
        """دریافت داده‌های بازار"""
        for attempt in range(retries):
            try:
                url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={CANDLES_TO_FETCH}&apikey={TWELVEDATA_API_KEY}'
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=60) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'values' in data and data['values']:
                                df = pd.DataFrame(data['values'])
                                df = df.iloc[::-1].reset_index(drop=True)
                                
                                numeric_columns = ['open', 'high', 'low', 'close']
                                for col in numeric_columns:
                                    if col in df.columns:
                                        df[col] = pd.to_numeric(df[col], errors='coerce')
                                
                                df = df.dropna(subset=numeric_columns)
                                return df
            except Exception as e:
                await asyncio.sleep(2)
        
        return None

    async def analyze_all_pairs(self, pairs: List[str]) -> List[Dict]:
        """تحلیل همه جفت ارزها"""
        logging.info(f"🚀 شروع تحلیل موازی برای {len(pairs)} جفت ارز")
        
        tasks = [self.analyze_pair(pair) for pair in pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_signals = []
        for result in results:
            if isinstance(result, Dict) and result.get('ACTION') != 'HOLD':
                valid_signals.append(result)
        
        logging.info(f"📊 تحلیل کامل شد. {len(valid_signals)} سیگنال معتبر شناسایی شد")
        return valid_signals

# =================================================================================
# --- تابع اصلی ---
# =================================================================================

async def main():
    """تابع اصلی اجرای برنامه"""
    logging.info("🎯 شروع سیستم تحلیل فارکس پیشرفته (Enhanced AI Engine v6.0)")
    
    import argparse
    parser = argparse.ArgumentParser(description='سیستم تحلیل فارکس با AI پیشرفته')
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
        pairs_to_analyze = CURRENCY_PAIRS_TO_ANALYZE[:3]
        logging.info(f"استفاده از 3 جفت ارز اصلی به صورت پیش‌فرض")

    logging.info(f"🔍 جفت ارزهای مورد تحلیل: {', '.join(pairs_to_analyze)}")
    
    analyzer = EnhancedForexAnalyzer()
    signals = await analyzer.analyze_all_pairs(pairs_to_analyze)

    # تقسیم سیگنال‌ها
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

    # ذخیره سیگنال‌ها
    if strong_signals:
        with open("strong_consensus_signals.json", 'w', encoding='utf-8') as f:
            json.dump(strong_signals, f, indent=4, ensure_ascii=False)
        logging.info(f"🎯 {len(strong_signals)} سیگنال با توافق قوی ذخیره شد")

    if medium_signals:
        with open("medium_consensus_signals.json", 'w', encoding='utf-8') as f:
            json.dump(medium_signals, f, indent=4, ensure_ascii=False)
        logging.info(f"📊 {len(medium_signals)} سیگنال با توافق متوسط ذخیره شد")

    if weak_signals:
        with open("weak_consensus_signals.json", 'w', encoding='utf-8') as f:
            json.dump(weak_signals, f, indent=4, ensure_ascii=False)
        logging.info(f"📈 {len(weak_signals)} سیگنال با توافق ضعیف ذخیره شد")

    # نمایش نتایج
    logging.info("📈 خلاصه سیگنال‌های معاملاتی:")
    
    for category, signals_list, icon in [
        ("سیگنال‌های با توافق قوی", strong_signals, "🎯"),
        ("سیگنال‌های با توافق متوسط", medium_signals, "📊"), 
        ("سیگنال‌های با توافق ضعیف", weak_signals, "📈")
    ]:
        if signals_list:
            logging.info(f"{icon} {category}:")
            for signal in signals_list:
                action_icon = "🟢" if signal['ACTION'] == 'BUY' else "🔴" if signal['ACTION'] == 'SELL' else "⚪"
                logging.info(f"  {action_icon} {signal['SYMBOL']}: {signal['ACTION']} (اعتماد: {signal['CONFIDENCE']}/10, توافق: {signal['AGREEMENT_LEVEL']}/{signal['TOTAL_MODELS']})")

    if not signals:
        logging.info("🔍 هیچ سیگنال معاملاتی‌ای شناسایی نشد")

    # نمایش وضعیت نهایی
    analyzer.api_manager.save_usage_data()
    logging.info(analyzer.api_manager.get_usage_summary())
    
    logging.info("🏁 پایان اجرای سیستم")

if __name__ == "__main__":
    asyncio.run(main())
