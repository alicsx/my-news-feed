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
    "GBP/JPY", "EUR/JPY", "AUD/JPY", "NZD/USD", "USD/CAD",
    "EUR/GBP", "AUD/NZD", "EUR/AUD", "GBP/CHF", "CAD/JPY"
]

CACHE_FILE = "signal_cache.json"
CACHE_DURATION_HOURS = 2
LOG_FILE = "trading_log.log"

# مدل‌های AI
GEMINI_MODEL = 'gemini-2.5-flash'
CLOUDFLARE_MODELS = [
    "@cf/meta/llama-4-scout-17b-16e-instruct",
    "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b"
]
GROQ_MODELS = [
    "gemma-7b-it",
    "mixtral-8x7b-32768"
]

# سیستم مدیریت Rate Limiting
API_RATE_LIMITS = {
    "google_gemini": {"requests_per_minute": 15, "requests_per_day": 1500},
    "cloudflare": {"requests_per_minute": 10, "requests_per_day": 1000},
    "groq": {"requests_per_minute": 30, "requests_per_day": 10000}  # محدودیت واقعی Groq
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

class AdvancedRateLimiter:
    """مدیریت پیشرفته Rate Limiting برای APIهای مختلف"""
    def __init__(self):
        self.limits = API_RATE_LIMITS
        self.usage = {service: {"minute": [], "day": []} for service in self.limits}
        self._lock = asyncio.Lock()
    
    async def acquire(self, service: str):
        async with self._lock:
            now = time.time()
            
            # پاک‌سازی تاریخچه قدیمی
            self.usage[service]["minute"] = [t for t in self.usage[service]["minute"] if now - t < 60]
            self.usage[service]["day"] = [t for t in self.usage[service]["day"] if now - t < 86400]
            
            # بررسی محدودیت دقیقه
            if len(self.usage[service]["minute"]) >= self.limits[service]["requests_per_minute"]:
                oldest = self.usage[service]["minute"][0]
                wait_time = 60 - (now - oldest)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    return await self.acquire(service)
            
            # بررسی محدودیت روزانه
            if len(self.usage[service]["day"]) >= self.limits[service]["requests_per_day"]:
                raise Exception(f"محدودیت روزانه {service} تمام شده است")
            
            # ثبت درخواست
            self.usage[service]["minute"].append(now)
            self.usage[service]["day"].append(now)

# =================================================================================
# --- کلاس مدیریت کش هوشمند ---
# =================================================================================

class SmartCacheManager:
    def __init__(self, cache_file: str, cache_duration_hours: int):
        self.cache_file = cache_file
        self.cache_duration_hours = cache_duration_hours
        self.cache = self.load_cache()
        
    def load_cache(self) -> Dict:
        if not os.path.exists(self.cache_file):
            return {}
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                return self.clean_old_cache(cache)
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"خطا در بارگذاری کش: {e}")
            return {}
    
    def clean_old_cache(self, cache: Dict) -> Dict:
        cleaned_cache = {}
        current_time = datetime.now(UTC)
        
        for pair, cache_data in cache.items():
            if isinstance(cache_data, str):
                last_signal_time = datetime.fromisoformat(cache_data)
                if current_time - last_signal_time < timedelta(hours=self.cache_duration_hours):
                    cleaned_cache[pair] = cache_data
            elif isinstance(cache_data, dict):
                signal_time = datetime.fromisoformat(cache_data.get('timestamp', ''))
                if current_time - signal_time < timedelta(hours=self.cache_duration_hours):
                    cleaned_cache[pair] = cache_data
                    
        return cleaned_cache
    
    def is_pair_on_cooldown(self, pair: str) -> bool:
        if pair not in self.cache:
            return False
            
        cache_data = self.cache[pair]
        if isinstance(cache_data, str):
            last_signal_time = datetime.fromisoformat(cache_data)
        else:
            last_signal_time = datetime.fromisoformat(cache_data.get('timestamp', ''))
            
        if datetime.now(UTC) - last_signal_time < timedelta(hours=self.cache_duration_hours):
            logging.info(f"جفت ارز {pair} در دوره استراحت قرار دارد")
            return True
        return False
    
    def update_cache(self, pair: str, signal_data: Dict = None):
        self.cache[pair] = {
            'timestamp': datetime.now(UTC).isoformat(),
            'signal': signal_data or {}
        }
        self.save_cache()
    
    def save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=4, ensure_ascii=False)
        except IOError as e:
            logging.error(f"خطا در ذخیره کش: {e}")

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
                        logging.warning(f"Could not calculate candle pattern '{pattern}': {e}")
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
        
        # محاسبه فاصله از سطوح کلیدی
        distance_to_resistance = abs(current_price - resistance_1) if resistance_1 > 0 else 0
        distance_to_support = abs(current_price - support_1) if support_1 > 0 else 0
        
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
            'current_price_position': self._get_price_position(current_price, support_1, resistance_1),
            'distances': {
                'to_resistance': distance_to_resistance,
                'to_support': distance_to_support
            }
        }

    def _get_bb_position(self, price: float, bb_upper: float, bb_lower: float) -> str:
        """تعیین موقعیت قیمت در باندهای بولینگر"""
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
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        
        return {
            'type': candle_type,
            'direction': direction,
            'body_ratio': body_ratio,
            'strength': "قوی" if body_ratio > 0.6 else "متوسط" if body_ratio > 0.3 else "ضعیف",
            'upper_shadow_ratio': upper_shadow / total_range if total_range > 0 else 0,
            'lower_shadow_ratio': lower_shadow / total_range if total_range > 0 else 0
        }

    def _analyze_trend_strength(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> Dict:
        """تحلیل قدرت روند با استفاده از چندین اندیکاتور"""
        last_htf = htf_df.iloc[-1]
        last_ltf = ltf_df.iloc[-1]
        
        # قدرت ADX
        adx_htf = last_htf.get('ADX_14', 0)
        adx_ltf = last_ltf.get('ADX_14', 0)
        
        # همسویی EMAها
        ema_alignment_htf = self._get_ema_alignment_score(last_htf)
        ema_alignment_ltf = self._get_ema_alignment_score(last_ltf)
        
        # قدرت مومنتوم
        rsi_htf = last_htf.get('RSI_14', 50)
        rsi_ltf = last_ltf.get('RSI_14', 50)
        
        # محاسبه امتیاز کلی قدرت روند
        trend_strength_score = (
            (adx_htf / 50) * 0.3 +  # ADX اهمیت 30%
            (adx_ltf / 50) * 0.2 +  # ADX اهمیت 20%
            ema_alignment_htf * 0.3 +  # همسویی EMA اهمیت 30%
            ema_alignment_ltf * 0.2    # همسویی EMA اهمیت 20%
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
        """محاسبه امتیاز همسویی EMAها"""
        ema_21 = data.get('EMA_21', 0)
        ema_50 = data.get('EMA_50', 0)
        ema_200 = data.get('EMA_200', 0)
        
        if ema_21 > ema_50 > ema_200:  # صعودی کامل
            return 1.0
        elif ema_21 < ema_50 < ema_200:  # نزولی کامل
            return 1.0
        elif (ema_21 > ema_50 and ema_50 > ema_200) or (ema_21 < ema_50 and ema_50 < ema_200):  # تقریباً همسو
            return 0.7
        elif abs(ema_21 - ema_50) < abs(ema_50 - ema_200) * 0.1:  # نزدیک به هم
            return 0.3
        else:  # نامرتب
            return 0.0

# =================================================================================
# --- کلاس مدیریت 5 هوش مصنوعی (Gemini + 2 Cloudflare + 2 Groq) ---
# =================================================================================

class FiveAIManager:
    def __init__(self, gemini_api_key: str, cloudflare_api_key: str, groq_api_key: str):
        self.gemini_api_key = gemini_api_key
        self.cloudflare_api_key = cloudflare_api_key
        self.groq_api_key = groq_api_key
        self.gemini_model = GEMINI_MODEL
        self.rate_limiter = AdvancedRateLimiter()
        
        # تنظیمات Cloudflare
        self.cloudflare_account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "your_account_id")
        self.cloudflare_models = CLOUDFLARE_MODELS
        
        # تنظیمات Groq
        self.groq_models = GROQ_MODELS
        self.groq_base_url = "https://api.groq.com/openai/v1/chat/completions"
        
        genai.configure(api_key=gemini_api_key)
    
    async def get_five_ai_analysis(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        """دریافت تحلیل از 5 مدل AI و بررسی توافق"""
        tasks = [
            self._get_gemini_analysis(symbol, technical_analysis),
            self._get_cloudflare_analysis(symbol, technical_analysis, self.cloudflare_models[0], "Cloudflare-Llama"),
            self._get_cloudflare_analysis(symbol, technical_analysis, self.cloudflare_models[1], "Cloudflare-DeepSeek"),
            self._get_groq_analysis(symbol, technical_analysis, self.groq_models[0], "Groq-Llama2"),
            self._get_groq_analysis(symbol, technical_analysis, self.groq_models[1], "Groq-Mixtral")
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # پردازش نتایج و مدیریت خطاها
            processed_results = []
            model_names = ["Gemini", "Cloudflare-Llama", "Cloudflare-DeepSeek", "Groq-Llama2", "Groq-Mixtral"]
            
            for i, (name, result) in enumerate(zip(model_names, results)):
                if isinstance(result, Exception):
                    logging.error(f"خطا در {name} برای {symbol}: {result}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)
            
            return self._combine_and_classify_signals(symbol, processed_results, technical_analysis)
            
        except Exception as e:
            logging.error(f"خطا در تحلیل پنج‌گانه برای {symbol}: {e}")
            return None
    
    async def _get_gemini_analysis(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        """تحلیل با Gemini"""
        try:
            await self.rate_limiter.acquire("google_gemini")
            prompt = self._create_advanced_analysis_prompt(symbol, technical_analysis)
            model = genai.GenerativeModel(self.gemini_model)
            
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                request_options={'timeout': 120}
            )
            
            return self._parse_ai_response(response.text, symbol, "Gemini")
            
        except Exception as e:
            logging.warning(f"خطا در تحلیل Gemini برای {symbol}: {e}")
            return None
    
    async def _get_cloudflare_analysis(self, symbol: str, technical_analysis: Dict, model_name: str, model_display_name: str) -> Optional[Dict]:
        """تحلیل با Cloudflare AI"""
        if not self.cloudflare_api_key or self.cloudflare_account_id == "your_account_id":
            logging.warning("کلید یا شناسه حساب Cloudflare API تنظیم نشده است")
            return None
            
        try:
            await self.rate_limiter.acquire("cloudflare")
            prompt = self._create_advanced_analysis_prompt(symbol, technical_analysis)
            
            headers = {
                "Authorization": f"Bearer {self.cloudflare_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are an expert forex trading analyst with 20 years experience. Provide precise technical analysis in valid JSON format only."
                    },
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
            
            url = f"https://api.cloudflare.com/client/v4/accounts/{self.cloudflare_account_id}/ai/run/{model_name}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=120) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "result" in data and "response" in data["result"]:
                            content = data["result"]["response"]
                            return self._parse_ai_response(content, symbol, model_display_name)
                        elif "response" in data:
                            content = data["response"]
                            return self._parse_ai_response(content, symbol, model_display_name)
                        else:
                            logging.warning(f"فرمت پاسخ Cloudflare نامعتبر است: {data}")
                            return None
                    else:
                        error_text = await response.text()
                        logging.warning(f"خطا در پاسخ Cloudflare: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            logging.warning(f"خطا در تحلیل {model_display_name} برای {symbol}: {e}")
            return None

    async def _get_groq_analysis(self, symbol: str, technical_analysis: Dict, model_name: str, model_display_name: str) -> Optional[Dict]:
        """تحلیل با Groq API"""
        if not self.groq_api_key:
            logging.warning("کلید Groq API تنظیم نشده است")
            return None
            
        try:
            await self.rate_limiter.acquire("groq")
            prompt = self._create_advanced_analysis_prompt(symbol, technical_analysis)
            
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional forex trading expert. Analyze the technical data and provide only valid JSON output without any additional text."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "model": model_name,
                "temperature": 0.1,
                "max_tokens": 1024,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.groq_base_url, headers=headers, json=payload, timeout=120) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "choices" in data and len(data["choices"]) > 0:
                            content = data["choices"][0]["message"]["content"]
                            return self._parse_ai_response(content, symbol, model_display_name)
                        else:
                            logging.warning(f"فرمت پاسخ Groq نامعتبر است: {data}")
                            return None
                    else:
                        error_text = await response.text()
                        logging.warning(f"خطا در پاسخ Groq: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            logging.warning(f"خطا در تحلیل {model_display_name} برای {symbol}: {e}")
            return None

    def _create_advanced_analysis_prompt(self, symbol: str, technical_analysis: Dict) -> str:
        """ایجاد پرامپت تحلیل پیشرفته برای تشخیص سیگنال"""
        base_currency, quote_currency = symbol.split('/')
        
        return f"""
ANALYZE THIS FOREX PAIR AS EXPERT TECHNICAL ANALYST:

SYMBOL: {symbol}
CURRENT TECHNICAL SITUATION:

TREND ANALYSIS:
- Higher Timeframe (4H): {technical_analysis['htf_trend']['direction']} trend, Strength: {technical_analysis['htf_trend']['strength']} (ADX: {technical_analysis['htf_trend']['adx']:.1f})
- Lower Timeframe (1H): {technical_analysis['ltf_trend']['direction']} trend, Strength: {technical_analysis['ltf_trend']['strength']} (ADX: {technical_analysis['ltf_trend']['adx']:.1f})
- Overall Trend Strength: {technical_analysis['trend_strength']['level']} (Score: {technical_analysis['trend_strength']['score']})

MOMENTUM INDICATORS:
- RSI 14: {technical_analysis['momentum']['rsi']['value']:.1f} → {technical_analysis['momentum']['rsi']['signal']}
- MACD: {technical_analysis['momentum']['macd']['signal']} (Histogram: {technical_analysis['momentum']['macd']['histogram']:.5f})
- Stochastic: K={technical_analysis['momentum']['stochastic']['k']:.1f}, D={technical_analysis['momentum']['stochastic']['d']:.1f} → {technical_analysis['momentum']['stochastic']['signal']}
- Stochastic Cross: {technical_analysis['momentum']['stochastic']['cross']}

KEY LEVELS & PRICE POSITION:
- Current Position: {technical_analysis['key_levels']['current_price_position']}
- Bollinger Band Position: {technical_analysis['key_levels']['dynamic']['bb_position']}
- Resistance 1: {technical_analysis['key_levels']['static']['resistance_1']:.5f}
- Support 1: {technical_analysis['key_levels']['static']['support_1']:.5f}
- Resistance 2: {technical_analysis['key_levels']['static']['resistance_2']:.5f}
- Support 2: {technical_analysis['key_levels']['static']['support_2']:.5f}

CANDLE PATTERNS:
- Current Candle: {technical_analysis['candle_patterns']['current_candle']['type']} ({technical_analysis['candle_patterns']['current_candle']['direction']}), Strength: {technical_analysis['candle_patterns']['current_candle']['strength']}
- Recent Patterns: {', '.join(technical_analysis['candle_patterns']['recent_patterns']) if technical_analysis['candle_patterns']['recent_patterns'] else 'None'}

VOLATILITY:
- ATR (14): {technical_analysis['volatility']:.5f}

CRITICAL ANALYSIS REQUIRED:
1. Evaluate trend consistency across timeframes
2. Check momentum convergence/divergence
3. Assess price position relative to key levels
4. Identify potential breakout/breakdown levels
5. Evaluate risk-reward ratio based on support/resistance

RETURN ONLY THIS JSON FORMAT (NO OTHER TEXT):
{{
  "SYMBOL": "{symbol}",
  "ACTION": "BUY/SELL/HOLD",
  "CONFIDENCE": 1-10,
  "ENTRY_ZONE": "price_number",
  "STOP_LOSS": "price_number",
  "TAKE_PROFIT": "price_number",
  "RISK_REWARD_RATIO": "number",
  "ANALYSIS": "brief_farsi_analysis",
  "EXPIRATION_H": "hours_number",
  "TRADE_RATIONALE": "key_reasons_farsi"
}}
"""

    def _parse_ai_response(self, response: str, symbol: str, ai_name: str) -> Optional[Dict]:
        """پارس کردن پاسخ AI"""
        try:
            cleaned_response = response.strip()
            
            json_patterns = [
                r'```json\s*(\{.*?\})\s*```',
                r'```\s*(\{.*?\})\s*```',
                r'(\{.*\})'
            ]
            
            json_match = None
            for pattern in json_patterns:
                json_match = re.search(pattern, cleaned_response, re.DOTALL)
                if json_match:
                    break
            
            if json_match:
                json_str = json_match.group(1)
                signal_data = json.loads(json_str)
                
                if not self._validate_signal_data(signal_data, symbol):
                    return None
                
                signal_data['ai_model'] = ai_name
                signal_data['timestamp'] = datetime.now(UTC).isoformat()
                logging.info(f"✅ {ai_name} سیگنال برای {symbol}: {signal_data.get('ACTION', 'HOLD')} (اعتماد: {signal_data.get('CONFIDENCE', 0)})")
                return signal_data
            else:
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

    def _extract_numeric_value(self, value: str) -> Optional[float]:
        """استخراج ایمن مقدار عددی از رشته"""
        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, str):
            return None
            
        match = re.search(r'[-+]?\d*\.\d+|\d+', value.replace(',', ''))
        if match:
            try:
                return float(match.group(0))
            except (ValueError, TypeError):
                return None
        return None

    def _combine_and_classify_signals(self, symbol: str, results: List[Optional[Dict]], technical_analysis: Dict) -> Optional[Dict]:
        """ترکیب نتایج 5 مدل AI و طبقه‌بندی بر اساس توافق"""
        valid_results = []
        model_names = ["Gemini", "Cloudflare-Llama", "Cloudflare-DeepSeek", "Groq-Llama2", "Groq-Mixtral"]
        
        for i, result in enumerate(results):
            if result and self._validate_signal_data(result, symbol):
                valid_results.append((model_names[i], result))
        
        if not valid_results:
            logging.info(f"هیچ سیگنال معتبری از مدل‌های AI برای {symbol} دریافت نشد")
            return {
                'SYMBOL': symbol, 
                'ACTION': 'HOLD', 
                'CONFIDENCE': 0,
                'AGREEMENT_LEVEL': 0,
                'AGREEMENT_TYPE': 'NO_CONSENSUS',
                'VALID_MODELS': 0,
                'TOTAL_MODELS': 5,
                'ANALYSIS': 'عدم وجود سیگنال معتبر از مدل‌های AI'
            }
        
        # شمارش سیگنال‌های مختلف
        action_counts = {}
        for model_name, result in valid_results:
            action = result['ACTION'].upper()
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # تعیین سطح توافق
        total_valid_models = len(valid_results)
        max_agreement = max(action_counts.values())
        agreement_level = max_agreement
        
        if agreement_level >= 3:
            # اکثریت قوی (3 مدل یا بیشتر)
            majority_action = max(action_counts, key=action_counts.get)
            agreement_type = 'STRONG_CONSENSUS'
            
            # ترکیب سیگنال‌های موافق
            agreeing_results = [result for _, result in valid_results if result['ACTION'].upper() == majority_action]
            combined_signal = self._average_agreeing_signals(symbol, agreeing_results, majority_action)
            
        elif agreement_level == 2:
            # اکثریت ضعیف (2 مدل)
            majority_action = max(action_counts, key=action_counts.get)
            agreement_type = 'WEAK_CONSENSUS'
            
            agreeing_results = [result for _, result in valid_results if result['ACTION'].upper() == majority_action]
            combined_signal = self._average_agreeing_signals(symbol, agreeing_results, majority_action)
            # کاهش اعتماد برای توافق ضعیف
            combined_signal['CONFIDENCE'] = max(1, int(float(combined_signal.get('CONFIDENCE', 5)) - 1))
            
        else:
            # عدم توافق
            agreement_type = 'NO_CONSENSUS'
            # انتخاب مدل با بیشترین اعتماد
            highest_confidence_model = max(valid_results, key=lambda x: float(x[1].get('CONFIDENCE', 0)))
            combined_signal = highest_confidence_model[1]
            combined_signal['CONFIDENCE'] = max(1, int(float(combined_signal.get('CONFIDENCE', 5)) - 2))
        
        combined_signal['AGREEMENT_LEVEL'] = agreement_level
        combined_signal['AGREEMENT_TYPE'] = agreement_type
        combined_signal['VALID_MODELS'] = total_valid_models
        combined_signal['TOTAL_MODELS'] = 5
        combined_signal['CONSENSUS_ANALYSIS'] = self._generate_consensus_analysis(agreement_type, agreement_level, total_valid_models)
        combined_signal['MODEL_BREAKDOWN'] = {name: result.get('ACTION', 'HOLD') for name, result in valid_results}
        
        return combined_signal

    def _average_agreeing_signals(self, symbol: str, agreeing_results: List[Dict], majority_action: str) -> Dict:
        """میانگین‌گیری سیگنال‌های موافق"""
        if len(agreeing_results) == 1:
            result = agreeing_results[0]
            result['CONSENSUS_DETAIL'] = f"سیگنال از {result['ai_model']} - نیاز به تأیید بیشتر"
            return result
        
        averaged = {'SYMBOL': symbol, 'ACTION': majority_action}
        
        # میانگین CONFIDENCE
        confidences = [float(result.get('CONFIDENCE', 5)) for result in agreeing_results]
        averaged['CONFIDENCE'] = round(sum(confidences) / len(confidences), 1)
        
        # میانگین مقادیر عددی
        numeric_fields = ['ENTRY_ZONE', 'STOP_LOSS', 'TAKE_PROFIT', 'EXPIRATION_H']
        
        for field in numeric_fields:
            values = []
            for result in agreeing_results:
                val = self._extract_numeric_value(result.get(field, '0'))
                if val is not None:
                    values.append(val)
            
            if values:
                avg_val = sum(values) / len(values)
                if field == 'EXPIRATION_H':
                    averaged[field] = int(round(avg_val))
                else:
                    averaged[field] = round(avg_val, 5)
            else:
                averaged[field] = 0
        
        # سایر فیلدها
        averaged['RISK_REWARD_RATIO'] = agreeing_results[0].get('RISK_REWARD_RATIO', 'N/A')
        
        # ترکیب تحلیل‌ها
        model_analyses = {}
        for result in agreeing_results:
            model_analyses[result['ai_model']] = {
                'analysis': result.get('ANALYSIS', ''),
                'confidence': result.get('CONFIDENCE', 0),
                'trade_rationale': result.get('TRADE_RATIONALE', '')
            }
        
        averaged['MODEL_ANALYSES'] = model_analyses
        averaged['CONSENSUS_DETAIL'] = f"توافق بین {len(agreeing_results)} مدل از {len(agreeing_results)} مدل معتبر"
        
        return averaged

    def _generate_consensus_analysis(self, agreement_type: str, agreement_level: int, total_models: int) -> str:
        """تولید تحلیل توافق"""
        if agreement_type == 'STRONG_CONSENSUS':
            if agreement_level >= 4:
                return "توافق بسیار قوی بین اکثریت مدل‌ها - سیگنال با اعتماد بسیار بالا"
            elif agreement_level == 3:
                return f"توافق قوی بین ۳ مدل از {total_models} مدل - سیگنال با اعتماد بالا"
        elif agreement_type == 'WEAK_CONSENSUS':
            return f"توافق ضعیف بین ۲ مدل از {total_models} مدل - سیگنال با اعتماد متوسط"
        else:
            return "عدم توافق بین مدل‌ها - سیگنال با اعتماد پایین"

# =================================================================================
# --- کلاس اصلی تحلیلگر فارکس ---
# =================================================================================

class AdvancedForexAnalyzer:
    def __init__(self):
        self.api_rate_limiter = AdvancedRateLimiter()
        self.cache_manager = SmartCacheManager(CACHE_FILE, CACHE_DURATION_HOURS)
        self.technical_analyzer = AdvancedTechnicalAnalyzer()
        self.ai_manager = FiveAIManager(google_api_key, CLOUDFLARE_AI_API_KEY, GROQ_API_KEY)

    async def analyze_pair(self, pair: str) -> Optional[Dict]:
        """تحلیل کامل یک جفت ارز"""
        if self.cache_manager.is_pair_on_cooldown(pair):
            return None
        
        logging.info(f"🔍 شروع تحلیل پیشرفته برای {pair}")
        
        try:
            # دریافت داده‌های بازار
            htf_df = await self.get_market_data_async(pair, HIGH_TIMEFRAME)
            ltf_df = await self.get_market_data_async(pair, LOW_TIMEFRAME)
            
            if htf_df is None or ltf_df is None or htf_df.empty or ltf_df.empty:
                logging.warning(f"داده‌های بازار برای {pair} دریافت نشد یا خالی است")
                return None
            
            # تحلیل تکنیکال
            htf_df_processed = self.technical_analyzer.calculate_advanced_indicators(htf_df)
            ltf_df_processed = self.technical_analyzer.calculate_advanced_indicators(ltf_df)
            
            if htf_df_processed is None or ltf_df_processed is None:
                logging.warning(f"خطا در محاسبه اندیکاتورها برای {pair}")
                return None
            
            technical_analysis = self.technical_analyzer.generate_technical_analysis(pair, htf_df_processed, ltf_df_processed)
            
            if not technical_analysis:
                logging.warning(f"تحلیل تکنیکال برای {pair} ناموفق بود")
                return None
            
            # تحلیل پنج‌گانه AI
            ai_analysis = await self.ai_manager.get_five_ai_analysis(pair, technical_analysis)
            
            if ai_analysis and ai_analysis.get('ACTION') != 'HOLD':
                self.cache_manager.update_cache(pair, ai_analysis)
                logging.info(f"✅ سیگنال معاملاتی برای {pair}: {ai_analysis['ACTION']} (توافق: {ai_analysis.get('AGREEMENT_LEVEL', 0)}/5)")
                return ai_analysis
            else:
                logging.info(f"🔍 هیچ سیگنال معاملاتی برای {pair} شناسایی نشد")
                return None
                
        except Exception as e:
            logging.error(f"خطا در تحلیل {pair}: {e}")
            return None

    async def get_market_data_async(self, symbol: str, interval: str, retries: int = 3) -> Optional[pd.DataFrame]:
        """دریافت داده‌های بازار به صورت آسنکرون"""
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
                                
                                if 'datetime' in df.columns:
                                    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                                
                                df = df.dropna(subset=numeric_columns)
                                
                                if len(df) > 0:
                                    return df
                                else:
                                    logging.warning(f"داده‌های {symbol} پس از پاک‌سازی خالی است")
                                    return None
                            else:
                                logging.warning(f"داده‌های {symbol} خالی است یا ساختار نامعتبر دارد")
                                return None
                        else:
                            logging.warning(f"خطای HTTP {response.status} برای {symbol}")
                            if response.status == 429:
                                await asyncio.sleep(10)
                            
            except Exception as e:
                logging.warning(f"خطا در دریافت داده‌های {symbol} (تلاش {attempt + 1}): {e}")
                await asyncio.sleep(2)
        
        logging.error(f"عدم موفقیت در دریافت داده‌های {symbol} پس از {retries} تلاش")
        return None

    async def analyze_all_pairs(self, pairs: List[str]) -> List[Dict]:
        """تحلیل همه جفت ارزها به صورت موازی"""
        logging.info(f"🚀 شروع تحلیل موازی برای {len(pairs)} جفت ارز")
        
        semaphore = asyncio.Semaphore(2)
        
        async def bounded_analyze(pair):
            async with semaphore:
                result = await self.analyze_pair(pair)
                await asyncio.sleep(1)
                return result
        
        tasks = [bounded_analyze(pair) for pair in pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_signals = []
        for result in results:
            if isinstance(result, Dict) and result.get('ACTION') != 'HOLD':
                valid_signals.append(result)
            elif isinstance(result, Exception):
                logging.error(f"خطا در تحلیل: {result}")
        
        logging.info(f"📊 تحلیل کامل شد. {len(valid_signals)} سیگنال معتبر شناسایی شد")
        return valid_signals

# =================================================================================
# --- تابع اصلی ---
# =================================================================================

async def main():
    """تابع اصلی اجرای برنامه"""
    logging.info("🎯 شروع سیستم تحلیل فارکس پیشرفته (5 AI Engine v4.0)")
    
    import argparse
    parser = argparse.ArgumentParser(description='سیستم تحلیل فارکس با 5 هوش مصنوعی')
    parser.add_argument("--pair", type=str, help="تحلیل جفت ارز مشخص (مثال: EUR/USD)")
    parser.add_argument("--all", action="store_true", help="تحلیل همه جفت ارزهای پیش‌فرض")
    parser.add_argument("--pairs", type=str, help="تحلیل جفت ارزهای مشخص شده (جدا شده با کاما)")
    
    args = parser.parse_args()

    if args.pair:
        pairs_to_analyze = [args.pair]
    elif args.pairs:
        pairs_to_analyze = [p.strip() for p in args.pairs.split(',')]
    elif args.all:
        pairs_to_analyze = CURRENCY_PAIRS_TO_ANALYZE
    else:
        pairs_to_analyze = CURRENCY_PAIRS_TO_ANALYZE[:5]
        logging.info(f"استفاده از 5 جفت ارز اصلی به صورت پیش‌فرض")

    logging.info(f"🔍 جفت ارزهای مورد تحلیل: {', '.join(pairs_to_analyze)}")
    
    analyzer = AdvancedForexAnalyzer()
    signals = await analyzer.analyze_all_pairs(pairs_to_analyze)

    # تقسیم سیگنال‌ها بر اساس سطح توافق
    strong_consensus_signals = []  # 4-5 موافق
    medium_consensus_signals = []  # 3 موافق
    weak_consensus_signals = []    # 2 موافق
    no_consensus_signals = []      # 0-1 موافق
    
    for signal in signals:
        agreement_level = signal.get('AGREEMENT_LEVEL', 0)
        if agreement_level >= 4:
            strong_consensus_signals.append(signal)
        elif agreement_level == 3:
            medium_consensus_signals.append(signal)
        elif agreement_level == 2:
            weak_consensus_signals.append(signal)
        else:
            no_consensus_signals.append(signal)

    # ذخیره سیگنال‌های با توافق قوی (4-5 موافق)
    if strong_consensus_signals:
        strong_conf_file = "strong_consensus_signals.json"
        cleaned_strong_signals = []
        
        for signal in strong_consensus_signals:
            cleaned_signal = {
                'SYMBOL': signal.get('SYMBOL', 'Unknown'),
                'ACTION': signal.get('ACTION', 'HOLD'),
                'CONFIDENCE': signal.get('CONFIDENCE', 0),
                'AGREEMENT_LEVEL': signal.get('AGREEMENT_LEVEL', 0),
                'VALID_MODELS': signal.get('VALID_MODELS', 0),
                'TOTAL_MODELS': signal.get('TOTAL_MODELS', 5),
                'AGREEMENT_TYPE': signal.get('AGREEMENT_TYPE', 'UNKNOWN'),
                'ENTRY_ZONE': signal.get('ENTRY_ZONE', 'N/A'),
                'STOP_LOSS': signal.get('STOP_LOSS', 'N/A'),
                'TAKE_PROFIT': signal.get('TAKE_PROFIT', 'N/A'),
                'RISK_REWARD_RATIO': signal.get('RISK_REWARD_RATIO', 'N/A'),
                'EXPIRATION_H': signal.get('EXPIRATION_H', 0),
                'CONSENSUS_ANALYSIS': signal.get('CONSENSUS_ANALYSIS', ''),
                'TRADE_RATIONALE': signal.get('TRADE_RATIONALE', ''),
                'MODEL_BREAKDOWN': signal.get('MODEL_BREAKDOWN', {}),
                'TIMESTAMP': signal.get('timestamp', datetime.now(UTC).isoformat())
            }
            cleaned_strong_signals.append(cleaned_signal)
        
        with open(strong_conf_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_strong_signals, f, indent=4, ensure_ascii=False)
        
        logging.info(f"🎯 {len(strong_consensus_signals)} سیگنال با توافق قوی در {strong_conf_file} ذخیره شد")

    # ذخیره سیگنال‌های با توافق متوسط (3 موافق)
    if medium_consensus_signals:
        medium_conf_file = "medium_consensus_signals.json"
        cleaned_medium_signals = []
        
        for signal in medium_consensus_signals:
            cleaned_signal = {
                'SYMBOL': signal.get('SYMBOL', 'Unknown'),
                'ACTION': signal.get('ACTION', 'HOLD'),
                'CONFIDENCE': signal.get('CONFIDENCE', 0),
                'AGREEMENT_LEVEL': signal.get('AGREEMENT_LEVEL', 0),
                'VALID_MODELS': signal.get('VALID_MODELS', 0),
                'TOTAL_MODELS': signal.get('TOTAL_MODELS', 5),
                'AGREEMENT_TYPE': signal.get('AGREEMENT_TYPE', 'UNKNOWN'),
                'ENTRY_ZONE': signal.get('ENTRY_ZONE', 'N/A'),
                'STOP_LOSS': signal.get('STOP_LOSS', 'N/A'),
                'TAKE_PROFIT': signal.get('TAKE_PROFIT', 'N/A'),
                'RISK_REWARD_RATIO': signal.get('RISK_REWARD_RATIO', 'N/A'),
                'EXPIRATION_H': signal.get('EXPIRATION_H', 0),
                'CONSENSUS_ANALYSIS': signal.get('CONSENSUS_ANALYSIS', ''),
                'TRADE_RATIONALE': signal.get('TRADE_RATIONALE', ''),
                'MODEL_BREAKDOWN': signal.get('MODEL_BREAKDOWN', {}),
                'TIMESTAMP': signal.get('timestamp', datetime.now(UTC).isoformat())
            }
            cleaned_medium_signals.append(cleaned_signal)
        
        with open(medium_conf_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_medium_signals, f, indent=4, ensure_ascii=False)
        
        logging.info(f"📊 {len(medium_consensus_signals)} سیگنال با توافق متوسط در {medium_conf_file} ذخیره شد")

    # ذخیره سیگنال‌های با توافق ضعیف و بدون توافق
    other_signals = weak_consensus_signals + no_consensus_signals
    if other_signals:
        other_conf_file = "other_signals.json"
        cleaned_other_signals = []
        
        for signal in other_signals:
            cleaned_signal = {
                'SYMBOL': signal.get('SYMBOL', 'Unknown'),
                'ACTION': signal.get('ACTION', 'HOLD'),
                'CONFIDENCE': signal.get('CONFIDENCE', 0),
                'AGREEMENT_LEVEL': signal.get('AGREEMENT_LEVEL', 0),
                'VALID_MODELS': signal.get('VALID_MODELS', 0),
                'TOTAL_MODELS': signal.get('TOTAL_MODELS', 5),
                'AGREEMENT_TYPE': signal.get('AGREEMENT_TYPE', 'UNKNOWN'),
                'ENTRY_ZONE': signal.get('ENTRY_ZONE', 'N/A'),
                'STOP_LOSS': signal.get('STOP_LOSS', 'N/A'),
                'TAKE_PROFIT': signal.get('TAKE_PROFIT', 'N/A'),
                'RISK_REWARD_RATIO': signal.get('RISK_REWARD_RATIO', 'N/A'),
                'EXPIRATION_H': signal.get('EXPIRATION_H', 0),
                'CONSENSUS_ANALYSIS': signal.get('CONSENSUS_ANALYSIS', ''),
                'TRADE_RATIONALE': signal.get('TRADE_RATIONALE', ''),
                'MODEL_BREAKDOWN': signal.get('MODEL_BREAKDOWN', {}),
                'TIMESTAMP': signal.get('timestamp', datetime.now(UTC).isoformat())
            }
            cleaned_other_signals.append(cleaned_signal)
        
        with open(other_conf_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_other_signals, f, indent=4, ensure_ascii=False)
        
        logging.info(f"📈 {len(other_signals)} سیگنال با توافق ضعیف/بدون توافق در {other_conf_file} ذخیره شد")

    # نمایش خلاصه نتایج
    logging.info("📈 خلاصه سیگنال‌های معاملاتی:")
    
    logging.info("🎯 سیگنال‌های با توافق قوی (4-5 موافق):")
    for signal in strong_consensus_signals:
        action_icon = "🟢" if signal['ACTION'] == 'BUY' else "🔴" if signal['ACTION'] == 'SELL' else "⚪"
        logging.info(f"  {action_icon} {signal['SYMBOL']}: {signal['ACTION']} (اعتماد: {signal['CONFIDENCE']}/10, توافق: {signal['AGREEMENT_LEVEL']}/5)")
    
    logging.info("📊 سیگنال‌های با توافق متوسط (3 موافق):")
    for signal in medium_consensus_signals:
        action_icon = "🟢" if signal['ACTION'] == 'BUY' else "🔴" if signal['ACTION'] == 'SELL' else "⚪"
        logging.info(f"  {action_icon} {signal['SYMBOL']}: {signal['ACTION']} (اعتماد: {signal['CONFIDENCE']}/10, توافق: {signal['AGREEMENT_LEVEL']}/5)")
    
    logging.info("📈 سیگنال‌های با توافق ضعیف (2 موافق):")
    for signal in weak_consensus_signals:
        action_icon = "🟢" if signal['ACTION'] == 'BUY' else "🔴" if signal['ACTION'] == 'SELL' else "⚪"
        logging.info(f"  {action_icon} {signal['SYMBOL']}: {signal['ACTION']} (اعتماد: {signal['CONFIDENCE']}/10, توافق: {signal['AGREEMENT_LEVEL']}/5)")

    if not signals:
        logging.info("🔍 هیچ سیگنال معاملاتی‌ای در این اجرا شناسایی نشد")

    logging.info("🏁 پایان اجرای سیستم")

if __name__ == "__main__":
    asyncio.run(main())
