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

if not all([google_api_key, TWELVEDATA_API_KEY]):
    raise ValueError("لطفاً کلیدهای API را تنظیم کنید: GOOGLE_API_KEY, TWELVEDATA_API_KEY")

# تنظیمات اصلی سیستم
HIGH_TIMEFRAME = "4h"
LOW_TIMEFRAME = "1h"
CANDLES_TO_FETCH = 300
CURRENCY_PAIRS_TO_ANALYZE = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
    "GBP/JPY", "EUR/JPY", "AUD/JPY", "NZD/USD", "USD/CAD", "EUR/AUD", "CAD/JPY"
]

CACHE_FILE = "signal_cache.json"
CACHE_DURATION_HOURS = 2
LOG_FILE = "trading_log.log"

# مدل‌های AI
GEMINI_MODEL = 'gemini-2.5-flash'
CLOUDFLARE_MODELS = [
    "@cf/meta/llama-4-scout-17b-16e-instruct",
    "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b"  # مدل DeepSeek
]

# راه‌اندازی سیستم لاگ‌گیری پیشرفته
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class AsyncRateLimiter:
    def __init__(self, rate_limit: int, period: int):
        self.rate_limit = rate_limit
        self.period = period
        self.request_timestamps = []
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        async with self._lock:
            while True:
                now = time.time()
                self.request_timestamps = [
                    t for t in self.request_timestamps if now - t < self.period
                ]
                if len(self.request_timestamps) < self.rate_limit:
                    self.request_timestamps.append(now)
                    break
                
                oldest_request_time = self.request_timestamps[0]
                sleep_duration = self.period - (now - oldest_request_time)
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

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
            'volume': ['obv', 'volume_sma_20'],
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
            
            # حجم
            if 'volume' in df.columns and not df['volume'].isnull().all():
                logging.info(f"ستون 'volume' شناسایی شد. محاسبه اندیکاتورهای حجم...")
                df.ta.obv(append=True)
                df['volume_sma_20'] = df['volume'].rolling(20).mean()
            else:
                logging.warning("ستون 'volume' در داده‌ها یافت نشد. اندیکاتورهای OBV و Volume SMA نادیده گرفته شدند.")
            
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
        prev_ltf = ltf_df.iloc[-2] if len(ltf_df) > 1 else last_ltf
        
        # تحلیل روند
        htf_trend = self._analyze_trend(last_htf)
        ltf_trend = self._analyze_trend(last_ltf)
        
        # تحلیل مومنتوم
        momentum = self._analyze_momentum(last_ltf)
        
        # تحلیل سطوح کلیدی
        key_levels = self._analyze_key_levels(htf_df, ltf_df, last_ltf['close'])
        
        # تحلیل الگوهای کندل استیک
        candle_analysis = self._analyze_candle_patterns(ltf_df)
        
        return {
            'symbol': symbol,
            'htf_trend': htf_trend,
            'ltf_trend': ltf_trend,
            'momentum': momentum,
            'key_levels': key_levels,
            'candle_patterns': candle_analysis,
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
        bb_middle = ltf_df.get('BBM_20_2.0', pd.Series([0])).iloc[-1]
        
        support_1 = ltf_df.get('sup_1', pd.Series([0])).iloc[-1]
        resistance_1 = ltf_df.get('res_1', pd.Series([0])).iloc[-1]
        support_2 = ltf_df.get('sup_2', pd.Series([0])).iloc[-1]
        resistance_2 = ltf_df.get('res_2', pd.Series([0])).iloc[-1]
        
        return {
            'dynamic': {
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_middle': bb_middle
            },
            'static': {
                'support_1': support_1,
                'resistance_1': resistance_1,
                'support_2': support_2,
                'resistance_2': resistance_2
            },
            'current_price_position': self._get_price_position(current_price, support_1, resistance_1)
        }

    def _get_price_position(self, price: float, support: float, resistance: float) -> str:
        if resistance == support or resistance <= support:
            return "در محدوده خنثی"
        
        range_size = resistance - support
        position = (price - support) / range_size
        
        if position < 0.3:
            return "نزدیک حمایت"
        elif position > 0.7:
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

# =================================================================================
# --- کلاس مدیریت AI سه‌گانه (Gemini + 2 مدل Cloudflare) ---
# =================================================================================

class TripleAIManager:
    def __init__(self, gemini_api_key: str, cloudflare_api_key: str):
        self.gemini_api_key = gemini_api_key
        self.cloudflare_api_key = cloudflare_api_key
        self.gemini_model = GEMINI_MODEL
        
        # تنظیمات Cloudflare
        self.cloudflare_account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "your_account_id")
        self.cloudflare_models = CLOUDFLARE_MODELS
        
        genai.configure(api_key=gemini_api_key)
    
    async def get_triple_analysis(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        """دریافت تحلیل از سه مدل AI و بررسی توافق"""
        tasks = [
            self._get_gemini_analysis(symbol, technical_analysis),
            self._get_cloudflare_analysis(symbol, technical_analysis, self.cloudflare_models[0], "Llama"),
            self._get_cloudflare_analysis(symbol, technical_analysis, self.cloudflare_models[1], "DeepSeek")
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            gemini_result, llama_result, deepseek_result = results
            
            # لاگ خطاها
            for i, (name, result) in enumerate(zip(["Gemini", "Llama", "DeepSeek"], results)):
                if isinstance(result, Exception):
                    logging.error(f"خطا در {name} برای {symbol}: {result}")
                    results[i] = None
            
            return self._combine_and_classify_signals(symbol, gemini_result, llama_result, deepseek_result, technical_analysis)
            
        except Exception as e:
            logging.error(f"خطا در تحلیل سه‌گانه برای {symbol}: {e}")
            return None
    
    async def _get_gemini_analysis(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        """تحلیل با Gemini"""
        try:
            prompt = self._create_analysis_prompt(symbol, technical_analysis)
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
            prompt = self._create_analysis_prompt(symbol, technical_analysis)
            
            headers = {
                "Authorization": f"Bearer {self.cloudflare_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are an expert forex trading analyst. Provide concise analysis in valid JSON format only."
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

    def _create_analysis_prompt(self, symbol: str, technical_analysis: Dict) -> str:
        """ایجاد پرامپت تحلیل"""
        return f"""
به عنوان یک تحلیلگر حرفه‌ای بازار فارکس، تحلیل تکنیکال زیر را برای جفت ارز {symbol} بررسی کنید و فقط و فقط یک آبجکت JSON معتبر برگردانید.

📊 **وضعیت تکنیکال {symbol}:**
- روند بلندمدت (HTF): {technical_analysis['htf_trend']['direction']} - قدرت: {technical_analysis['htf_trend']['strength']}
- روند کوتاه‌مدت (LTF): {technical_analysis['ltf_trend']['direction']}
- مومنتوم: RSI {technical_analysis['momentum']['rsi']['value']:.1f} ({technical_analysis['momentum']['rsi']['signal']})
- موقعیت قیمت: {technical_analysis['key_levels']['current_price_position']}

🎯 **سطوح کلیدی:**
- مقاومت ۱: {technical_analysis['key_levels']['static']['resistance_1']:.5f}
- حمایت ۱: {technical_analysis['key_levels']['static']['support_1']:.5f}
- مقاومت ۲: {technical_analysis['key_levels']['static']['resistance_2']:.5f}
- حمایت ۲: {technical_analysis['key_levels']['static']['support_2']:.5f}

**لطفاً پاسخ را فقط در قالب JSON زیر ارائه دهید (بدون هیچ متن اضافی):**

{{
  "SYMBOL": "{symbol}",
  "ACTION": "BUY/SELL/HOLD",
  "CONFIDENCE": 1-10,
  "ENTRY_ZONE": "عدد اعشاری (مثال: 1.12345)",
  "STOP_LOSS": "عدد اعشاری (مثال: 1.12000)", 
  "TAKE_PROFIT": "عدد اعشاری (مثال: 1.13000)",
  "RISK_REWARD_RATIO": "نسبت عددی (مثال: 1.5)",
  "ANALYSIS": "تحلیل مختصر فارسی",
  "EXPIRATION_H": "عدد صحیح (مثال: 4)"
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
                logging.info(f"✅ {ai_name} سیگنال برای {symbol}: {signal_data.get('ACTION', 'HOLD')}")
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

    def _combine_and_classify_signals(self, symbol: str, gemini_result: Dict, llama_result: Dict, deepseek_result: Dict, technical_analysis: Dict) -> Optional[Dict]:
        """ترکیب نتایج سه مدل AI و طبقه‌بندی بر اساس توافق"""
        valid_results = []
        
        if gemini_result and self._validate_signal_data(gemini_result, symbol):
            valid_results.append(('Gemini', gemini_result))
        
        if llama_result and self._validate_signal_data(llama_result, symbol):
            valid_results.append(('Llama', llama_result))
        
        if deepseek_result and self._validate_signal_data(deepseek_result, symbol):
            valid_results.append(('DeepSeek', deepseek_result))
        
        if not valid_results:
            logging.info(f"هیچ سیگنال معتبری از مدل‌های AI برای {symbol} دریافت نشد")
            return {
                'SYMBOL': symbol, 
                'ACTION': 'HOLD', 
                'CONFIDENCE': 0,
                'AGREEMENT_LEVEL': 0,
                'AGREEMENT_TYPE': 'NO_CONSENSUS',
                'VALID_MODELS': 0,
                'ANALYSIS': 'عدم وجود سیگنال معتبر از مدل‌های AI'
            }
        
        # شمارش سیگنال‌های مختلف
        action_counts = {}
        for model_name, result in valid_results:
            action = result['ACTION'].upper()
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # تعیین سطح توافق
        total_models = len(valid_results)
        max_agreement = max(action_counts.values())
        agreement_level = max_agreement
        
        if agreement_level >= 2:
            # پیدا کردن اکثریت
            majority_action = max(action_counts, key=action_counts.get)
            agreement_type = 'MAJORITY_CONSENSUS'
            
            # ترکیب سیگنال‌های موافق
            agreeing_results = [result for _, result in valid_results if result['ACTION'].upper() == majority_action]
            combined_signal = self._average_agreeing_signals(symbol, agreeing_results, majority_action)
            
        else:
            # عدم توافق (همه مدل‌ها سیگنال‌های مختلف)
            agreement_type = 'NO_CONSENSUS'
            # انتخاب مدل با بیشترین اعتماد
            highest_confidence_model = max(valid_results, key=lambda x: float(x[1].get('CONFIDENCE', 0)))
            combined_signal = highest_confidence_model[1]
            combined_signal['CONFIDENCE'] = max(1, int(float(combined_signal.get('CONFIDENCE', 5)) - 2))
        
        combined_signal['AGREEMENT_LEVEL'] = agreement_level
        combined_signal['AGREEMENT_TYPE'] = agreement_type
        combined_signal['VALID_MODELS'] = total_models
        combined_signal['CONSENSUS_ANALYSIS'] = self._generate_consensus_analysis(agreement_type, agreement_level, total_models)
        
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
        
        # ذخیره تحلیل‌های جداگانه
        model_analyses = {}
        for result in agreeing_results:
            model_analyses[result['ai_model']] = result.get('ANALYSIS', '')
        
        averaged['MODEL_ANALYSES'] = model_analyses
        averaged['CONSENSUS_DETAIL'] = f"توافق بین {len(agreeing_results)} مدل از {len(agreeing_results)} مدل معتبر"
        
        return averaged

    def _generate_consensus_analysis(self, agreement_type: str, agreement_level: int, total_models: int) -> str:
        """تولید تحلیل توافق"""
        if agreement_type == 'MAJORITY_CONSENSUS':
            if agreement_level == 3:
                return "توافق کامل بین هر سه مدل AI - سیگنال با اعتماد بسیار بالا"
            elif agreement_level == 2:
                return f"توافق بین ۲ مدل از {total_models} مدل - سیگنال با اعتماد بالا"
        else:
            return "عدم توافق بین مدل‌ها - سیگنال با اعتماد پایین"

# =================================================================================
# --- کلاس اصلی تحلیلگر فارکس ---
# =================================================================================

class AdvancedForexAnalyzer:
    def __init__(self):
        self.api_rate_limiter = AsyncRateLimiter(rate_limit=8, period=60)
        self.cache_manager = SmartCacheManager(CACHE_FILE, CACHE_DURATION_HOURS)
        self.technical_analyzer = AdvancedTechnicalAnalyzer()
        self.ai_manager = TripleAIManager(google_api_key, CLOUDFLARE_AI_API_KEY)

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
            
            # تحلیل سه‌گانه AI
            ai_analysis = await self.ai_manager.get_triple_analysis(pair, technical_analysis)
            
            if ai_analysis and ai_analysis.get('ACTION') != 'HOLD':
                self.cache_manager.update_cache(pair, ai_analysis)
                logging.info(f"✅ سیگنال معاملاتی برای {pair}: {ai_analysis['ACTION']} (توافق: {ai_analysis.get('AGREEMENT_LEVEL', 0)})")
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
                async with self.api_rate_limiter:
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
    logging.info("🎯 شروع سیستم تحلیل فارکس پیشرفته (Triple AI v3.0)")
    
    import argparse
    parser = argparse.ArgumentParser(description='سیستم تحلیل فارکس با AI سه‌گانه')
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
    high_confidence_signals = []
    low_confidence_signals = []
    
    for signal in signals:
        agreement_level = signal.get('AGREEMENT_LEVEL', 0)
        if agreement_level >= 2:  # توافق حداقل ۲ مدل از ۳ مدل
            high_confidence_signals.append(signal)
        else:
            low_confidence_signals.append(signal)

    # ذخیره سیگنال‌های با توافق بالا
    if high_confidence_signals:
        high_conf_file = "high_confidence_signals.json"
        cleaned_high_signals = []
        
        for signal in high_confidence_signals:
            cleaned_signal = {
                'SYMBOL': signal.get('SYMBOL', 'Unknown'),
                'ACTION': signal.get('ACTION', 'HOLD'),
                'CONFIDENCE': signal.get('CONFIDENCE', 0),
                'AGREEMENT_LEVEL': signal.get('AGREEMENT_LEVEL', 0),
                'VALID_MODELS': signal.get('VALID_MODELS', 0),
                'AGREEMENT_TYPE': signal.get('AGREEMENT_TYPE', 'UNKNOWN'),
                'ENTRY_ZONE': signal.get('ENTRY_ZONE', 'N/A'),
                'STOP_LOSS': signal.get('STOP_LOSS', 'N/A'),
                'TAKE_PROFIT': signal.get('TAKE_PROFIT', 'N/A'),
                'RISK_REWARD_RATIO': signal.get('RISK_REWARD_RATIO', 'N/A'),
                'EXPIRATION_H': signal.get('EXPIRATION_H', 0),
                'CONSENSUS_ANALYSIS': signal.get('CONSENSUS_ANALYSIS', ''),
                'TIMESTAMP': signal.get('timestamp', datetime.now(UTC).isoformat())
            }
            cleaned_high_signals.append(cleaned_signal)
        
        with open(high_conf_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_high_signals, f, indent=4, ensure_ascii=False)
        
        logging.info(f"✅ {len(high_confidence_signals)} سیگنال با توافق بالا در {high_conf_file} ذخیره شد")

    # ذخیره سیگنال‌های با توافق پایین
    if low_confidence_signals:
        low_conf_file = "low_confidence_signals.json"
        cleaned_low_signals = []
        
        for signal in low_confidence_signals:
            cleaned_signal = {
                'SYMBOL': signal.get('SYMBOL', 'Unknown'),
                'ACTION': signal.get('ACTION', 'HOLD'),
                'CONFIDENCE': signal.get('CONFIDENCE', 0),
                'AGREEMENT_LEVEL': signal.get('AGREEMENT_LEVEL', 0),
                'VALID_MODELS': signal.get('VALID_MODELS', 0),
                'AGREEMENT_TYPE': signal.get('AGREEMENT_TYPE', 'UNKNOWN'),
                'ENTRY_ZONE': signal.get('ENTRY_ZONE', 'N/A'),
                'STOP_LOSS': signal.get('STOP_LOSS', 'N/A'),
                'TAKE_PROFIT': signal.get('TAKE_PROFIT', 'N/A'),
                'RISK_REWARD_RATIO': signal.get('RISK_REWARD_RATIO', 'N/A'),
                'EXPIRATION_H': signal.get('EXPIRATION_H', 0),
                'CONSENSUS_ANALYSIS': signal.get('CONSENSUS_ANALYSIS', ''),
                'TIMESTAMP': signal.get('timestamp', datetime.now(UTC).isoformat())
            }
            cleaned_low_signals.append(cleaned_signal)
        
        with open(low_conf_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_low_signals, f, indent=4, ensure_ascii=False)
        
        logging.info(f"📊 {len(low_confidence_signals)} سیگنال با توافق پایین در {low_conf_file} ذخیره شد")

    # نمایش خلاصه نتایج
    logging.info("📈 خلاصه سیگنال‌های معاملاتی:")
    
    logging.info("🎯 سیگنال‌های با توافق بالا (حداقل ۲ مدل):")
    for signal in high_confidence_signals:
        action_icon = "🟢" if signal['ACTION'] == 'BUY' else "🔴" if signal['ACTION'] == 'SELL' else "⚪"
        logging.info(f"  {action_icon} {signal['SYMBOL']}: {signal['ACTION']} (اعتماد: {signal['CONFIDENCE']}/10, توافق: {signal['AGREEMENT_LEVEL']}/3)")
    
    logging.info("📊 سیگنال‌های با توافق پایین:")
    for signal in low_confidence_signals:
        action_icon = "🟢" if signal['ACTION'] == 'BUY' else "🔴" if signal['ACTION'] == 'SELL' else "⚪"
        logging.info(f"  {action_icon} {signal['SYMBOL']}: {signal['ACTION']} (اعتماد: {signal['CONFIDENCE']}/10, توافق: {signal['AGREEMENT_LEVEL']}/3)")

    if not signals:
        logging.info("🔍 هیچ سیگنال معاملاتی‌ای در این اجرا شناسایی نشد")

    logging.info("🏁 پایان اجرای سیستم")

if __name__ == "__main__":
    asyncio.run(main())
