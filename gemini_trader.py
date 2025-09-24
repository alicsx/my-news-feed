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
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

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
CACHE_DURATION_HOURS = 2  # منطبق با اجرای هر 2 ساعت
LOG_FILE = "trading_log.log"

# مدل‌های AI
GEMINI_MODEL = 'gemini-1.5-flash-latest'  # سریع‌تر و اقتصادی‌تر
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

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
# --- کلاس مدیریت کش هوشمند ---
# =================================================================================

class SmartCacheManager:
    def __init__(self, cache_file: str, cache_duration_hours: int):
        self.cache_file = cache_file
        self.cache_duration_hours = cache_duration_hours
        self.cache = self.load_cache()
        
    def load_cache(self) -> Dict:
        """بارگذاری کش با قابلیت بازیابی از خطا"""
        if not os.path.exists(self.cache_file):
            return {}
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                # پاک‌سازی کش‌های قدیمی
                return self.clean_old_cache(cache)
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"خطا در بارگذاری کش: {e}")
            return {}
    
    def clean_old_cache(self, cache: Dict) -> Dict:
        """حذف ورودی‌های قدیمی از کش"""
        cleaned_cache = {}
        current_time = datetime.now(UTC)
        
        for pair, cache_data in cache.items():
            if isinstance(cache_data, str):
                # فرمت قدیمی
                last_signal_time = datetime.fromisoformat(cache_data)
                if current_time - last_signal_time < timedelta(hours=self.cache_duration_hours):
                    cleaned_cache[pair] = cache_data
            elif isinstance(cache_data, dict):
                # فرمت جدید با داده‌های کامل
                signal_time = datetime.fromisoformat(cache_data.get('timestamp', ''))
                if current_time - signal_time < timedelta(hours=self.cache_duration_hours):
                    cleaned_cache[pair] = cache_data
                    
        return cleaned_cache
    
    def is_pair_on_cooldown(self, pair: str) -> bool:
        """بررسی وضعیت cooldown برای جفت ارز"""
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
        """به‌روزرسانی کش با داده‌های سیگنال"""
        self.cache[pair] = {
            'timestamp': datetime.now(UTC).isoformat(),
            'signal': signal_data or {}
        }
        self.save_cache()
    
    def save_cache(self):
        """ذخیره ایمن کش"""
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
        """محاسبه اندیکاتورهای پیشرفته با مدیریت خطا"""
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
                popular_patterns = ['doji', 'hammer', 'engulfing', 'harami', 'morning_star', 'evening_star']
                for pattern in popular_patterns:
                    try:
                        df.ta.cdl_pattern(pattern, append=True)
                    except:
                        continue
            
            df.dropna(inplace=True)
            return df
            
        except Exception as e:
            logging.error(f"خطا در محاسبه اندیکاتورها: {e}")
            return None
    
    def generate_technical_analysis(self, symbol: str, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> Dict:
        """تولید تحلیل تکنیکال جامع"""
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
        """تحلیل روند بر اساس اندیکاتورها"""
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
        """تحلیل مومنتوم"""
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
        """تحلیل سطوح حمایت و مقاومت"""
        # سطوح داینامیک از باندهای بولینگر
        bb_upper = ltf_df['BBU_20_2.0'].iloc[-1]
        bb_lower = ltf_df['BBL_20_2.0'].iloc[-1]
        bb_middle = ltf_df['BBM_20_2.0'].iloc[-1]
        
        # سطوح استاتیک
        support_1 = ltf_df['sup_1'].iloc[-1]
        resistance_1 = ltf_df['res_1'].iloc[-1]
        support_2 = ltf_df['sup_2'].iloc[-1]
        resistance_2 = ltf_df['res_2'].iloc[-1]
        
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
        """تعیین موقعیت قیمت نسبت به سطوح"""
        range_size = resistance - support
        if range_size == 0:
            return "در محدوده خنثی"
        
        position = (price - support) / range_size
        if position < 0.3:
            return "نزدیک حمایت"
        elif position > 0.7:
            return "نزدیک مقاومت"
        else:
            return "در میانه رنج"
    
    def _analyze_candle_patterns(self, df: pd.DataFrame) -> Dict:
        """تحلیل الگوهای کندل استیک"""
        if len(df) < 3:
            return {}
            
        last_candle = df.iloc[-1]
        patterns = []
        
        # بررسی الگوهای محبوب
        candle_indicators = [col for col in df.columns if col.startswith('CDL_')]
        for indicator in candle_indicators:
            if abs(last_candle.get(indicator, 0)) > 0:
                pattern_name = indicator.replace('CDL_', '')
                direction = "صعودی" if last_candle[indicator] > 0 else "نزولی"
                patterns.append(f"{pattern_name} ({direction})")
        
        # تحلیل ساختار کندل فعلی
        current_candle = self._analyze_single_candle(df.iloc[-1])
        
        return {
            'patterns': patterns,
            'current_candle': current_candle,
            'recent_patterns': patterns[-3:] if patterns else []
        }
    
    def _analyze_single_candle(self, candle: pd.Series) -> Dict:
        """تحلیل تک کندل"""
        open_price = candle['open']
        close = candle['close']
        high = candle['high']
        low = candle['low']
        
        body_size = abs(close - open_price)
        total_range = high - low
        
        if total_range == 0:
            return {"type": "تعریف نشده"}
            
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
# --- کلاس مدیریت AI ترکیبی ---
# =================================================================================

class HybridAIManager:
    def __init__(self, gemini_api_key: str, deepseek_api_key: str):
        self.gemini_api_key = gemini_api_key
        self.deepseek_api_key = deepseek_api_key
        self.gemini_model = GEMINI_MODEL
        self.deepseek_url = DEEPSEEK_API_URL
        
        # تنظیم Gemini
        genai.configure(api_key=gemini_api_key)
    
    async def get_combined_analysis(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        """دریافت تحلیل ترکیبی از دو مدل AI"""
        tasks = [
            self._get_gemini_analysis(symbol, technical_analysis),
            self._get_deepseek_analysis(symbol, technical_analysis)
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            gemini_result, deepseek_result = results
            
            return self._combine_analyses(symbol, gemini_result, deepseek_result, technical_analysis)
            
        except Exception as e:
            logging.error(f"خطا در تحلیل ترکیبی برای {symbol}: {e}")
            return None
    
    async def _get_gemini_analysis(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        """تحلیل با Gemini"""
        try:
            prompt = self._create_analysis_prompt(symbol, technical_analysis, "Gemini")
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
    
    async def _get_deepseek_analysis(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        """تحلیل با DeepSeek"""
        if not self.deepseek_api_key:
            logging.warning("کلید DeepSeek API تنظیم نشده است")
            return None
            
        try:
            prompt = self._create_analysis_prompt(symbol, technical_analysis, "DeepSeek")
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.deepseek_api_key}"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are an expert forex trading analyst."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1500
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.deepseek_url, headers=headers, json=payload, timeout=120) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"]
                        return self._parse_ai_response(content, symbol, "DeepSeek")
                    else:
                        logging.warning(f"خطا در پاسخ DeepSeek: {response.status}")
                        return None
                        
        except Exception as e:
            logging.warning(f"خطا در تحلیل DeepSeek برای {symbol}: {e}")
            return None
    
    def _create_analysis_prompt(self, symbol: str, technical_analysis: Dict, ai_name: str) -> str:
        """ایجاد پرمپت تحلیل"""
        base_currency, quote_currency = symbol.split('/')
        
        return f"""
به عنوان یک تحلیلگر حرفه‌ای بازار فارکس، تحلیل تکنیکال زیر را برای جفت ارز {symbol} بررسی کنید:

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

🕯️ **الگوهای کندلی:**
{chr(10).join(technical_analysis['candle_patterns']['patterns'][-3:]) if technical_analysis['candle_patterns']['patterns'] else 'الگوی خاصی شناسایی نشد'}

**لطفاً تحلیل خود را ارائه داده و در صورت وجود سیگنال معتبر، موارد زیر را مشخص کنید:**

```json
{{
  "SYMBOL": "{symbol}",
  "ACTION": "BUY/SELL/HOLD",
  "CONFIDENCE": 1-10,
  "ENTRY_ZONE": "محدوده ورود",
  "STOP_LOSS": "حد ضرر",
  "TAKE_PROFIT": "حد سود", 
  "RISK_REWARD_RATIO": "نسبت risk/reward",
  "ANALYSIS": "تحلیل کلی وضعیت"
}}
در صورت عدم وجود سیگنال واضح، از ACTION: "HOLD" استفاده کنید.
"""
def _parse_ai_response(self, response: str, symbol: str, ai_name: str) -> Optional[Dict]:
    """پارس کردن پاسخ AI"""
    try:
        # جستجوی JSON در پاسخ
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if not json_match:
            json_match = re.search(r'(\{.*?\})', response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
            signal_data = json.loads(json_str)
            
            # افزودن متا دیتا
            signal_data['ai_model'] = ai_name
            signal_data['timestamp'] = datetime.now(UTC).isoformat()
            
            logging.info(f"✅ {ai_name} سیگنال برای {symbol}: {signal_data.get('ACTION', 'HOLD')}")
            return signal_data
        else:
            logging.warning(f"❌ پاسخ {ai_name} برای {symbol} فاقد فرمت JSON بود")
            return None
            
    except Exception as e:
        logging.error(f"خطا در پارس کردن پاسخ {ai_name} برای {symbol}: {e}")
        return None

def _combine_analyses(self, symbol: str, gemini_result: Dict, deepseek_result: Dict, technical_analysis: Dict) -> Optional[Dict]:
    """ترکیب نتایج دو مدل AI"""
    results = []
    
    if gemini_result and gemini_result.get('ACTION') != 'HOLD':
        results.append(('Gemini', gemini_result))
    if deepseek_result and deepseek_result.get('ACTION') != 'HOLD':
        results.append(('DeepSeek', deepseek_result))
    
    if not results:
        logging.info(f"هر دو مدل AI برای {symbol} سیگنال HOLD دادند")
        return {
            'SYMBOL': symbol,
            'ACTION': 'HOLD',
            'CONFIDENCE': 0,
            'COMBINED_ANALYSIS': True,
            'MODELS_AGREE': True,
            'ANALYSIS': 'عدم وجود سیگنال واضح از هر دو مدل'
        }
    
    # اگر فقط یک مدل سیگنال داد
    if len(results) == 1:
        model_name, result = results[0]
        result['COMBINED_ANALYSIS'] = True
        result['MODELS_AGREE'] = False
        result['CONFIDENCE'] = max(1, result.get('CONFIDENCE', 5) - 2)  # کاهش اعتماد
        result['ANALYSIS'] = f"سیگنال از {model_name} - نیاز به تأیید بیشتر"
        return result
    
    # اگر هر دو مدل سیگنال دادند
    gemini_action = gemini_result.get('ACTION')
    deepseek_action = deepseek_result.get('ACTION')
    
    if gemini_action == deepseek_action:
        # توافق کامل
        combined_confidence = (gemini_result.get('CONFIDENCE', 5) + deepseek_result.get('CONFIDENCE', 5)) // 2
        return {
            'SYMBOL': symbol,
            'ACTION': gemini_action,
            'CONFIDENCE': min(10, combined_confidence + 1),  # افزایش اعتماد به دلیل توافق
            'COMBINED_ANALYSIS': True,
            'MODELS_AGREE': True,
            'GEMINI_ANALYSIS': gemini_result.get('ANALYSIS', ''),
            'DEEPSEEK_ANALYSIS': deepseek_result.get('ANALYSIS', ''),
            'FINAL_ANALYSIS': f"توافق کامل بین مدل‌ها - سیگنال {gemini_action} با اعتماد بالا"
        }
    else:
        # تضاد بین مدل‌ها - انتخاب محتاطانه
        gemini_conf = gemini_result.get('CONFIDENCE', 5)
        deepseek_conf = deepseek_result.get('CONFIDENCE', 5)
        
        if abs(gemini_conf - deepseek_conf) >= 3:
            # انتخاب مدل با اعتماد بالاتر
            selected_result = gemini_result if gemini_conf > deepseek_conf else deepseek_result
            selected_model = 'Gemini' if gemini_conf > deepseek_conf else 'DeepSeek'
            
            selected_result['COMBINED_ANALYSIS'] = True
            selected_result['MODELS_AGREE'] = False
            selected_result['ANALYSIS'] = f"سیگنال از {selected_model} با اعتماد بالاتر - مدل دیگر مخالف است"
            return selected_result
        else:
            # عدم قطعیت - HOLD
            return {
                'SYMBOL': symbol,
                'ACTION': 'HOLD',
                'CONFIDENCE': 0,
                'COMBINED_ANALYSIS': True,
                'MODELS_AGREE': False,
                'ANALYSIS': 'تضاد بین مدل‌ها - نیاز به بررسی بیشتر'
            }

class AdvancedForexAnalyzer:
    def __init__(self): 
        self.cache_manager = SmartCacheManager(CACHE_FILE, CACHE_DURATION_HOURS)
        self.technical_analyzer = AdvancedTechnicalAnalyzer()
        self.ai_manager = HybridAIManager(google_api_key, DEEPSEEK_API_KEY)

    async def analyze_pair(self, pair: str) -> Optional[Dict]:
        """تحلیل کامل یک جفت ارز"""
        if self.cache_manager.is_pair_on_cooldown(pair):
            return None
        
        logging.info(f"🔍 شروع تحلیل پیشرفته برای {pair}")
        
        try:
            # دریافت داده‌های بازار
            htf_df = await self.get_market_data_async(pair, HIGH_TIMEFRAME)
            ltf_df = await self.get_market_data_async(pair, LOW_TIMEFRAME)
            
            if htf_df is None or ltf_df is None:
                logging.warning(f"داده‌های بازار برای {pair} دریافت نشد")
                return None
            
            # تحلیل تکنیکال
            htf_df = self.technical_analyzer.calculate_advanced_indicators(htf_df)
            ltf_df = self.technical_analyzer.calculate_advanced_indicators(ltf_df)
            
            if htf_df is None or ltf_df is None:
                logging.warning(f"خطا در محاسبه اندیکاتورها برای {pair}")
                return None
            
            technical_analysis = self.technical_analyzer.generate_technical_analysis(pair, htf_df, ltf_df)
            
            if not technical_analysis:
                logging.warning(f"تحلیل تکنیکال برای {pair} ناموفق بود")
                return None
            
            # تحلیل ترکیبی AI
            ai_analysis = await self.ai_manager.get_combined_analysis(pair, technical_analysis)
            
            if ai_analysis and ai_analysis.get('ACTION') != 'HOLD':
                self.cache_manager.update_cache(pair, ai_analysis)
                return ai_analysis
            else:
                logging.info(f"هیچ سیگنال معاملاتی برای {pair} شناسایی نشد")
                return None
                
        except Exception as e:
            logging.error(f"خطا در تحلیل {pair}: {e}")
            return None

    async def get_market_data_async(self, symbol: str, interval: str, retries: int = 3) -> Optional[pd.DataFrame]:
        """دریافت داده‌های بازار به صورت async"""
        for attempt in range(retries):
            try:
                url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={CANDLES_TO_FETCH}&apikey={TWELVEDATA_API_KEY}'
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=60) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'values' in data and len(data['values']) > 0:
                                df = pd.DataFrame(data['values'])
                                df = df.iloc[::-1].reset_index(drop=True)
                                
                                # تبدیل انواع داده
                                for col in ['open', 'high', 'low', 'close']:
                                    if col in df.columns:
                                        df[col] = pd.to_numeric(df[col], errors='coerce')
                                
                                df['datetime'] = pd.to_datetime(df['datetime'])
                                df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
                                
                                return df
                        else:
                            logging.warning(f"خطای HTTP {response.status} برای {symbol}")
                            
            except Exception as e:
                logging.warning(f"خطا در دریافت داده‌های {symbol} (تلاش {attempt + 1}): {e}")
                await asyncio.sleep(2)
        
        return None

    async def analyze_all_pairs(self, pairs: List[str]) -> List[Dict]:
        """تحلیل همه جفت ارزها به صورت موازی"""
        logging.info(f"🚀 شروع تحلیل موازی برای {len(pairs)} جفت ارز")
        
        # محدود کردن concurrent tasks برای مدیریت rate limits
        semaphore = asyncio.Semaphore(1)  # حداکثر ۳ تحلیل همزمان
        
        async def bounded_analyze(pair):
            async with semaphore:
                result = await self.analyze_pair(pair)
                # یک ثانیه تأخیر بین تحلیل هر جفت‌ارز برای جلوگیری از رسیدن به سقف محدودیت API
                await asyncio.sleep(1)
                return result
        
        tasks = [bounded_analyze(pair) for pair in pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # فیلتر کردن نتایج موفق
        valid_signals = []
        for result in results:
            if isinstance(result, Dict) and result.get('ACTION') != 'HOLD':
                valid_signals.append(result)
            elif isinstance(result, Exception):
                logging.error(f"خطا در تحلیل: {result}")
        
        return valid_signals

async def main():
    # This entire block is now correctly indented
    logging.info("🎯 شروع سیستم تحلیل فارکس پیشرفته (Hybrid AI v2.0)")
    analyzer = AdvancedForexAnalyzer()

    # بررسی جفت ارزهای مشخص شده از طریق آرگومان
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, help="تحلیل جفت ارز مشخص")
    parser.add_argument("--all", action="store_true", help="تحلیل همه جفت ارزها")
    args = parser.parse_args()

    if args.pair:
        pairs_to_analyze = [args.pair]
    elif args.all:
        pairs_to_analyze = CURRENCY_PAIRS_TO_ANALYZE
    else:
        # اگر هیچ آرگومانی داده نشود، همه را تحلیل کن
        pairs_to_analyze = CURRENCY_PAIRS_TO_ANALYZE

    # اجرای تحلیل
    signals = await analyzer.analyze_all_pairs(pairs_to_analyze)

    # ذخیره نتایج
    if signals:
        output_file = "hybrid_ai_signals.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(signals, f, indent=4, ensure_ascii=False)
        
        logging.info(f"✅ تحلیل کامل شد. {len(signals)} سیگنال در {output_file} ذخیره شد")
        
        # نمایش خلاصه نتایج
        for signal in signals:
            logging.info(f"📈 {signal['SYMBOL']}: {signal['ACTION']} (اعتماد: {signal.get('CONFIDENCE', 'N/A')}/10)")
    else:
        logging.info("🔍 هیچ سیگنال معاملاتی‌ای در این اجرا شناسایی نشد")

    logging.info("🏁 پایان اجرای سیستم")

if __name__ == "__main__":
    asyncio.run(main())
