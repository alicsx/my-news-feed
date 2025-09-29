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
import sys

# =================================================================================
# --- تنظیمات پیشرفته برای GitHub Actions ---
# =================================================================================

# کلیدهای API - برای GitHub Actions از Secrets استفاده می‌شود
google_api_key = os.getenv("GOOGLE_API_KEY")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
CLOUDFLARE_AI_API_KEY = os.getenv("CLOUDFLARE_AI_API_KEY")

if not all([google_api_key, TWELVEDATA_API_KEY]):
    raise ValueError("لطفاً کلیدهای API را در GitHub Secrets تنظیم کنید")

# تنظیمات بهینه‌شده برای GitHub Actions
HIGH_TIMEFRAME = "4h"
LOW_TIMEFRAME = "1h"
CANDLES_TO_FETCH = 200  # کاهش برای صرفه‌جویی در API calls
CURRENCY_PAIRS_TO_ANALYZE = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
    "GBP/JPY", "EUR/JPY", "AUD/JPY", "NZD/USD", "USD/CAD"
]

CACHE_FILE = "signal_cache.json"
CACHE_DURATION_HOURS = 2
LOG_FILE = "trading_log.log"

# مدل‌های AI پیشرفته
GEMINI_MODEL = 'gemini-2.5-flash'
IMPROVED_CLOUDFLARE_MODELS = [
    "@cf/meta/llama-3-8b-instruct",  # مدل اصلی
    "@cf/qwen/qwen1.5-14b-chat-awq",  # مدل fallback
    "@cf/meta/llama-2-7b-chat-fp16"   # مدل reserve
]

# راه‌اندازی سیستم لاگ‌گیری بهینه‌شده
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

class AsyncRateLimiter:
    """مدیریت پیشرفته محدودیت نرخ درخواست‌ها"""
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
# --- کلاس تحلیل تکنیکال پیشرفته‌تر ---
# =================================================================================

class EnhancedTechnicalAnalyzer:
    def __init__(self):
        self.indicators_config = {
            'trend': ['ema_8', 'ema_21', 'ema_50', 'ema_200', 'adx_14', 'psar'],
            'momentum': ['rsi_14', 'stoch_14_3_3', 'macd', 'cci_20', 'williams_14'],
            'volatility': ['bb_20_2', 'atr_14', 'kc_20'],
            'volume': ['obv', 'volume_sma_20', 'mfi_14'],
            'ichimoku': True,
            'support_resistance': True,
            'candle_patterns': True,
            'pivot_points': True
        }

    def calculate_enhanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or len(df) < 100:
            return None
            
        try:
            # اندیکاتورهای روند پیشرفته
            df.ta.ema(length=8, append=True)
            df.ta.ema(length=21, append=True)
            df.ta.ema(length=50, append=True)
            df.ta.ema(length=200, append=True)
            df.ta.adx(length=14, append=True)
            df.ta.psar(append=True)
            
            # اندیکاتورهای مومنتوم پیشرفته
            df.ta.rsi(length=14, append=True)
            df.ta.stoch(append=True)
            df.ta.macd(append=True)
            df.ta.cci(length=20, append=True)
            df.ta.willr(length=14, append=True)
            
            # اندیکاتورهای نوسان
            df.ta.bbands(length=20, std=2, append=True)
            df.ta.atr(length=14, append=True)
            df.ta.kc(length=20, append=True)
            
            # حجم پیشرفته
            if 'volume' in df.columns and not df['volume'].isnull().all():
                df.ta.obv(append=True)
                df['volume_sma_20'] = df['volume'].rolling(20).mean()
                df.ta.mfi(length=14, append=True)
            
            # ایچیموکو
            df.ta.ichimoku(append=True)
            
            # سطوح حمایت و مقاومت داینامیک
            df['sup_1'] = df['low'].rolling(20).min().shift(1)
            df['res_1'] = df['high'].rolling(20).max().shift(1)
            df['sup_2'] = df['low'].rolling(50).min().shift(1)
            df['res_2'] = df['high'].rolling(50).max().shift(1)
            
            # پیوت پوینت‌ها
            df = self.calculate_pivot_points(df)
            
            # الگوهای کندل استیک
            if self.indicators_config['candle_patterns']:
                enhanced_patterns = ['doji', 'hammer', 'engulfing', 'harami', 'morningstar', 
                                   'eveningstar', 'piercing', 'darkcloudcover']
                for pattern in enhanced_patterns:
                    try:
                        df.ta.cdl_pattern(name=pattern, append=True)
                    except Exception as e:
                        continue
            
            df.dropna(inplace=True)
            return df
            
        except Exception as e:
            logging.error(f"خطا در محاسبه اندیکاتورهای پیشرفته: {e}")
            return None

    def calculate_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه پیوت پوینت‌های استاندارد"""
        if len(df) < 2:
            return df
            
        try:
            # استفاده از قیمت‌های روز قبل برای پیوت
            prev_high = df['high'].shift(1)
            prev_low = df['low'].shift(1)
            prev_close = df['close'].shift(1)
            
            # محاسبه پیوت اصلی
            df['pivot'] = (prev_high + prev_low + prev_close) / 3
            df['r1'] = 2 * df['pivot'] - prev_low
            df['s1'] = 2 * df['pivot'] - prev_high
            df['r2'] = df['pivot'] + (prev_high - prev_low)
            df['s2'] = df['pivot'] - (prev_high - prev_low)
            df['r3'] = df['pivot'] + 2 * (prev_high - prev_low)
            df['s3'] = df['pivot'] - 2 * (prev_high - prev_low)
            
        except Exception as e:
            logging.warning(f"خطا در محاسبه پیوت پوینت‌ها: {e}")
            
        return df

    def generate_enhanced_analysis(self, symbol: str, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> Dict:
        if htf_df.empty or ltf_df.empty:
            return None
            
        last_htf = htf_df.iloc[-1]
        last_ltf = ltf_df.iloc[-1]
        prev_ltf = ltf_df.iloc[-2] if len(ltf_df) > 1 else last_ltf
        
        # تحلیل روند پیشرفته
        htf_trend = self._analyze_enhanced_trend(last_htf)
        ltf_trend = self._analyze_enhanced_trend(last_ltf)
        
        # تحلیل مومنتوم پیشرفته
        momentum = self._analyze_enhanced_momentum(last_ltf)
        
        # تحلیل سطوح کلیدی پیشرفته
        key_levels = self._analyze_enhanced_key_levels(htf_df, ltf_df, last_ltf['close'])
        
        # تحلیل الگوهای کندل استیک
        candle_analysis = self._analyze_candle_patterns(ltf_df)
        
        # تحلیل قدرت روند
        trend_strength = self._analyze_trend_strength(htf_df, ltf_df)
        
        # سیگنال‌های ترکیبی
        combined_signals = self._generate_combined_signals(htf_trend, ltf_trend, momentum, key_levels)
        
        return {
            'symbol': symbol,
            'htf_trend': htf_trend,
            'ltf_trend': ltf_trend,
            'momentum': momentum,
            'key_levels': key_levels,
            'candle_patterns': candle_analysis,
            'trend_strength': trend_strength,
            'combined_signals': combined_signals,
            'volatility': last_ltf.get('ATRr_14', 0),
            'timestamp': datetime.now(UTC).isoformat()
        }

    def _analyze_enhanced_trend(self, data: pd.Series) -> Dict:
        ema_8 = data.get('EMA_8', 0)
        ema_21 = data.get('EMA_21', 0)
        ema_50 = data.get('EMA_50', 0)
        ema_200 = data.get('EMA_200', 0)
        adx = data.get('ADX_14', 0)
        psar = data.get('PSARl_0.02_0.2', 0)
        
        # تحلیل پیشرفته روند
        ema_alignment = "صعودی قوی" if ema_8 > ema_21 > ema_50 > ema_200 else \
                       "نزولی قوی" if ema_8 < ema_21 < ema_50 < ema_200 else \
                       "صعودی ضعیف" if ema_8 > ema_21 else \
                       "نزولی ضعیف" if ema_8 < ema_21 else "خنثی"
        
        trend_strength = "بسیار قوی" if adx > 40 else "قوی" if adx > 25 else "متوسط" if adx > 20 else "ضعیف"
        
        psar_signal = "صعودی" if psar < data.get('close', 0) else "نزولی"
        
        return {
            'direction': ema_alignment,
            'strength': trend_strength,
            'adx': adx,
            'psar_signal': psar_signal,
            'ema_alignment': f"EMA8: {ema_8:.5f}, EMA21: {ema_21:.5f}, EMA50: {ema_50:.5f}"
        }

    def _analyze_enhanced_momentum(self, data: pd.Series) -> Dict:
        rsi = data.get('RSI_14', 50)
        macd_hist = data.get('MACDh_12_26_9', 0)
        stoch_k = data.get('STOCHk_14_3_3', 50)
        cci = data.get('CCI_20', 0)
        williams = data.get('WILLR_14', -50)
        
        rsi_signal = "اشباع خرید" if rsi > 70 else "اشباع فروش" if rsi < 30 else "خنثی"
        macd_signal = "صعودی قوی" if macd_hist > 0 and macd_hist > data.get('MACDh_12_26_9', 0) else \
                     "صعودی ضعیف" if macd_hist > 0 else \
                     "نزولی قوی" if macd_hist < 0 and macd_hist < data.get('MACDh_12_26_9', 0) else "نزولی ضعیف"
        stoch_signal = "اشباع خرید" if stoch_k > 80 else "اشباع فروش" if stoch_k < 20 else "خنثی"
        cci_signal = "صعودی" if cci > 100 else "نزولی" if cci < -100 else "خنثی"
        williams_signal = "اشباع خرید" if williams > -20 else "اشباع فروش" if williams < -80 else "خنثی"
        
        return {
            'rsi': {'value': rsi, 'signal': rsi_signal},
            'macd': {'signal': macd_signal, 'histogram': macd_hist},
            'stochastic': {'value': stoch_k, 'signal': stoch_signal},
            'cci': {'value': cci, 'signal': cci_signal},
            'williams': {'value': williams, 'signal': williams_signal},
            'overall_momentum': self._calculate_overall_momentum(rsi, macd_hist, stoch_k, cci)
        }

    def _calculate_overall_momentum(self, rsi: float, macd_hist: float, stoch_k: float, cci: float) -> str:
        """محاسبه مومنتوم کلی بر اساس چندین اندیکاتور"""
        score = 0
        if rsi > 50: score += 1
        if macd_hist > 0: score += 1
        if stoch_k > 50: score += 1
        if cci > 0: score += 1
        
        if score >= 3: return "صعودی قوی"
        if score == 2: return "صعودی ضعیف"
        if score <= 1: return "نزولی ضعیف"
        return "نزولی قوی"

    def _analyze_enhanced_key_levels(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame, current_price: float) -> Dict:
        bb_upper = ltf_df.get('BBU_20_2.0', pd.Series([0])).iloc[-1]
        bb_lower = ltf_df.get('BBL_20_2.0', pd.Series([0])).iloc[-1]
        bb_middle = ltf_df.get('BBM_20_2.0', pd.Series([0])).iloc[-1]
        
        kc_upper = ltf_df.get('KCUe_20_2', pd.Series([0])).iloc[-1]
        kc_lower = ltf_df.get('KCLe_20_2', pd.Series([0])).iloc[-1]
        
        support_1 = ltf_df.get('sup_1', pd.Series([0])).iloc[-1]
        resistance_1 = ltf_df.get('res_1', pd.Series([0])).iloc[-1]
        support_2 = ltf_df.get('sup_2', pd.Series([0])).iloc[-1]
        resistance_2 = ltf_df.get('res_2', pd.Series([0])).iloc[-1]
        
        pivot = ltf_df.get('pivot', pd.Series([0])).iloc[-1]
        r1 = ltf_df.get('r1', pd.Series([0])).iloc[-1]
        s1 = ltf_df.get('s1', pd.Series([0])).iloc[-1]
        
        return {
            'dynamic': {
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_middle': bb_middle,
                'kc_upper': kc_upper,
                'kc_lower': kc_lower
            },
            'static': {
                'support_1': support_1,
                'resistance_1': resistance_1,
                'support_2': support_2,
                'resistance_2': resistance_2
            },
            'pivot_points': {
                'pivot': pivot,
                'r1': r1,
                's1': s1
            },
            'current_price_position': self._get_enhanced_price_position(current_price, support_1, resistance_1, bb_lower, bb_upper)
        }

    def _get_enhanced_price_position(self, price: float, support: float, resistance: float, bb_lower: float, bb_upper: float) -> str:
        if resistance == support or resistance <= support:
            return "در محدوده خنثی"
        
        range_size = resistance - support
        position = (price - support) / range_size
        
        # تحلیل موقعیت نسبت به باندهای بولینگر
        bb_position = ""
        if price < bb_lower:
            bb_position = " (زیر باند پایینی)"
        elif price > bb_upper:
            bb_position = " (بالای باند بالایی)"
        
        if position < 0.2:
            return "نزدیک حمایت اصلی" + bb_position
        elif position > 0.8:
            return "نزدیک مقاومت اصلی" + bb_position
        elif position < 0.4:
            return "نزدیک حمایت" + bb_position
        elif position > 0.6:
            return "نزدیک مقاومت" + bb_position
        else:
            return "در میانه رنج" + bb_position

    def _analyze_trend_strength(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> Dict:
        """تحلیل قدرت روند"""
        htf_adx = htf_df.get('ADX_14', pd.Series([0])).iloc[-1]
        ltf_adx = ltf_df.get('ADX_14', pd.Series([0])).iloc[-1]
        
        htf_trend_dir = "صعودی" if htf_df.get('EMA_21', pd.Series([0])).iloc[-1] > htf_df.get('EMA_50', pd.Series([0])).iloc[-1] else "نزولی"
        ltf_trend_dir = "صعودی" if ltf_df.get('EMA_21', pd.Series([0])).iloc[-1] > ltf_df.get('EMA_50', pd.Series([0])).iloc[-1] else "نزولی"
        
        trend_alignment = "همسو" if htf_trend_dir == ltf_trend_dir else "غیرهمسو"
        
        return {
            'htf_strength': "قوی" if htf_adx > 25 else "ضعیف",
            'ltf_strength': "قوی" if ltf_adx > 25 else "ضعیف",
            'trend_alignment': trend_alignment,
            'overall_strength': "بسیار قوی" if htf_adx > 25 and ltf_adx > 25 and trend_alignment == "همسو" else "قوی" if (htf_adx > 25 or ltf_adx > 25) else "ضعیف"
        }

    def _generate_combined_signals(self, htf_trend: Dict, ltf_trend: Dict, momentum: Dict, key_levels: Dict) -> List[str]:
        """تولید سیگنال‌های ترکیبی"""
        signals = []
        
        # سیگنال روند
        if htf_trend['direction'] == ltf_trend['direction'] and "صعودی" in htf_trend['direction']:
            signals.append("همسویی روند صعودی")
        elif htf_trend['direction'] == ltf_trend['direction'] and "نزولی" in htf_trend['direction']:
            signals.append("همسویی روند نزولی")
        
        # سیگنال مومنتوم
        if momentum['overall_momentum'] == "صعودی قوی":
            signals.append("مومنتوم صعودی قوی")
        elif momentum['overall_momentum'] == "نزولی قوی":
            signals.append("مومنتوم نزولی قوی")
        
        # سیگنال موقعیت قیمت
        price_pos = key_levels['current_price_position']
        if "نزدیک حمایت" in price_pos and "صعودی" in momentum['overall_momentum']:
            signals.append("موقعیت خرید در حمایت")
        elif "نزدیک مقاومت" in price_pos and "نزولی" in momentum['overall_momentum']:
            signals.append("موقعیت فروش در مقاومت")
        
        return signals

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
            'recent_patterns': patterns[-3:] if patterns else [],
            'pattern_strength': "قوی" if len(patterns) > 0 and any("صعودی قوی" in p or "نزولی قوی" in p for p in patterns) else "متوسط" if patterns else "ضعیف"
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
        
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        
        if body_ratio < 0.1 and upper_shadow > 0 and lower_shadow > 0:
            candle_type = "دوجی"
        elif body_ratio < 0.3 and lower_shadow > 2 * body_size:
            candle_type = "چکش"
        elif body_ratio < 0.3 and upper_shadow > 2 * body_size:
            candle_type = "ستاره ثاقب"
        elif body_ratio > 0.7:
            candle_type = "ماروبوزو"
        else:
            candle_type = "عادی"
            
        direction = "صعودی" if close > open_price else "نزولی"
        
        shadow_ratio = (upper_shadow + lower_shadow) / total_range if total_range > 0 else 0
        
        return {
            'type': candle_type,
            'direction': direction,
            'body_ratio': round(body_ratio, 2),
            'shadow_ratio': round(shadow_ratio, 2),
            'strength': "قوی" if body_ratio > 0.6 else "متوسط" if body_ratio > 0.3 else "ضعیف"
        }

# =================================================================================
# --- کلاس مدیریت AI ترکیبی پیشرفته ---
# =================================================================================

class AdvancedHybridAIManager:
    def __init__(self, gemini_api_key: str, cloudflare_api_key: str):
        self.gemini_api_key = gemini_api_key
        self.cloudflare_api_key = cloudflare_api_key
        self.gemini_model = GEMINI_MODEL
        
        # تنظیمات پیشرفته Cloudflare
        self.cloudflare_account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "your_account_id")
        self.cloudflare_model_name = IMPROVED_CLOUDFLARE_MODELS[0]
        self.fallback_models = IMPROVED_CLOUDFLARE_MODELS[1:]
        self.current_model_index = 0
        
        genai.configure(api_key=gemini_api_key)
        
        # کش برای بهبود عملکرد
        self.analysis_cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    async def get_enhanced_analysis(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        """دریافت تحلیل پیشرفته ترکیبی"""
        cache_key = f"{symbol}_{hash(str(technical_analysis))}"
        current_time = time.time()
        
        # بررسی کش
        if cache_key in self.analysis_cache:
            cached_data, timestamp = self.analysis_cache[cache_key]
            if current_time - timestamp < self.cache_timeout:
                return cached_data
        
        tasks = [
            self._get_enhanced_gemini_analysis(symbol, technical_analysis),
            self._get_enhanced_cloudflare_analysis(symbol, technical_analysis)
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            gemini_result, cloudflare_result = results
            
            if isinstance(gemini_result, Exception):
                logging.error(f"خطا در Gemini برای {symbol}: {gemini_result}")
                gemini_result = None
            if isinstance(cloudflare_result, Exception):
                logging.error(f"خطا در Cloudflare برای {symbol}: {cloudflare_result}")
                cloudflare_result = None
            
            combined_result = self._combine_enhanced_analyses(symbol, gemini_result, cloudflare_result, technical_analysis)
            
            # ذخیره در کش
            if combined_result:
                self.analysis_cache[cache_key] = (combined_result, current_time)
            
            return combined_result
            
        except Exception as e:
            logging.error(f"خطا در تحلیل ترکیبی برای {symbol}: {e}")
            return None
    
    async def _get_enhanced_gemini_analysis(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        """تحلیل پیشرفته با Gemini"""
        try:
            prompt = self._create_enhanced_prompt(symbol, technical_analysis, "Gemini")
            model = genai.GenerativeModel(self.gemini_model)
            
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                request_options={'timeout': 60}
            )
            
            return self._parse_enhanced_ai_response(response.text, symbol, "Gemini")
            
        except Exception as e:
            logging.warning(f"خطا در تحلیل Gemini برای {symbol}: {e}")
            return None
    
    async def _get_enhanced_cloudflare_analysis(self, symbol: str, technical_analysis: Dict, retry_count: int = 0) -> Optional[Dict]:
        """تحلیل پیشرفته با Cloudflare AI با قابلیت retry"""
        if not self.cloudflare_api_key or self.cloudflare_account_id == "your_account_id":
            logging.warning("کلید یا شناسه حساب Cloudflare API تنظیم نشده است")
            return None
            
        try:
            prompt = self._create_enhanced_prompt(symbol, technical_analysis, "Cloudflare")
            
            headers = {
                "Authorization": f"Bearer {self.cloudflare_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are an expert forex trading analyst with 20 years experience. Provide precise analysis in valid JSON format only."
                    },
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "max_tokens": 1000
            }
            
            current_model = self.fallback_models[self.current_model_index] if retry_count > 0 else self.cloudflare_model_name
            cloudflare_url = f"https://api.cloudflare.com/client/v4/accounts/{self.cloudflare_account_id}/ai/run/{current_model}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(cloudflare_url, headers=headers, json=payload, timeout=60) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = self._extract_cloudflare_response(data)
                        if content:
                            return self._parse_enhanced_ai_response(content, symbol, f"Cloudflare ({current_model})")
                        else:
                            raise Exception("پاسخ خالی از Cloudflare")
                    else:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")
                        
        except Exception as e:
            logging.warning(f"خطا در تحلیل Cloudflare برای {symbol} (تلاش {retry_count + 1}): {e}")
            
            # تلاش مجدد با مدل fallback
            if retry_count < len(self.fallback_models) - 1:
                self.current_model_index = (self.current_model_index + 1) % len(self.fallback_models)
                await asyncio.sleep(1)
                return await self._get_enhanced_cloudflare_analysis(symbol, technical_analysis, retry_count + 1)
            else:
                return None

    def _extract_cloudflare_response(self, data: Dict) -> Optional[str]:
        """استخراج پاسخ از داده‌های Cloudflare"""
        if "result" in data and "response" in data["result"]:
            return data["result"]["response"]
        elif "response" in data:
            return data["response"]
        else:
            logging.warning(f"فرمت پاسخ Cloudflare نامعتبر است: {data}")
            return None

    def _create_enhanced_prompt(self, symbol: str, technical_analysis: Dict, ai_name: str) -> str:
        base_currency, quote_currency = symbol.split('/')
        
        return f"""
به عنوان یک تحلیلگر حرفه‌ای بازار فارکس با ۲۰ سال تجربه، تحلیل تکنیکال زیر را برای جفت ارز {symbol} بررسی کنید و فقط و فقط یک آبجکت JSON معتبر برگردانید.

📊 **وضعیت تکنیکال پیشرفته {symbol}:**

🎯 **روندها:**
- روند بلندمدت (HTF): {technical_analysis['htf_trend']['direction']} - قدرت: {technical_analysis['htf_trend']['strength']}
- روند کوتاه‌مدت (LTF): {technical_analysis['ltf_trend']['direction']} - قدرت: {technical_analysis['ltf_trend']['strength']}
- همسویی روندها: {technical_analysis['trend_strength']['trend_alignment']}
- قدرت کلی: {technical_analysis['trend_strength']['overall_strength']}

⚡ **مومنتوم:**
- RSI: {technical_analysis['momentum']['rsi']['value']:.1f} ({technical_analysis['momentum']['rsi']['signal']})
- MACD: {technical_analysis['momentum']['macd']['signal']}
- Stochastic: {technical_analysis['momentum']['stochastic']['value']:.1f} ({technical_analysis['momentum']['stochastic']['signal']})
- مومنتوم کلی: {technical_analysis['momentum']['overall_momentum']}

📈 **سطوح کلیدی:**
- موقعیت قیمت: {technical_analysis['key_levels']['current_price_position']}
- مقاومت ۱: {technical_analysis['key_levels']['static']['resistance_1']:.5f}
- حمایت ۱: {technical_analysis['key_levels']['static']['support_1']:.5f}
- مقاومت ۲: {technical_analysis['key_levels']['static']['resistance_2']:.5f}
- حمایت ۲: {technical_analysis['key_levels']['static']['support_2']:.5f}
- پیوت: {technical_analysis['key_levels']['pivot_points']['pivot']:.5f}

🕯️ **الگوهای کندل‌استیک:**
- کندل فعلی: {technical_analysis['candle_patterns']['current_candle']['type']} ({technical_analysis['candle_patterns']['current_candle']['direction']})
- الگوهای شناسایی شده: {', '.join(technical_analysis['candle_patterns']['patterns'][-2:])}

💡 **سیگنال‌های ترکیبی:**
{chr(10).join(['- ' + signal for signal in technical_analysis['combined_signals']])}

**لطفاً پاسخ را فقط در قالب JSON زیر ارائه دهید (بدون هیچ متن اضافی):**

{{
  "SYMBOL": "{symbol}",
  "ACTION": "BUY/SELL/HOLD",
  "CONFIDENCE": 1-10,
  "ENTRY_ZONE": "محدوده عددی (مثال: 1.12340-1.12360)",
  "STOP_LOSS": "عدد اعشاری دقیق (مثال: 1.12050)", 
  "TAKE_PROFIT_1": "عدد اعشاری دقیق (مثال: 1.12800)",
  "TAKE_PROFIT_2": "عدد اعشاری دقیق (مثال: 1.13000)",
  "RISK_REWARD_RATIO": "نسبت عددی (مثال: 1.8)",
  "ANALYSIS": "تحلیل مختصر و تخصصی فارسی",
  "EXPIRATION_H": "عدد صحیح (مثال: 6)",
  "PRIORITY": "HIGH/MEDIUM/LOW"
}}
"""

    def _parse_enhanced_ai_response(self, response: str, symbol: str, ai_name: str) -> Optional[Dict]:
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
                
                if not self._validate_enhanced_signal_data(signal_data, symbol):
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

    def _validate_enhanced_signal_data(self, signal_data: Dict, symbol: str) -> bool:
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

    def _combine_enhanced_analyses(self, symbol: str, gemini_result: Dict, cloudflare_result: Dict, technical_analysis: Dict) -> Optional[Dict]:
        valid_results = []
        
        if gemini_result and self._validate_enhanced_signal_data(gemini_result, symbol):
            valid_results.append(('Gemini', gemini_result))
        
        if cloudflare_result and self._validate_enhanced_signal_data(cloudflare_result, symbol):
            valid_results.append(('Cloudflare', cloudflare_result))
        
        if not valid_results:
            logging.info(f"هیچ سیگنال معتبری از مدل‌های AI برای {symbol} دریافت نشد")
            return self._create_hold_signal(symbol, technical_analysis)
        
        if len(valid_results) == 1:
            model_name, result = valid_results[0]
            return self._enhance_single_model_result(result, model_name, technical_analysis)
        
        gemini_data = valid_results[0][1] if valid_results[0][0] == 'Gemini' else valid_results[1][1]
        cloudflare_data = valid_results[0][1] if valid_results[0][0] == 'Cloudflare' else valid_results[1][1]
        
        gemini_action = gemini_data.get('ACTION', 'HOLD').upper()
        cloudflare_action = cloudflare_data.get('ACTION', 'HOLD').upper()
        
        if gemini_action == cloudflare_action:
            return self._create_consensus_signal(symbol, gemini_data, cloudflare_data, technical_analysis)
        else:
            return self._resolve_enhanced_conflict(symbol, gemini_data, cloudflare_data, technical_analysis)

    def _create_hold_signal(self, symbol: str, technical_analysis: Dict) -> Dict:
        return {
            'SYMBOL': symbol,
            'ACTION': 'HOLD',
            'CONFIDENCE': 0,
            'CONSENSUS': False,
            'ANALYSIS': 'عدم وجود سیگنال معتبر از مدل‌های AI',
            'TIMESTAMP': datetime.now(UTC).isoformat(),
            'TECHNICAL_CONTEXT': technical_analysis.get('combined_signals', [])
        }

    def _enhance_single_model_result(self, result: Dict, model_name: str, technical_analysis: Dict) -> Dict:
        result['CONSENSUS'] = False
        result['MODEL_SOURCE'] = f"Single: {model_name}"
        
        # کاهش اعتماد برای سیگنال‌های تک مدلی
        original_confidence = float(result.get('CONFIDENCE', 5))
        result['CONFIDENCE'] = max(1, original_confidence - 2)
        
        # اضافه کردن اطلاعات تکنیکال
        if 'TECHNICAL_CONTEXT' not in result:
            result['TECHNICAL_CONTEXT'] = technical_analysis.get('combined_signals', [])
        
        return result

    def _create_consensus_signal(self, symbol: str, gemini_data: Dict, cloudflare_data: Dict, technical_analysis: Dict) -> Dict:
        averaged_signal = self._average_enhanced_signals(symbol, gemini_data, cloudflare_data)
        averaged_signal['CONSENSUS'] = True
        averaged_signal['MODELS_AGREE'] = True
        averaged_signal['MODEL_SOURCE'] = "Gemini + Cloudflare Consensus"
        averaged_signal['PRIORITY'] = self._calculate_priority(averaged_signal, technical_analysis)
        
        # افزایش اعتماد برای سیگنال‌های با توافق
        original_confidence = float(averaged_signal.get('CONFIDENCE', 5))
        averaged_signal['CONFIDENCE'] = min(10, original_confidence + 1)
        
        averaged_signal['FINAL_ANALYSIS'] = f"توافق کامل بین مدل‌ها - سیگنال {gemini_data['ACTION']} با اعتماد بالا"
        averaged_signal['TECHNICAL_CONTEXT'] = technical_analysis.get('combined_signals', [])
        
        return averaged_signal

    def _resolve_enhanced_conflict(self, symbol: str, gemini_data: Dict, cloudflare_data: Dict, technical_analysis: Dict) -> Dict:
        gemini_conf = float(gemini_data.get('CONFIDENCE', 5))
        cloudflare_conf = float(cloudflare_data.get('CONFIDENCE', 5))
        
        # انتخاب بر اساس اعتماد و همسویی با تحلیل تکنیکال
        tech_signals = technical_analysis.get('combined_signals', [])
        gemini_score = self._calculate_model_score(gemini_data, gemini_conf, tech_signals)
        cloudflare_score = self._calculate_model_score(cloudflare_data, cloudflare_conf, tech_signals)
        
        if gemini_score >= cloudflare_score:
            selected = gemini_data
            selected_model = 'Gemini'
        else:
            selected = cloudflare_data
            selected_model = 'Cloudflare'
        
        selected['CONSENSUS'] = False
        selected['MODELS_AGREE'] = False
        selected['MODEL_SOURCE'] = f"Conflict Resolution: {selected_model}"
        selected['CONFIDENCE'] = max(1, float(selected.get('CONFIDENCE', 5)) - 1)
        selected['PRIORITY'] = 'LOW'
        selected['ANALYSIS'] = f"سیگنال از {selected_model} - تضاد با مدل دیگر - نیاز به تأیید"
        selected['TECHNICAL_CONTEXT'] = tech_signals
        
        return selected

    def _calculate_model_score(self, signal_data: Dict, confidence: float, tech_signals: List[str]) -> float:
        score = confidence
        
        # افزایش امتیاز بر اساس همسویی با سیگنال‌های تکنیکال
        action = signal_data.get('ACTION', '').upper()
        if action == 'BUY' and any('صعودی' in signal or 'خرید' in signal for signal in tech_signals):
            score += 2
        elif action == 'SELL' and any('نزولی' in signal or 'فروش' in signal for signal in tech_signals):
            score += 2
        
        return score

    def _calculate_priority(self, signal: Dict, technical_analysis: Dict) -> str:
        confidence = float(signal.get('CONFIDENCE', 5))
        trend_strength = technical_analysis.get('trend_strength', {}).get('overall_strength', 'ضعیف')
        
        if confidence >= 8 and trend_strength in ['بسیار قوی', 'قوی']:
            return 'HIGH'
        elif confidence >= 6:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _average_enhanced_signals(self, symbol: str, gemini_data: Dict, cloudflare_data: Dict) -> Dict:
        averaged = {'SYMBOL': symbol}
        
        averaged['ACTION'] = gemini_data['ACTION']
        
        gemini_conf = float(gemini_data.get('CONFIDENCE', 5))
        cloudflare_conf = float(cloudflare_data.get('CONFIDENCE', 5))
        averaged['CONFIDENCE'] = round((gemini_conf + cloudflare_conf) / 2, 1)
        
        numeric_fields = ['ENTRY_ZONE', 'STOP_LOSS', 'TAKE_PROFIT_1', 'TAKE_PROFIT_2', 'EXPIRATION_H']
        
        for field in numeric_fields:
            gemini_val = self._extract_numeric_value(gemini_data.get(field, '0'))
            cloudflare_val = self._extract_numeric_value(cloudflare_data.get(field, '0'))
            
            if gemini_val is not None and cloudflare_val is not None:
                avg_val = (gemini_val + cloudflare_val) / 2
                if field == 'EXPIRATION_H':
                    averaged[field] = int(round(avg_val))
                else:
                    averaged[field] = round(avg_val, 5)
            elif gemini_val is not None:
                averaged[field] = gemini_val
            elif cloudflare_val is not None:
                averaged[field] = cloudflare_val
        
        averaged['RISK_REWARD_RATIO'] = self._calculate_risk_reward(
            averaged.get('STOP_LOSS', 0), 
            averaged.get('TAKE_PROFIT_1', 0),
            gemini_data.get('close', 0)
        )
        
        averaged['GEMINI_ANALYSIS'] = gemini_data.get('ANALYSIS', '')
        averaged['CLOUDFLARE_ANALYSIS'] = cloudflare_data.get('ANALYSIS', '')
        averaged['ANALYSIS'] = f"ترکیب تحلیل‌ها: {gemini_data.get('ANALYSIS', '')}"
        
        return averaged

    def _extract_numeric_value(self, value: str) -> Optional[float]:
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

    def _calculate_risk_reward(self, stop_loss: float, take_profit: float, entry: float) -> float:
        if not all([stop_loss, take_profit, entry]) or stop_loss == take_profit:
            return 1.0
        
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        
        if risk == 0:
            return 1.0
        
        return round(reward / risk, 2)

# =================================================================================
# --- کلاس مدیریت سیگنال‌های با توافق ---
# =================================================================================

class ConsensusSignalManager:
    """مدیریت هوشمند سیگنال‌های با توافق و بدون توافق"""
    
    def __init__(self):
        self.consensus_file = "consensus_signals.json"
        self.non_consensus_file = "non_consensus_signals.json"
        self.consensus_threshold = 7  # آستانه اعتماد برای سیگنال‌های با توافق
        
    def categorize_signals(self, signals: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """دسته‌بندی سیگنال‌ها بر اساس توافق"""
        consensus_signals = []
        non_consensus_signals = []
        
        for signal in signals:
            if self._is_consensus_signal(signal):
                consensus_signals.append(signal)
            else:
                non_consensus_signals.append(signal)
        
        return consensus_signals, non_consensus_signals
    
    def _is_consensus_signal(self, signal: Dict) -> bool:
        """بررسی آیا سیگنال دارای توافق است"""
        # شرط 1: مدل‌ها با هم توافق دارند
        models_agree = signal.get('MODELS_AGREE', False)
        consensus_flag = signal.get('CONSENSUS', False)
        
        # شرط 2: سطح اعتماد بالا
        confidence = float(signal.get('CONFIDENCE', 0))
        high_confidence = confidence >= self.consensus_threshold
        
        # شرط 3: سیگنال از نوع HOLD نیست
        not_hold = signal.get('ACTION', 'HOLD') != 'HOLD'
        
        # شرط 4: اولویت بالا
        high_priority = signal.get('PRIORITY', 'LOW') in ['HIGH', 'MEDIUM']
        
        return (models_agree or consensus_flag) and high_confidence and not_hold and high_priority
    
    def save_categorized_signals(self, consensus_signals: List[Dict], non_consensus_signals: List[Dict]):
        """ذخیره سیگنال‌ها در فایل‌های جداگانه"""
        # ذخیره سیگنال‌های با توافق
        if consensus_signals:
            self._save_to_file(consensus_signals, self.consensus_file)
            logging.info(f"✅ {len(consensus_signals)} سیگنال با توافق در {self.consensus_file} ذخیره شد")
        
        # ذخیره سیگنال‌های بدون توافق
        if non_consensus_signals:
            self._save_to_file(non_consensus_signals, self.non_consensus_file)
            logging.info(f"📊 {len(non_consensus_signals)} سیگنال بدون توافق در {self.non_consensus_file} ذخیره شد")
            
    def _save_to_file(self, signals: List[Dict], filename: str):
        """ذخیره سیگنال‌ها در فایل"""
        try:
            # مرتب‌سازی بر اساس اعتماد
            sorted_signals = sorted(signals, key=lambda x: float(x.get('CONFIDENCE', 0)), reverse=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(sorted_signals, f, indent=4, ensure_ascii=False, default=str)
        except Exception as e:
            logging.error(f"خطا در ذخیره‌سازی {filename}: {e}")

# =================================================================================
# --- کلاس اصلی تحلیلگر فارکس پیشرفته ---
# =================================================================================

class AdvancedForexAnalyzer:
    def __init__(self):
        self.api_rate_limiter = AsyncRateLimiter(rate_limit=6, period=60)  # کاهش برای GitHub Actions
        self.cache_manager = SmartCacheManager(CACHE_FILE, CACHE_DURATION_HOURS)
        self.technical_analyzer = EnhancedTechnicalAnalyzer()
        self.ai_manager = AdvancedHybridAIManager(google_api_key, CLOUDFLARE_AI_API_KEY)
        self.signal_manager = ConsensusSignalManager()

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
            
            # تحلیل تکنیکال پیشرفته
            htf_df_processed = self.technical_analyzer.calculate_enhanced_indicators(htf_df)
            ltf_df_processed = self.technical_analyzer.calculate_enhanced_indicators(ltf_df)
            
            if htf_df_processed is None or ltf_df_processed is None:
                logging.warning(f"خطا در محاسبه اندیکاتورها برای {pair}")
                return None
            
            technical_analysis = self.technical_analyzer.generate_enhanced_analysis(pair, htf_df_processed, ltf_df_processed)
            
            if not technical_analysis:
                logging.warning(f"تحلیل تکنیکال برای {pair} ناموفق بود")
                return None
            
            # تحلیل ترکیبی AI
            ai_analysis = await self.ai_manager.get_enhanced_analysis(pair, technical_analysis)
            
            if ai_analysis and ai_analysis.get('ACTION') != 'HOLD':
                self.cache_manager.update_cache(pair, ai_analysis)
                logging.info(f"✅ سیگنال معاملاتی برای {pair}: {ai_analysis['ACTION']} (اعتماد: {ai_analysis.get('CONFIDENCE', 0)})")
                return ai_analysis
            else:
                logging.info(f"🔍 هیچ سیگنال معاملاتی برای {pair} شناسایی نشد")
                return None
                
        except Exception as e:
            logging.error(f"خطا در تحلیل {pair}: {e}")
            return None

    async def get_market_data_async(self, symbol: str, interval: str, retries: int = 2) -> Optional[pd.DataFrame]:
        """دریافت داده‌های بازار به صورت آسنکرون - بهینه‌شده برای GitHub Actions"""
        for attempt in range(retries):
            try:
                async with self.api_rate_limiter:
                    url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={CANDLES_TO_FETCH}&apikey={TWELVEDATA_API_KEY}'
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, timeout=30) as response:  # کاهش timeout
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
                                    
                                    if len(df) > 50:  # حداقل داده لازم
                                        return df
                                    else:
                                        logging.warning(f"داده‌های {symbol} ناکافی است")
                                        return None
                                else:
                                    logging.warning(f"داده‌های {symbol} خالی است")
                                    return None
                            else:
                                logging.warning(f"خطای HTTP {response.status} برای {symbol}")
                                if response.status == 429:
                                    await asyncio.sleep(15)  # افزایش زمان انتظار برای rate limit
                                
            except Exception as e:
                logging.warning(f"خطا در دریافت داده‌های {symbol} (تلاش {attempt + 1}): {e}")
                await asyncio.sleep(3)
        
        logging.error(f"عدم موفقیت در دریافت داده‌های {symbol} پس از {retries} تلاش")
        return None

    async def analyze_all_pairs(self, pairs: List[str]) -> List[Dict]:
        """تحلیل همه جفت ارزها به صورت موازی - بهینه‌شده برای GitHub Actions"""
        logging.info(f"🚀 شروع تحلیل موازی برای {len(pairs)} جفت ارز")
        
        # محدود کردن concurrent tasks برای GitHub Actions
        semaphore = asyncio.Semaphore(2)  # کاهش برای جلوگیری از overload
        
        async def bounded_analyze(pair):
            async with semaphore:
                result = await self.analyze_pair(pair)
                await asyncio.sleep(2)  # افزایش تأخیر برای احترام به rate limits
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
        
        logging.info(f"📊 تحلیل کامل شد. {len(valid_signals)} سیگنال معتبر شناسایی شد")
        return valid_signals

    def generate_comprehensive_report(self, signals: List[Dict]):
        """تولید گزارش جامع از تحلیل‌ها"""
        consensus_signals, non_consensus_signals = self.signal_manager.categorize_signals(signals)
        
        # ذخیره سیگنال‌های دسته‌بندی شده
        self.signal_manager.save_categorized_signals(consensus_signals, non_consensus_signals)
        
        # ایجاد گزارش خلاصه
        report = {
            'timestamp': datetime.now(UTC).isoformat(),
            'total_analyzed_pairs': len(CURRENCY_PAIRS_TO_ANALYZE),
            'total_signals': len(signals),
            'consensus_signals': len(consensus_signals),
            'non_consensus_signals': len(non_consensus_signals),
            'consensus_details': [],
            'market_summary': self._generate_market_summary(signals),
            'performance_metrics': self._calculate_performance_metrics(signals)
        }
        
        # اضافه کردن جزئیات سیگنال‌های با توافق
        for signal in consensus_signals:
            report['consensus_details'].append({
                'symbol': signal.get('SYMBOL'),
                'action': signal.get('ACTION'),
                'confidence': signal.get('CONFIDENCE'),
                'priority': signal.get('PRIORITY', 'MEDIUM'),
                'entry_zone': signal.get('ENTRY_ZONE'),
                'stop_loss': signal.get('STOP_LOSS'),
                'take_profit_1': signal.get('TAKE_PROFIT_1'),
                'take_profit_2': signal.get('TAKE_PROFIT_2'),
                'risk_reward': signal.get('RISK_REWARD_RATIO'),
                'expiration_hours': signal.get('EXPIRATION_H')
            })
        
        # ذخیره گزارش
        report_file = "comprehensive_analysis_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        
        logging.info(f"📋 گزارش جامع در {report_file} ذخیره شد")
        return report
    
    def _generate_market_summary(self, signals: List[Dict]) -> Dict:
        """تولید خلاصه وضعیت بازار"""
        buy_signals = [s for s in signals if s.get('ACTION') == 'BUY']
        sell_signals = [s for s in signals if s.get('ACTION') == 'SELL']
        
        avg_confidence = sum(float(s.get('CONFIDENCE', 0)) for s in signals) / len(signals) if signals else 0
        
        high_confidence_signals = [s for s in signals if float(s.get('CONFIDENCE', 0)) >= 8]
        medium_confidence_signals = [s for s in signals if 5 <= float(s.get('CONFIDENCE', 0)) < 8]
        
        return {
            'total_buy_signals': len(buy_signals),
            'total_sell_signals': len(sell_signals),
            'average_confidence': round(avg_confidence, 2),
            'high_confidence_count': len(high_confidence_signals),
            'medium_confidence_count': len(medium_confidence_signals),
            'market_bias': 'صعودی' if len(buy_signals) > len(sell_signals) else 'نزولی' if len(sell_signals) > len(buy_signals) else 'خنثی',
            'signal_quality': 'عالی' if len(high_confidence_signals) >= 3 else 'خوب' if len(signals) >= 2 else 'ضعیف'
        }
    
    def _calculate_performance_metrics(self, signals: List[Dict]) -> Dict:
        """محاسبه معیارهای عملکرد"""
        if not signals:
            return {'total_score': 0, 'quality_rating': 'ضعیف'}
        
        confidence_sum = sum(float(s.get('CONFIDENCE', 0)) for s in signals)
        avg_confidence = confidence_sum / len(signals)
        
        consensus_count = sum(1 for s in signals if s.get('CONSENSUS', False))
        consensus_ratio = consensus_count / len(signals) if signals else 0
        
        # محاسبه امتیاز کلی
        total_score = (avg_confidence * 0.6) + (consensus_ratio * 40 * 0.4)  # مقیاس 0-100
        
        if total_score >= 80:
            quality_rating = 'عالی'
        elif total_score >= 60:
            quality_rating = 'خوب'
        elif total_score >= 40:
            quality_rating = 'متوسط'
        else:
            quality_rating = 'ضعیف'
        
        return {
            'total_score': round(total_score, 1),
            'quality_rating': quality_rating,
            'average_confidence': round(avg_confidence, 2),
            'consensus_ratio': round(consensus_ratio, 2),
            'signal_diversity': len(set(s.get('SYMBOL') for s in signals))
        }

# =================================================================================
# --- تابع اصلی بهبود یافته برای GitHub Actions ---
# =================================================================================

async def github_actions_main():
    """تابع اصلی بهینه‌شده برای GitHub Actions"""
    logging.info("🎯 شروع سیستم تحلیل فارکس پیشرفته (GitHub Actions Optimized v2.1)")
    
    # بررسی آرگومان‌های خط فرمان
    import argparse
    parser = argparse.ArgumentParser(description='سیستم تحلیل فارکس با AI ترکیبی - GitHub Actions')
    parser.add_argument("--pair", type=str, help="تحلیل جفت ارز مشخص (مثال: EUR/USD)")
    parser.add_argument("--all", action="store_true", help="تحلیل همه جفت ارزهای پیش‌فرض")
    parser.add_argument("--pairs", type=str, help="تحلیل جفت ارزهای مشخص شده (جدا شده با کاما)")
    parser.add_argument("--consensus-only", action="store_true", help="فقط نمایش سیگنال‌های با توافق")
    
    args = parser.parse_args()

    # تعیین جفت ارزها برای تحلیل
    if args.pair:
        pairs_to_analyze = [args.pair]
    elif args.pairs:
        pairs_to_analyze = [p.strip() for p in args.pairs.split(',')]
    elif args.all:
        pairs_to_analyze = CURRENCY_PAIRS_TO_ANALYZE
    else:
        # تحلیل جفت ارزهای اصلی به صورت پیش‌فرض
        pairs_to_analyze = CURRENCY_PAIRS_TO_ANALYZE[:6]
        logging.info(f"استفاده از 6 جفت ارز اصلی به صورت پیش‌فرض")

    logging.info(f"🔍 جفت ارزهای مورد تحلیل: {', '.join(pairs_to_analyze)}")
    logging.info(f"🤖 مدل‌های AI فعال: Gemini {GEMINI_MODEL} + Cloudflare {IMPROVED_CLOUDFLARE_MODELS[0]}")
    
    # ایجاد تحلیلگر پیشرفته
    analyzer = AdvancedForexAnalyzer()
    
    # زمان‌سنج برای مانیتورینگ عملکرد
    start_time = time.time()
    signals = await analyzer.analyze_all_pairs(pairs_to_analyze)
    analysis_time = time.time() - start_time
    
    # تولید گزارش جامع
    if signals:
        report = analyzer.generate_comprehensive_report(signals)
        
        # نمایش نتایج بهینه‌شده
        consensus_signals = [s for s in signals if s.get('CONSENSUS', False)]
        
        logging.info("=" * 60)
        logging.info("📊 نتایج نهایی تحلیل:")
        logging.info("=" * 60)
        logging.info(f"   ⏱️  زمان تحلیل: {analysis_time:.1f} ثانیه")
        logging.info(f"   📈 کل سیگنال‌ها: {len(signals)}")
        logging.info(f"   ✅ سیگنال‌های با توافق: {len(consensus_signals)}")
        logging.info(f"   🎯 کیفیت کلی: {report['performance_metrics']['quality_rating']}")
        logging.info(f"   📊 تمایل بازار: {report['market_summary']['market_bias']}")
        
        # نمایش سیگنال‌های با توافق
        if consensus_signals:
            logging.info("🎯 سیگنال‌های با توافق بالا (اولویت معامله):")
            for signal in consensus_signals:
                action_icon = "🟢" if signal['ACTION'] == 'BUY' else "🔴"
                priority_icon = "🔥" if signal.get('PRIORITY') == 'HIGH' else "⚡" if signal.get('PRIORITY') == 'MEDIUM' else "💡"
                logging.info(f"   {action_icon} {priority_icon} {signal['SYMBOL']}: {signal['ACTION']} (اعتماد: {signal['CONFIDENCE']}/10)")
                
        # نمایش سایر سیگنال‌ها
        other_signals = [s for s in signals if not s.get('CONSENSUS', False)]
        if other_signals and not args.consensus_only:
            logging.info("📋 سایر سیگنال‌ها (نیاز به تأیید بیشتر):")
            for signal in other_signals[:3]:  # نمایش فقط ۳ تا برای خلاصه‌سازی
                action_icon = "🟢" if signal['ACTION'] == 'BUY' else "🔴"
                logging.info(f"   {action_icon} {signal['SYMBOL']}: {signal['ACTION']} (اعتماد: {signal['CONFIDENCE']}/10)")
            
        if args.consensus_only and not consensus_signals:
            logging.info("🔍 هیچ سیگنال با توافقی در این اجرا شناسایی نشد")
            
    else:
        logging.info("🔍 هیچ سیگنال معاملاتی‌ای در این اجرا شناسایی نشد")
        # ایجاد گزارش خالی برای consistency
        empty_report = {
            'timestamp': datetime.now(UTC).isoformat(),
            'total_analyzed_pairs': len(pairs_to_analyze),
            'total_signals': 0,
            'consensus_signals': 0,
            'non_consensus_signals': 0,
            'consensus_details': [],
            'market_summary': {'market_bias': 'خنثی', 'signal_quality': 'ضعیف'},
            'performance_metrics': {'total_score': 0, 'quality_rating': 'ضعیف'}
        }
        with open("comprehensive_analysis_report.json", 'w', encoding='utf-8') as f:
            json.dump(empty_report, f, indent=4, ensure_ascii=False)

    logging.info("🏁 پایان اجرای سیستم - آماده برای GitHub Actions")

if __name__ == "__main__":
    # اجرای سیستم
    asyncio.run(github_actions_main())
