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
    "@cf/meta/llama-2-7b-chat-fp16",
    "@cf/deepseek-ai/deepseek-math-7b-instruct"
]
GROQ_MODEL = "mixtral-8x7b-32768"  # مدل سریع و قوی Groq

# محدودیت‌های API برای مدیریت مصرف
API_RATE_LIMITS = {
    'google': {'daily_limit': 1500, 'requests_per_minute': 60},
    'cloudflare': {'daily_limit': 1000, 'requests_per_minute': 10},
    'groq': {'daily_limit': 10000, 'requests_per_minute': 30},
    'twelvedata': {'daily_limit': 800, 'requests_per_minute': 8}
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

class APIRateManager:
    """مدیریت هوشمند محدودیت‌های API"""
    def __init__(self):
        self.usage = {
            'google': {'daily': 0, 'minute': 0, 'last_reset_minute': time.time()},
            'cloudflare': {'daily': 0, 'minute': 0, 'last_reset_minute': time.time()},
            'groq': {'daily': 0, 'minute': 0, 'last_reset_minute': time.time()},
            'twelvedata': {'daily': 0, 'minute': 0, 'last_reset_minute': time.time()}
        }
        self.last_daily_reset = datetime.now(UTC).date()
    
    def can_make_request(self, api_name: str) -> bool:
        """بررسی امکان درخواست جدید"""
        self._reset_counters_if_needed()
        
        limits = API_RATE_LIMITS[api_name]
        usage = self.usage[api_name]
        
        if usage['daily'] >= limits['daily_limit']:
            logging.warning(f"محدودیت روزانه {api_name} رسیده است: {usage['daily']}/{limits['daily_limit']}")
            return False
        
        if usage['minute'] >= limits['requests_per_minute']:
            logging.warning(f"محدودیت دقیقه‌ای {api_name} رسیده است: {usage['minute']}/{limits['requests_per_minute']}")
            return False
        
        return True
    
    def record_request(self, api_name: str):
        """ثبت درخواست جدید"""
        self.usage[api_name]['daily'] += 1
        self.usage[api_name]['minute'] += 1
    
    def _reset_counters_if_needed(self):
        """بازنشانی شمارنده‌ها در صورت نیاز"""
        current_date = datetime.now(UTC).date()
        current_time = time.time()
        
        # بازنشانی روزانه
        if current_date != self.last_daily_reset:
            for api in self.usage:
                self.usage[api]['daily'] = 0
            self.last_daily_reset = current_date
            logging.info("📊 شمارنده‌های روزانه API بازنشانی شدند")
        
        # بازنشانی دقیقه‌ای
        for api in self.usage:
            if current_time - self.usage[api]['last_reset_minute'] > 60:
                self.usage[api]['minute'] = 0
                self.usage[api]['last_reset_minute'] = current_time

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
            'momentum': ['rsi_14', 'stoch_14_3_3', 'macd', 'williams_r'],
            'volatility': ['bb_20_2', 'atr_14', 'kc_20'],
            'volume': ['obv', 'volume_sma_20', 'mfi'],
            'ichimoku': True,
            'support_resistance': True,
            'candle_patterns': True,
            'pivot_points': True
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
            df.ta.psar(append=True)  # Parabolic SAR
            
            # اندیکاتورهای مومنتوم
            df.ta.rsi(length=14, append=True)
            df.ta.stoch(append=True)
            df.ta.macd(append=True)
            df.ta.willr(length=14, append=True)  # Williams %R
            df.ta.cci(length=20, append=True)   # Commodity Channel Index
            
            # اندیکاتورهای نوسان
            df.ta.bbands(length=20, std=2, append=True)
            df.ta.atr(length=14, append=True)
            df.ta.kc(length=20, append=True)    # Keltner Channel
            
            # حجم
            if 'volume' in df.columns and not df['volume'].isnull().all():
                logging.info(f"ستون 'volume' شناسایی شد. محاسبه اندیکاتورهای حجم...")
                df.ta.obv(append=True)
                df['volume_sma_20'] = df['volume'].rolling(20).mean()
                df.ta.mfi(length=14, append=True)  # Money Flow Index
            else:
                logging.warning("ستون 'volume' در داده‌ها یافت نشد. اندیکاتورهای حجم نادیده گرفته شدند.")
            
            # ایچیموکو
            df.ta.ichimoku(append=True)
            
            # سطوح حمایت و مقاومت
            df['sup_1'] = df['low'].rolling(20).min().shift(1)
            df['res_1'] = df['high'].rolling(20).max().shift(1)
            df['sup_2'] = df['low'].rolling(50).min().shift(1)
            df['res_2'] = df['high'].rolling(50).max().shift(1)
            
            # پیوت پوینت‌ها
            df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
            df['r1'] = 2 * df['pivot'] - df['low']
            df['s1'] = 2 * df['pivot'] - df['high']
            df['r2'] = df['pivot'] + (df['high'] - df['low'])
            df['s2'] = df['pivot'] - (df['high'] - df['low'])
            
            # الگوهای کندل استیک
            if self.indicators_config['candle_patterns']:
                popular_patterns = ['doji', 'hammer', 'engulfing', 'harami', 'morningstar', 
                                  'eveningstar', 'piercing', 'darkcloud', 'hikkake']
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
        
        # تحلیل قدرت روند
        trend_strength = self._analyze_trend_strength(ltf_df)
        
        # سیگنال‌های ترکیبی
        composite_signals = self._generate_composite_signals(last_ltf, htf_trend, ltf_trend, momentum)
        
        return {
            'symbol': symbol,
            'htf_trend': htf_trend,
            'ltf_trend': ltf_trend,
            'momentum': momentum,
            'key_levels': key_levels,
            'candle_patterns': candle_analysis,
            'trend_strength': trend_strength,
            'composite_signals': composite_signals,
            'volatility': last_ltf.get('ATRr_14', 0),
            'timestamp': datetime.now(UTC).isoformat()
        }

    def _analyze_trend(self, data: pd.Series) -> Dict:
        ema_21 = data.get('EMA_21', 0)
        ema_50 = data.get('EMA_50', 0)
        ema_200 = data.get('EMA_200', 0)
        adx = data.get('ADX_14', 0)
        psar = data.get('PSARl_0.02_0.2', 0)
        
        # تحلیل پیشرفته روند
        ema_alignment = "صعودی" if ema_21 > ema_50 > ema_200 else "نزولی" if ema_21 < ema_50 < ema_200 else "خنثی"
        psar_signal = "صعودی" if data['close'] > psar else "نزولی"
        
        # قدرت روند بر اساس ADX
        if adx > 40:
            trend_strength = "بسیار قوی"
        elif adx > 25:
            trend_strength = "قوی"
        elif adx > 20:
            trend_strength = "متوسط"
        else:
            trend_strength = "ضعیف"
        
        return {
            'direction': ema_alignment,
            'strength': trend_strength,
            'adx': adx,
            'psar_signal': psar_signal,
            'ema_alignment': f"EMA21: {ema_21:.5f}, EMA50: {ema_50:.5f}, EMA200: {ema_200:.5f}"
        }

    def _analyze_momentum(self, data: pd.Series) -> Dict:
        rsi = data.get('RSI_14', 50)
        macd_hist = data.get('MACDh_12_26_9', 0)
        stoch_k = data.get('STOCHk_14_3_3', 50)
        williams_r = data.get('WILLR_14', -50)
        cci = data.get('CCI_20_0.015', 0)
        mfi = data.get('MFI_14', 50)
        
        rsi_signal = "اشباع خرید" if rsi > 70 else "اشباع فروش" if rsi < 30 else "خنثی"
        macd_signal = "صعودی" if macd_hist > 0 else "نزولی"
        stoch_signal = "اشباع خرید" if stoch_k > 80 else "اشباع فروش" if stoch_k < 20 else "خنثی"
        williams_signal = "اشباع خرید" if williams_r > -20 else "اشباع فروش" if williams_r < -80 else "خنثی"
        cci_signal = "اشباع خرید" if cci > 100 else "اشباع فروش" if cci < -100 else "خنثی"
        mfi_signal = "اشباع خرید" if mfi > 80 else "اشباع فروش" if mfi < 20 else "خنثی"
        
        # سیگنال ترکیبی مومنتوم
        bullish_signals = sum([
            macd_signal == "صعودی",
            rsi_signal == "اشباع فروش",
            stoch_signal == "اشباع فروش",
            williams_signal == "اشباع فروش"
        ])
        
        bearish_signals = sum([
            macd_signal == "نزولی",
            rsi_signal == "اشباع خرید",
            stoch_signal == "اشباع خرید",
            williams_signal == "اشباع خرید"
        ])
        
        momentum_bias = "صعودی" if bullish_signals > bearish_signals else "نزولی" if bearish_signals > bullish_signals else "خنثی"
        
        return {
            'rsi': {'value': rsi, 'signal': rsi_signal},
            'macd': {'signal': macd_signal, 'histogram': macd_hist},
            'stochastic': {'value': stoch_k, 'signal': stoch_signal},
            'williams_r': {'value': williams_r, 'signal': williams_signal},
            'cci': {'value': cci, 'signal': cci_signal},
            'mfi': {'value': mfi, 'signal': mfi_signal},
            'momentum_bias': momentum_bias,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals
        }

    def _analyze_trend_strength(self, df: pd.DataFrame) -> Dict:
        """تحلیل قدرت روند با استفاده از چندین اندیکاتور"""
        if len(df) < 2:
            return {'strength': 'ضعیف', 'score': 0}
        
        last = df.iloc[-1]
        
        # محاسبه امتیاز قدرت روند (0-100)
        strength_score = 0
        
        # ADX (0-25)
        adx = last.get('ADX_14', 0)
        strength_score += min(adx, 25)
        
        # EMA Alignment (0-25)
        ema_21 = last.get('EMA_21', 0)
        ema_50 = last.get('EMA_50', 0)
        ema_200 = last.get('EMA_200', 0)
        if (ema_21 > ema_50 > ema_200) or (ema_21 < ema_50 < ema_200):
            strength_score += 25
        
        # قیمت نسبت به باندهای بولینگر (0-25)
        bb_upper = last.get('BBU_20_2.0', 0)
        bb_lower = last.get('BBL_20_2.0', 0)
        close = last['close']
        bb_position = (close - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        if bb_position > 0.8 or bb_position < 0.2:
            strength_score += 25
        
        # حجم (0-25) - اگر موجود باشد
        if 'volume_sma_20' in last and last['volume_sma_20'] > 0:
            volume_ratio = last.get('volume', 0) / last['volume_sma_20']
            if volume_ratio > 1.2:
                strength_score += 25
        
        strength_level = "بسیار قوی" if strength_score > 75 else "قوی" if strength_score > 50 else "متوسط" if strength_score > 25 else "ضعیف"
        
        return {
            'strength': strength_level,
            'score': strength_score,
            'adx_contribution': min(adx, 25)
        }

    def _generate_composite_signals(self, data: pd.Series, htf_trend: Dict, ltf_trend: Dict, momentum: Dict) -> Dict:
        """تولید سیگنال‌های ترکیبی"""
        signals = []
        confidence = 0
        
        # سیگنال هماهنگی روندها
        if htf_trend['direction'] == ltf_trend['direction']:
            signals.append("هماهنگی روند بلندمدت و کوتاه‌مدت")
            confidence += 25
        
        # سیگنال قدرت روند
        if htf_trend['strength'] in ["قوی", "بسیار قوی"]:
            signals.append("روند بلندمدت قوی")
            confidence += 20
        
        # سیگنال مومنتوم
        if momentum['momentum_bias'] == htf_trend['direction']:
            signals.append("هماهنگی مومنتوم با روند")
            confidence += 20
        
        # سیگنال اشباع خرید/فروش
        oversold_conditions = sum([
            momentum['rsi']['signal'] == "اشباع فروش",
            momentum['stochastic']['signal'] == "اشباع فروش",
            momentum['williams_r']['signal'] == "اشباع فروش"
        ])
        
        overbought_conditions = sum([
            momentum['rsi']['signal'] == "اشباع خرید",
            momentum['stochastic']['signal'] == "اشباع خرید",
            momentum['williams_r']['signal'] == "اشباع خرید"
        ])
        
        if oversold_conditions >= 2 and htf_trend['direction'] == "صعودی":
            signals.append("اشباع فروش در روند صعودی")
            confidence += 15
        
        if overbought_conditions >= 2 and htf_trend['direction'] == "نزولی":
            signals.append("اشباع خرید در روند نزولی")
            confidence += 15
        
        # سیگنال شکست سطوح
        bb_upper = data.get('BBU_20_2.0', 0)
        bb_lower = data.get('BBL_20_2.0', 0)
        close = data['close']
        
        if close > bb_upper:
            signals.append("شکست مقاومت بولینگر")
        elif close < bb_lower:
            signals.append("شکست حمایت بولینگر")
        
        return {
            'signals': signals,
            'confidence_score': min(confidence, 100),
            'signal_strength': "قوی" if confidence >= 60 else "متوسط" if confidence >= 40 else "ضعیف"
        }

    def _analyze_key_levels(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame, current_price: float) -> Dict:
        bb_upper = ltf_df.get('BBU_20_2.0', pd.Series([0])).iloc[-1]
        bb_lower = ltf_df.get('BBL_20_2.0', pd.Series([0])).iloc[-1]
        bb_middle = ltf_df.get('BBM_20_2.0', pd.Series([0])).iloc[-1]
        
        kc_upper = ltf_df.get('KCUe_20_2', pd.Series([0])).iloc[-1]
        kc_lower = ltf_df.get('KCLe_20_2', pd.Series([0])).iloc[-1]
        
        support_1 = ltf_df.get('sup_1', pd.Series([0])).iloc[-1]
        resistance_1 = ltf_df.get('res_1', pd.Series([0])).iloc[-1]
        support_2 = ltf_df.get('sup_2', pd.Series([0])).iloc[-1]
        resistance_2 = ltf_df.get('res_2', pd.Series([0])).iloc[-1]
        
        # پیوت پوینت‌ها
        pivot = ltf_df.get('pivot', pd.Series([0])).iloc[-1]
        r1 = ltf_df.get('r1', pd.Series([0])).iloc[-1]
        s1 = ltf_df.get('s1', pd.Series([0])).iloc[-1]
        r2 = ltf_df.get('r2', pd.Series([0])).iloc[-1]
        s2 = ltf_df.get('s2', pd.Series([0])).iloc[-1]
        
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
                'resistance_1': r1,
                'support_1': s1,
                'resistance_2': r2,
                'support_2': s2
            },
            'current_price_position': self._get_price_position(current_price, support_1, resistance_1)
        }

    def _get_price_position(self, price: float, support: float, resistance: float) -> str:
        if resistance == support or resistance <= support:
            return "در محدوده خنثی"
        
        range_size = resistance - support
        position = (price - support) / range_size
        
        if position < 0.2:
            return "نزدیک حمایت قوی"
        elif position < 0.4:
            return "نزدیک حمایت"
        elif position > 0.8:
            return "نزدیک مقاومت قوی"
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
        
        # تحلیل سایه‌ها
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        total_shadow = upper_shadow + lower_shadow
        
        shadow_analysis = ""
        if upper_shadow > body_size * 2 and lower_shadow < body_size * 0.5:
            shadow_analysis = "سایه بالایی بلند - فشار فروش"
        elif lower_shadow > body_size * 2 and upper_shadow < body_size * 0.5:
            shadow_analysis = "سایه پایینی بلند - فشار خرید"
        
        return {
            'type': candle_type,
            'direction': direction,
            'body_ratio': body_ratio,
            'strength': "قوی" if body_ratio > 0.6 else "متوسط" if body_ratio > 0.3 else "ضعیف",
            'shadow_analysis': shadow_analysis,
            'upper_shadow_ratio': upper_shadow / total_range if total_range > 0 else 0,
            'lower_shadow_ratio': lower_shadow / total_range if total_range > 0 else 0
        }

# =================================================================================
# --- کلاس مدیریت AI چهارگانه (Gemini + 2 مدل Cloudflare + Groq) ---
# =================================================================================

class QuadAIManager:
    def __init__(self, gemini_api_key: str, cloudflare_api_key: str, groq_api_key: str):
        self.gemini_api_key = gemini_api_key
        self.cloudflare_api_key = cloudflare_api_key
        self.groq_api_key = groq_api_key
        self.gemini_model = GEMINI_MODEL
        
        # تنظیمات Cloudflare
        self.cloudflare_account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "your_account_id")
        self.cloudflare_models = CLOUDFLARE_MODELS
        
        # تنظیمات Groq
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        
        genai.configure(api_key=gemini_api_key)
        self.rate_manager = APIRateManager()
    
    async def get_quad_analysis(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        """دریافت تحلیل از چهار مدل AI و بررسی توافق"""
        tasks = [
            self._get_gemini_analysis(symbol, technical_analysis),
            self._get_cloudflare_analysis(symbol, technical_analysis, self.cloudflare_models[0], "Llama"),
            self._get_cloudflare_analysis(symbol, technical_analysis, self.cloudflare_models[1], "DeepSeek"),
            self._get_groq_analysis(symbol, technical_analysis)
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            gemini_result, llama_result, deepseek_result, groq_result = results
            
            # لاگ خطاها
            model_names = ["Gemini", "Llama", "DeepSeek", "Groq"]
            for i, (name, result) in enumerate(zip(model_names, results)):
                if isinstance(result, Exception):
                    logging.error(f"خطا در {name} برای {symbol}: {result}")
                    results[i] = None
            
            return self._combine_and_classify_signals(symbol, gemini_result, llama_result, deepseek_result, groq_result, technical_analysis)
            
        except Exception as e:
            logging.error(f"خطا در تحلیل چهارگانه برای {symbol}: {e}")
            return None
    
    async def _get_gemini_analysis(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        """تحلیل با Gemini"""
        if not self.rate_manager.can_make_request('google'):
            logging.warning("محدودیت Gemini API رسیده است")
            return None
            
        try:
            prompt = self._create_enhanced_analysis_prompt(symbol, technical_analysis)
            model = genai.GenerativeModel(self.gemini_model)
            
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                request_options={'timeout': 120}
            )
            
            self.rate_manager.record_request('google')
            return self._parse_ai_response(response.text, symbol, "Gemini")
            
        except Exception as e:
            logging.warning(f"خطا در تحلیل Gemini برای {symbol}: {e}")
            return None
    
    async def _get_cloudflare_analysis(self, symbol: str, technical_analysis: Dict, model_name: str, model_display_name: str) -> Optional[Dict]:
        """تحلیل با Cloudflare AI"""
        if not self.rate_manager.can_make_request('cloudflare'):
            logging.warning("محدودیت Cloudflare API رسیده است")
            return None
            
        if not self.cloudflare_api_key or self.cloudflare_account_id == "your_account_id":
            logging.warning("کلید یا شناسه حساب Cloudflare API تنظیم نشده است")
            return None
            
        try:
            prompt = self._create_enhanced_analysis_prompt(symbol, technical_analysis)
            
            headers = {
                "Authorization": f"Bearer {self.cloudflare_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are an expert forex trading analyst with 20 years experience. Provide concise analysis in valid JSON format only."
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
                        self.rate_manager.record_request('cloudflare')
                        
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

    async def _get_groq_analysis(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        """تحلیل با Groq API"""
        if not self.rate_manager.can_make_request('groq'):
            logging.warning("محدودیت Groq API رسیده است")
            return None
            
        if not self.groq_api_key:
            logging.warning("کلید Groq API تنظیم نشده است")
            return None
            
        try:
            prompt = self._create_enhanced_analysis_prompt(symbol, technical_analysis)
            
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional forex trading analyst. Always respond with valid JSON format only, no additional text."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "model": GROQ_MODEL,
                "temperature": 0.1,
                "max_tokens": 1024,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.groq_url, headers=headers, json=payload, timeout=120) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.rate_manager.record_request('groq')
                        
                        if "choices" in data and len(data["choices"]) > 0:
                            content = data["choices"][0]["message"]["content"]
                            return self._parse_ai_response(content, symbol, "Groq")
                        else:
                            logging.warning(f"فرمت پاسخ Groq نامعتبر است: {data}")
                            return None
                    else:
                        error_text = await response.text()
                        logging.warning(f"خطا در پاسخ Groq: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            logging.warning(f"خطا در تحلیل Groq برای {symbol}: {e}")
            return None

    def _create_enhanced_analysis_prompt(self, symbol: str, technical_analysis: Dict) -> str:
        """ایجاد پرامپت تحلیل پیشرفته"""
        composite = technical_analysis.get('composite_signals', {})
        trend_strength = technical_analysis.get('trend_strength', {})
        
        return f"""
به عنوان یک تحلیلگر حرفه‌ای بازار فارکس با ۲۰ سال تجربه، تحلیل تکنیکال زیر را برای جفت ارز {symbol} بررسی کنید و فقط و فقط یک آبجکت JSON معتبر برگردانید.

📊 **وضعیت تکنیکال پیشرفته {symbol}:**

🎯 **تحلیل روند:**
- روند بلندمدت (HTF): {technical_analysis['htf_trend']['direction']} - قدرت: {technical_analysis['htf_trend']['strength']} (ADX: {technical_analysis['htf_trend']['adx']:.1f})
- روند کوتاه‌مدت (LTF): {technical_analysis['ltf_trend']['direction']}
- قدرت کلی روند: {trend_strength.get('strength', 'نامشخص')} (امتیاز: {trend_strength.get('score', 0)}/100)

📈 **تحلیل مومنتوم:**
- RSI: {technical_analysis['momentum']['rsi']['value']:.1f} ({technical_analysis['momentum']['rsi']['signal']})
- MACD: {technical_analysis['momentum']['macd']['signal']}
- Stochastic: {technical_analysis['momentum']['stochastic']['signal']}
- تمایل مومنتوم: {technical_analysis['momentum']['momentum_bias']}
- سیگنال‌های صعودی: {technical_analysis['momentum']['bullish_signals']} | سیگنال‌های نزولی: {technical_analysis['momentum']['bearish_signals']}

🛡️ **سطوح کلیدی:**
- موقعیت قیمت: {technical_analysis['key_levels']['current_price_position']}
- مقاومت فوری: {technical_analysis['key_levels']['static']['resistance_1']:.5f}
- حمایت فوری: {technical_analysis['key_levels']['static']['support_1']:.5f}
- مقاومت اصلی: {technical_analysis['key_levels']['static']['resistance_2']:.5f}
- حمایت اصلی: {technical_analysis['key_levels']['static']['support_2']:.5f}
- باند بالایی: {technical_analysis['key_levels']['dynamic']['bb_upper']:.5f}
- باند پایینی: {technical_analysis['key_levels']['dynamic']['bb_lower']:.5f}

⚡ **سیگنال‌های ترکیبی:**
- قدرت سیگنال: {composite.get('signal_strength', 'نامشخص')}
- امتیاز اطمینان: {composite.get('confidence_score', 0)}/100
- سیگنال‌های شناسایی شده: {', '.join(composite.get('signals', []))}

💎 **الگوهای کندل استیک:**
- کندل فعلی: {technical_analysis['candle_patterns']['current_candle']['type']} - جهت: {technical_analysis['candle_patterns']['current_candle']['direction']}
- الگوهای اخیر: {', '.join(technical_analysis['candle_patterns']['recent_patterns'])}

**لطفاً پاسخ را فقط در قالب JSON زیر ارائه دهید (بدون هیچ متن اضافی):**

{{
  "SYMBOL": "{symbol}",
  "ACTION": "BUY/SELL/HOLD",
  "CONFIDENCE": 1-10,
  "ENTRY_ZONE": "عدد اعشاری (مثال: 1.12340-1.12400)",
  "STOP_LOSS": "عدد اعشاری (مثال: 1.12000)", 
  "TAKE_PROFIT_1": "عدد اعشاری (مثال: 1.12800)",
  "TAKE_PROFIT_2": "عدد اعشاری (مثال: 1.13000)",
  "RISK_REWARD_RATIO": "نسبت عددی (مثال: 1.8)",
  "ANALYSIS": "تحلیل مختصر فارسی با دلایل فنی",
  "EXPIRATION_H": "عدد صحیح (مثال: 6)",
  "TRADE_TYPE": "TREND_FOLLOWING/REVERSAL/BREAKOUT"
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

    def _combine_and_classify_signals(self, symbol: str, gemini_result: Dict, llama_result: Dict, deepseek_result: Dict, groq_result: Dict, technical_analysis: Dict) -> Optional[Dict]:
        """ترکیب نتایج چهار مدل AI و طبقه‌بندی بر اساس توافق"""
        valid_results = []
        
        if gemini_result and self._validate_signal_data(gemini_result, symbol):
            valid_results.append(('Gemini', gemini_result))
        
        if llama_result and self._validate_signal_data(llama_result, symbol):
            valid_results.append(('Llama', llama_result))
        
        if deepseek_result and self._validate_signal_data(deepseek_result, symbol):
            valid_results.append(('DeepSeek', deepseek_result))
            
        if groq_result and self._validate_signal_data(groq_result, symbol):
            valid_results.append(('Groq', groq_result))
        
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
        
        if agreement_level >= 3:
            # اجماع قوی (۳ یا ۴ موافق)
            majority_action = max(action_counts, key=action_counts.get)
            agreement_type = 'STRONG_CONSENSUS'
            
            agreeing_results = [result for _, result in valid_results if result['ACTION'].upper() == majority_action]
            combined_signal = self._average_agreeing_signals(symbol, agreeing_results, majority_action)
            
        elif agreement_level == 2:
            # اجماع متوسط (۲ موافق)
            majority_action = max(action_counts, key=action_counts.get)
            agreement_type = 'MEDIUM_CONSENSUS'
            
            agreeing_results = [result for _, result in valid_results if result['ACTION'].upper() == majority_action]
            combined_signal = self._average_agreeing_signals(symbol, agreeing_results, majority_action)
            
        else:
            # عدم توافق (۰ یا ۱ موافق)
            agreement_type = 'NO_CONSENSUS'
            # انتخاب مدل با بیشترین اعتماد
            highest_confidence_model = max(valid_results, key=lambda x: float(x[1].get('CONFIDENCE', 0)))
            combined_signal = highest_confidence_model[1]
            combined_signal['CONFIDENCE'] = max(1, int(float(combined_signal.get('CONFIDENCE', 5)) - 3))
        
        combined_signal['AGREEMENT_LEVEL'] = agreement_level
        combined_signal['AGREEMENT_TYPE'] = agreement_type
        combined_signal['VALID_MODELS'] = total_models
        combined_signal['CONSENSUS_ANALYSIS'] = self._generate_consensus_analysis(agreement_type, agreement_level, total_models)
        combined_signal['TECHNICAL_SCORE'] = technical_analysis.get('composite_signals', {}).get('confidence_score', 0)
        
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
        numeric_fields = ['ENTRY_ZONE', 'STOP_LOSS', 'TAKE_PROFIT_1', 'TAKE_PROFIT_2', 'EXPIRATION_H']
        
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
        averaged['TRADE_TYPE'] = agreeing_results[0].get('TRADE_TYPE', 'TREND_FOLLOWING')
        
        # ذخیره تحلیل‌های جداگانه
        model_analyses = {}
        for result in agreeing_results:
            model_analyses[result['ai_model']] = result.get('ANALYSIS', '')
        
        averaged['MODEL_ANALYSES'] = model_analyses
        averaged['CONSENSUS_DETAIL'] = f"توافق بین {len(agreeing_results)} مدل از {len(agreeing_results)} مدل معتبر"
        
        return averaged

    def _generate_consensus_analysis(self, agreement_type: str, agreement_level: int, total_models: int) -> str:
        """تولید تحلیل توافق"""
        if agreement_type == 'STRONG_CONSENSUS':
            if agreement_level == 4:
                return "توافق کامل بین هر چهار مدل AI - سیگنال با اعتماد بسیار بالا"
            elif agreement_level == 3:
                return f"توافق قوی بین ۳ مدل از {total_models} مدل - سیگنال با اعتماد بالا"
        elif agreement_type == 'MEDIUM_CONSENSUS':
            return f"توافق متوسط بین ۲ مدل از {total_models} مدل - سیگنال با اعتماد متوسط"
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
        self.ai_manager = QuadAIManager(google_api_key, CLOUDFLARE_AI_API_KEY, GROQ_API_KEY)

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
            
            # تحلیل چهارگانه AI
            ai_analysis = await self.ai_manager.get_quad_analysis(pair, technical_analysis)
            
            if ai_analysis and ai_analysis.get('ACTION') != 'HOLD':
                self.cache_manager.update_cache(pair, ai_analysis)
                logging.info(f"✅ سیگنال معاملاتی برای {pair}: {ai_analysis['ACTION']} (توافق: {ai_analysis.get('AGREEMENT_LEVEL', 0)}/4)")
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
    logging.info("🎯 شروع سیستم تحلیل فارکس پیشرفته (Quad AI v4.0)")
    
    import argparse
    parser = argparse.ArgumentParser(description='سیستم تحلیل فارکس با AI چهارگانه')
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
    strong_consensus_signals = []
    medium_consensus_signals = []
    weak_consensus_signals = []
    
    for signal in signals:
        agreement_level = signal.get('AGREEMENT_LEVEL', 0)
        if agreement_level >= 3:  # توافق ۳ یا ۴ مدل
            strong_consensus_signals.append(signal)
        elif agreement_level == 2:  # توافق ۲ مدل
            medium_consensus_signals.append(signal)
        else:  # توافق ۰ یا ۱ مدل
            weak_consensus_signals.append(signal)

    # ذخیره سیگنال‌های با توافق قوی
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
                'AGREEMENT_TYPE': signal.get('AGREEMENT_TYPE', 'UNKNOWN'),
                'ENTRY_ZONE': signal.get('ENTRY_ZONE', 'N/A'),
                'STOP_LOSS': signal.get('STOP_LOSS', 'N/A'),
                'TAKE_PROFIT_1': signal.get('TAKE_PROFIT_1', 'N/A'),
                'TAKE_PROFIT_2': signal.get('TAKE_PROFIT_2', 'N/A'),
                'RISK_REWARD_RATIO': signal.get('RISK_REWARD_RATIO', 'N/A'),
                'EXPIRATION_H': signal.get('EXPIRATION_H', 0),
                'TRADE_TYPE': signal.get('TRADE_TYPE', 'N/A'),
                'TECHNICAL_SCORE': signal.get('TECHNICAL_SCORE', 0),
                'CONSENSUS_ANALYSIS': signal.get('CONSENSUS_ANALYSIS', ''),
                'TIMESTAMP': signal.get('timestamp', datetime.now(UTC).isoformat())
            }
            cleaned_strong_signals.append(cleaned_signal)
        
        with open(strong_conf_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_strong_signals, f, indent=4, ensure_ascii=False)
        
        logging.info(f"🎯 {len(strong_consensus_signals)} سیگنال با توافق قوی در {strong_conf_file} ذخیره شد")

    # ذخیره سیگنال‌های با توافق متوسط
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
                'AGREEMENT_TYPE': signal.get('AGREEMENT_TYPE', 'UNKNOWN'),
                'ENTRY_ZONE': signal.get('ENTRY_ZONE', 'N/A'),
                'STOP_LOSS': signal.get('STOP_LOSS', 'N/A'),
                'TAKE_PROFIT_1': signal.get('TAKE_PROFIT_1', 'N/A'),
                'TAKE_PROFIT_2': signal.get('TAKE_PROFIT_2', 'N/A'),
                'RISK_REWARD_RATIO': signal.get('RISK_REWARD_RATIO', 'N/A'),
                'EXPIRATION_H': signal.get('EXPIRATION_H', 0),
                'TRADE_TYPE': signal.get('TRADE_TYPE', 'N/A'),
                'TECHNICAL_SCORE': signal.get('TECHNICAL_SCORE', 0),
                'CONSENSUS_ANALYSIS': signal.get('CONSENSUS_ANALYSIS', ''),
                'TIMESTAMP': signal.get('timestamp', datetime.now(UTC).isoformat())
            }
            cleaned_medium_signals.append(cleaned_signal)
        
        with open(medium_conf_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_medium_signals, f, indent=4, ensure_ascii=False)
        
        logging.info(f"📊 {len(medium_consensus_signals)} سیگنال با توافق متوسط در {medium_conf_file} ذخیره شد")

    # ذخیره سیگنال‌های با توافق ضعیف
    if weak_consensus_signals:
        weak_conf_file = "weak_consensus_signals.json"
        cleaned_weak_signals = []
        
        for signal in weak_consensus_signals:
            cleaned_signal = {
                'SYMBOL': signal.get('SYMBOL', 'Unknown'),
                'ACTION': signal.get('ACTION', 'HOLD'),
                'CONFIDENCE': signal.get('CONFIDENCE', 0),
                'AGREEMENT_LEVEL': signal.get('AGREEMENT_LEVEL', 0),
                'VALID_MODELS': signal.get('VALID_MODELS', 0),
                'AGREEMENT_TYPE': signal.get('AGREEMENT_TYPE', 'UNKNOWN'),
                'ENTRY_ZONE': signal.get('ENTRY_ZONE', 'N/A'),
                'STOP_LOSS': signal.get('STOP_LOSS', 'N/A'),
                'TAKE_PROFIT_1': signal.get('TAKE_PROFIT_1', 'N/A'),
                'TAKE_PROFIT_2': signal.get('TAKE_PROFIT_2', 'N/A'),
                'RISK_REWARD_RATIO': signal.get('RISK_REWARD_RATIO', 'N/A'),
                'EXPIRATION_H': signal.get('EXPIRATION_H', 0),
                'TRADE_TYPE': signal.get('TRADE_TYPE', 'N/A'),
                'TECHNICAL_SCORE': signal.get('TECHNICAL_SCORE', 0),
                'CONSENSUS_ANALYSIS': signal.get('CONSENSUS_ANALYSIS', ''),
                'TIMESTAMP': signal.get('timestamp', datetime.now(UTC).isoformat())
            }
            cleaned_weak_signals.append(cleaned_signal)
        
        with open(weak_conf_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_weak_signals, f, indent=4, ensure_ascii=False)
        
        logging.info(f"📈 {len(weak_consensus_signals)} سیگنال با توافق ضعیف در {weak_conf_file} ذخیره شد")

    # نمایش خلاصه نتایج
    logging.info("📈 خلاصه سیگنال‌های معاملاتی:")
    
    logging.info("🎯 سیگنال‌های با توافق قوی (۳-۴ مدل):")
    for signal in strong_consensus_signals:
        action_icon = "🟢" if signal['ACTION'] == 'BUY' else "🔴" if signal['ACTION'] == 'SELL' else "⚪"
        logging.info(f"  {action_icon} {signal['SYMBOL']}: {signal['ACTION']} (اعتماد: {signal['CONFIDENCE']}/10, توافق: {signal['AGREEMENT_LEVEL']}/4)")
    
    logging.info("📊 سیگنال‌های با توافق متوسط (۲ مدل):")
    for signal in medium_consensus_signals:
        action_icon = "🟢" if signal['ACTION'] == 'BUY' else "🔴" if signal['ACTION'] == 'SELL' else "⚪"
        logging.info(f"  {action_icon} {signal['SYMBOL']}: {signal['ACTION']} (اعتماد: {signal['CONFIDENCE']}/10, توافق: {signal['AGREEMENT_LEVEL']}/4)")
    
    logging.info("📈 سیگنال‌های با توافق ضعیف (۰-۱ مدل):")
    for signal in weak_consensus_signals:
        action_icon = "🟢" if signal['ACTION'] == 'BUY' else "🔴" if signal['ACTION'] == 'SELL' else "⚪"
        logging.info(f"  {action_icon} {signal['SYMBOL']}: {signal['ACTION']} (اعتماد: {signal['CONFIDENCE']}/10, توافق: {signal['AGREEMENT_LEVEL']}/4)")

    if not signals:
        logging.info("🔍 هیچ سیگنال معاملاتی‌ای در این اجرا شناسایی نشد")

    logging.info("🏁 پایان اجرای سیستم")

if __name__ == "__main__":
    asyncio.run(main())
