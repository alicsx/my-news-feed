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
import numpy as np
from scipy import stats
import traceback
import yfinance as yf
from dataclasses import dataclass
from enum import Enum
import random
import hashlib
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# =================================================================================
# --- Enhanced Main Configuration Section ---
# =================================================================================

# API Keys
google_api_key = os.getenv("GOOGLE_API_KEY")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
CLOUDFLARE_AI_API_KEY = os.getenv("CLOUDFLARE_AI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not all([google_api_key, TWELVEDATA_API_KEY]):
    logging.warning("⚠️ Some API keys are missing. System will use fallback methods.")

# Main system configuration
HIGH_TIMEFRAME = "4h"
LOW_TIMEFRAME = "1h"
CANDLES_TO_FETCH = 500

CURRENCY_PAIRS_TO_ANALYZE = [
    "EUR/USD", "GBP/USD", "USD/CHF", "EUR/JPY", 
    "AUD/JPY", "GBP/JPY", "EUR/AUD", "NZD/CAD"
]

# Alternative symbol formats for different data sources
YAHOO_SYMBOLS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X", 
    "USD/CHF": "USDCHF=X",
    "EUR/JPY": "EURJPY=X",
    "AUD/JPY": "AUDJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "EUR/AUD": "EURAUD=X",
    "NZD/CAD": "NZDCAD=X"
}

CACHE_FILE = "signal_cache.json"
USAGE_TRACKER_FILE = "api_usage_tracker.json"
LOG_FILE = "trading_log.log"

# Updated AI models with more diversity
GEMINI_MODEL = 'gemini-2.5-flash'

# Enhanced Cloudflare models
CLOUDFLARE_MODELS = [
    "@cf/meta/llama-4-scout-17b-16e-instruct",  # جدیدترین
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast", # جدید
    "@cf/meta/llama-3.1-8b-instruct-fast",      # جدید
    "@cf/google/gemma-3-12b-it",                # جدید
    "@cf/mistralai/mistral-small-3.1-24b-instruct" # جدید
]

# Enhanced Groq models
GROQ_MODELS = [
    "qwen/qwen3-32b",                                   # جدید
    "llama-3.3-70b-versatile",                          # جدیدترین
    "meta-llama/llama-4-maverick-17b-128e-instruct",    # جدید
    "meta-llama/llama-4-scout-17b-16e-instruct",        # جدید
    "llama-3.1-8b-instant"                              # جدید
]

# Daily API limits
API_DAILY_LIMITS = {
    "google_gemini": 1500,
    "cloudflare": 10000,
    "groq": 10000
}

# Data source priorities
DATA_SOURCE_PRIORITY = ["twelvedata", "yahoo", "synthetic"]

# Rate limiting configuration
TWELVEDATA_RATE_LIMIT = 8  # requests per minute
MIN_REQUEST_INTERVAL = 60 / TWELVEDATA_RATE_LIMIT  # seconds between requests

# Advanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# =================================================================================
# --- Enhanced Performance Monitoring System ---
# =================================================================================

class PerformanceMonitor:
    """System performance monitoring and optimization"""
    
    def __init__(self):
        self.analysis_times = deque(maxlen=100)
        self.api_response_times = deque(maxlen=50)
        self.error_rates = deque(maxlen=50)
        self.successful_analyses = 0
        self.failed_analyses = 0
        
    def record_analysis_time(self, symbol: str, duration: float):
        """Record analysis duration"""
        self.analysis_times.append((symbol, duration))
        
    def record_api_time(self, provider: str, duration: float):
        """Record API response time"""
        self.api_response_times.append((provider, duration))
        
    def record_success(self):
        """Record successful analysis"""
        self.successful_analyses += 1
        
    def record_failure(self):
        """Record failed analysis"""
        self.failed_analyses += 1
        
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        total_analyses = self.successful_analyses + self.failed_analyses
        success_rate = (self.successful_analyses / total_analyses * 100) if total_analyses > 0 else 0
        
        avg_analysis_time = np.mean([t[1] for t in self.analysis_times]) if self.analysis_times else 0
        avg_api_time = np.mean([t[1] for t in self.api_response_times]) if self.api_response_times else 0
        
        return {
            "total_analyses": total_analyses,
            "success_rate": round(success_rate, 2),
            "avg_analysis_time_sec": round(avg_analysis_time, 2),
            "avg_api_response_time_sec": round(avg_api_time, 2),
            "recent_analysis_times": list(self.analysis_times)[-5:],
            "recent_api_times": list(self.api_response_times)[-5:]
        }

# =================================================================================
# --- Enhanced Data Source Management ---
# =================================================================================

class DataSource(Enum):
    TWELVEDATA = "twelvedata"
    YAHOO = "yahoo"
    SYNTHETIC = "synthetic"

@dataclass
class DataFetchResult:
    success: bool
    data: Optional[pd.DataFrame]
    source: DataSource
    symbol: str
    error: Optional[str] = None

class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self.last_request_time = 0
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.interval:
                wait_time = self.interval - time_since_last
                logging.debug(f"⏳ Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            self.last_request_time = time.time()

class EnhancedDataFetcher:
    def __init__(self):
        self.rate_limiter = RateLimiter(TWELVEDATA_RATE_LIMIT)
        self.data_source_priority = DATA_SOURCE_PRIORITY.copy()
        self.last_data_source = {}
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache
        
    async def get_market_data(self, symbol: str, interval: str, max_retries: int = 2) -> Optional[pd.DataFrame]:
        """Get market data with multiple fallback sources and rate limiting"""
        
        # Check cache first
        cache_key = f"{symbol}_{interval}"
        current_time = time.time()
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_ttl:
                logging.info(f"📦 Using cached data for {symbol} ({interval})")
                return cached_data
        
        for source in self.data_source_priority:
            try:
                if source == "twelvedata" and TWELVEDATA_API_KEY:
                    result = await self._get_twelvedata_with_retry(symbol, interval, max_retries)
                    if result.success:
                        self.last_data_source[symbol] = DataSource.TWELVEDATA
                        self.cache[cache_key] = (result.data, current_time)
                        return result.data
                        
                elif source == "yahoo":
                    result = await self._get_yahoo_data(symbol, interval)
                    if result.success:
                        self.last_data_source[symbol] = DataSource.YAHOO
                        self.cache[cache_key] = (result.data, current_time)
                        return result.data
                        
                elif source == "synthetic":
                    result = await self._get_synthetic_data(symbol, interval)
                    if result.success:
                        self.last_data_source[symbol] = DataSource.SYNTHETIC
                        self.cache[cache_key] = (result.data, current_time)
                        return result.data
                        
            except Exception as e:
                logging.warning(f"❌ {source} failed for {symbol}: {str(e)}")
                continue
                
        logging.error(f"❌ All data sources failed for {symbol}")
        return None

    async def _get_twelvedata_with_retry(self, symbol: str, interval: str, max_retries: int) -> DataFetchResult:
        """Get data from Twelve Data with rate limiting and retry logic"""
        
        for attempt in range(max_retries):
            try:
                # Apply rate limiting
                await self.rate_limiter.acquire()
                
                result = await self._get_twelvedata_data(symbol, interval)
                if result.success:
                    return result
                    
                logging.warning(f"⚠️ TwelveData attempt {attempt + 1} failed for {symbol}: {result.error}")
                
                # Wait before retry
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    await asyncio.sleep(wait_time)
                    
            except Exception as e:
                logging.warning(f"❌ TwelveData error on attempt {attempt + 1} for {symbol}: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep((attempt + 1) * 2)
                    
        return DataFetchResult(False, None, DataSource.TWELVEDATA, symbol, "All retries failed")

    async def _get_twelvedata_data(self, symbol: str, interval: str) -> DataFetchResult:
        """Get data from Twelve Data API with enhanced error handling"""
        try:
            url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={CANDLES_TO_FETCH}&apikey={TWELVEDATA_API_KEY}'
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check for API errors
                        if 'code' in data and data['code'] != 200:
                            error_msg = data.get('message', 'Unknown API error')
                            
                            # Handle specific error codes
                            if data['code'] == 429:
                                logging.warning(f"🔁 Rate limit hit for {symbol}, will use fallback")
                                return DataFetchResult(False, None, DataSource.TWELVEDATA, symbol, "Rate limit exceeded")
                            elif data['code'] == 400:
                                logging.warning(f"⚠️ Invalid symbol {symbol} for TwelveData")
                                return DataFetchResult(False, None, DataSource.TWELVEDATA, symbol, "Invalid symbol")
                            else:
                                logging.warning(f"⚠️ TwelveData API error for {symbol}: {error_msg}")
                                return DataFetchResult(False, None, DataSource.TWELVEDATA, symbol, error_msg)
                        
                        if 'values' in data and data['values']:
                            df = pd.DataFrame(data['values'])
                            # Reverse to get chronological order
                            df = df.iloc[::-1].reset_index(drop=True)
                            
                            # Convert to numeric
                            for col in ['open', 'high', 'low', 'close']:
                                if col in df.columns:
                                    df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                            # Remove any rows with NaN values in essential columns
                            df = df.dropna(subset=['open', 'high', 'low', 'close'])
                            
                            if len(df) > 50:
                                logging.info(f"✅ TwelveData: {len(df)} candles for {symbol} ({interval})")
                                return DataFetchResult(True, df, DataSource.TWELVEDATA, symbol)
                            else:
                                return DataFetchResult(False, None, DataSource.TWELVEDATA, symbol, "Insufficient data after cleaning")
                        else:
                            return DataFetchResult(False, None, DataSource.TWELVEDATA, symbol, "No values in response")
                    else:
                        error_text = await response.text()
                        logging.warning(f"⚠️ TwelveData HTTP {response.status} for {symbol}: {error_text}")
                        return DataFetchResult(False, None, DataSource.TWELVEDATA, symbol, f"HTTP {response.status}")
                        
        except asyncio.TimeoutError:
            logging.warning(f"⏰ TwelveData timeout for {symbol}")
            return DataFetchResult(False, None, DataSource.TWELVEDATA, symbol, "Timeout")
        except Exception as e:
            logging.warning(f"❌ TwelveData exception for {symbol}: {str(e)}")
            return DataFetchResult(False, None, DataSource.TWELVEDATA, symbol, str(e))

    async def _get_yahoo_data(self, symbol: str, interval: str) -> DataFetchResult:
        """Get data from Yahoo Finance as fallback"""
        try:
            # Convert symbol to Yahoo format
            yahoo_symbol = YAHOO_SYMBOLS.get(symbol, symbol.replace("/", "") + "=X")
            
            # Map intervals
            interval_map = {
                "1h": "1h",
                "4h": "4h", 
                "1d": "1d",
                "1m": "1m",
                "5m": "5m",
                "15m": "15m"
            }
            
            yf_interval = interval_map.get(interval, "1h")
            period = "60d" if interval in ["1h", "4h"] else "120d"
            
            # Download data using async thread
            ticker = yf.Ticker(yahoo_symbol)
            df = await asyncio.to_thread(
                ticker.history, 
                period=period, 
                interval=yf_interval,
                timeout=30
            )
            
            if df.empty:
                return DataFetchResult(False, None, DataSource.YAHOO, symbol, "Empty DataFrame from Yahoo")
                
            # Reset index and rename columns
            df = df.reset_index()
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close']
            if all(col in df.columns for col in required_cols):
                # Clean data
                for col in required_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna(subset=required_cols)
                
                if len(df) > 50:
                    logging.info(f"✅ Yahoo Finance: {len(df)} candles for {symbol} ({interval})")
                    return DataFetchResult(True, df.tail(CANDLES_TO_FETCH), DataSource.YAHOO, symbol)
                else:
                    return DataFetchResult(False, None, DataSource.YAHOO, symbol, "Insufficient data after cleaning")
            else:
                return DataFetchResult(False, None, DataSource.YAHOO, symbol, "Missing required columns")
                
        except Exception as e:
            logging.warning(f"❌ Yahoo Finance error for {symbol}: {str(e)}")
            return DataFetchResult(False, None, DataSource.YAHOO, symbol, str(e))

    async def _get_synthetic_data(self, symbol: str, interval: str) -> DataFetchResult:
        """Generate synthetic data as last resort fallback"""
        try:
            # Create realistic synthetic data based on common forex patterns
            np.random.seed(hash(symbol) % 10000)
            
            # Base prices for different pairs (approximate)
            base_prices = {
                "EUR/USD": 1.0850, "GBP/USD": 1.2650, "USD/CHF": 0.9050,
                "EUR/JPY": 158.50, "AUD/JPY": 97.50, "GBP/JPY": 187.50,
                "EUR/AUD": 1.6350, "NZD/CAD": 0.8150
            }
            
            base_price = base_prices.get(symbol, 1.0)
            volatility = 0.002  # 0.2% volatility
            
            # Generate synthetic data
            n_candles = CANDLES_TO_FETCH
            returns = np.random.normal(0, volatility, n_candles)
            prices = base_price * (1 + returns).cumprod()
            
            # Create OHLC data with some randomness
            df = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.0005, n_candles)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.001, n_candles))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.001, n_candles))),
                'close': prices
            })
            
            # Ensure high >= open, high >= close, low <= open, low <= close
            df['high'] = df[['open', 'close', 'high']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.0002, n_candles)))
            df['low'] = df[['open', 'close', 'low']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.0002, n_candles)))
            
            # Add datetime index
            end_date = datetime.now(UTC)
            if interval == "1h":
                dates = pd.date_range(end=end_date, periods=n_candles, freq='1H')
            else:  # 4h
                dates = pd.date_range(end=end_date, periods=n_candles, freq='4H')
            
            df['datetime'] = dates
            df = df.set_index('datetime')
            
            logging.info(f"🔄 Synthetic data generated for {symbol} ({interval}): {len(df)} candles")
            return DataFetchResult(True, df, DataSource.SYNTHETIC, symbol)
            
        except Exception as e:
            logging.warning(f"❌ Synthetic data generation failed for {symbol}: {str(e)}")
            return DataFetchResult(False, None, DataSource.SYNTHETIC, symbol, str(e))

    def get_data_source_stats(self) -> Dict:
        """Get statistics about data sources used"""
        stats = {}
        for source in DataSource:
            count = list(self.last_data_source.values()).count(source)
            stats[source.value] = count
        return stats

# =================================================================================
# --- Advanced Technical Analysis with Machine Learning Features ---
# =================================================================================

class AdvancedTechnicalAnalyzer:
    def __init__(self):
        self.indicators_config = {
            'trend': ['ema_8', 'ema_21', 'ema_50', 'ema_200', 'wma_34', 'hma_55', 'adx_14', 'ichimoku'],
            'momentum': ['rsi_14', 'stoch_14_3_3', 'macd', 'cci_20', 'williams_14', 'momentum_10'],
            'volatility': ['bb_20_2', 'bb_20_1.5', 'atr_14', 'kc_20_2'],
            'volume': ['obv', 'cmf_20', 'vwap'],
            'advanced': ['supertrend', 'parabolic_sar', 'donchian_20', 'pivot_points']
        }
        self.ml_features = {}

    def calculate_enhanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators with robust error handling"""
        if df is None or df.empty:
            logging.warning("Empty DataFrame provided to indicator calculation")
            return None
            
        try:
            # Make a copy to avoid modifying original
            df_indicators = df.copy()
            
            # Ensure numeric columns and handle errors
            for col in ['open', 'high', 'low', 'close']:
                if col in df_indicators.columns:
                    df_indicators[col] = pd.to_numeric(df_indicators[col], errors='coerce')
            
            # Remove any rows with NaN values in essential price columns
            df_indicators = df_indicators.dropna(subset=['open', 'high', 'low', 'close'])
            
            if len(df_indicators) < 100:
                logging.warning(f"Insufficient data after cleaning: {len(df_indicators)} rows")
                return None

            # Calculate indicators with individual error handling
            indicators_added = []
            
            # Trend indicators
            trend_indicators = ['EMA_8', 'EMA_21', 'EMA_50', 'EMA_200', 'WMA_34', 'HMA_55', 'ADX_14']
            for indicator in trend_indicators:
                try:
                    if indicator.startswith('EMA'):
                        length = int(indicator.split('_')[1])
                        df_indicators.ta.ema(length=length, append=True)
                        indicators_added.append(indicator)
                    elif indicator == 'WMA_34':
                        df_indicators.ta.wma(length=34, append=True)
                        indicators_added.append(indicator)
                    elif indicator == 'HMA_55':
                        df_indicators.ta.hma(length=55, append=True)
                        indicators_added.append(indicator)
                    elif indicator == 'ADX_14':
                        df_indicators.ta.adx(length=14, append=True)
                        indicators_added.append(indicator)
                except Exception as e:
                    logging.warning(f"Failed to calculate {indicator}: {e}")

            # Momentum indicators
            momentum_indicators = ['RSI_14', 'STOCHk_14_3_3', 'MACD_12_26_9', 'CCI_20_0.015', 'WILLR_14', 'MOM_10']
            for indicator in momentum_indicators:
                try:
                    if indicator == 'RSI_14':
                        df_indicators.ta.rsi(length=14, append=True)
                    elif indicator == 'STOCHk_14_3_3':
                        df_indicators.ta.stoch(append=True)
                    elif indicator == 'MACD_12_26_9':
                        df_indicators.ta.macd(append=True)
                    elif indicator == 'CCI_20_0.015':
                        df_indicators.ta.cci(length=20, append=True)
                    elif indicator == 'WILLR_14':
                        df_indicators.ta.willr(length=14, append=True)
                    elif indicator == 'MOM_10':
                        df_indicators.ta.mom(length=10, append=True)
                    indicators_added.append(indicator)
                except Exception as e:
                    logging.warning(f"Failed to calculate {indicator}: {e}")

            # Volatility indicators
            volatility_indicators = ['BBL_20_2.0', 'BBU_20_2.0', 'ATRr_14', 'KCLe_20_2', 'KCUe_20_2']
            for indicator in volatility_indicators:
                try:
                    if indicator.startswith('BB'):
                        df_indicators.ta.bbands(length=20, std=2, append=True)
                    elif indicator == 'ATRr_14':
                        df_indicators.ta.atr(length=14, append=True)
                    elif indicator.startswith('KC'):
                        df_indicators.ta.kc(length=20, scalar=2, append=True)
                    indicators_added.append(indicator)
                except Exception as e:
                    logging.warning(f"Failed to calculate {indicator}: {e}")

            # Advanced indicators
            advanced_indicators = ['SUPERT_7_3.0', 'PSARl_0.02_0.2', 'DCP_20', 'DCM_20', 'DCU_20']
            for indicator in advanced_indicators:
                try:
                    if indicator.startswith('SUPERT'):
                        df_indicators.ta.supertrend(append=True)
                    elif indicator.startswith('PSAR'):
                        df_indicators.ta.psar(append=True)
                    elif indicator.startswith('DC'):
                        df_indicators.ta.donchian(lower_length=20, upper_length=20, append=True)
                    indicators_added.append(indicator)
                except Exception as e:
                    logging.warning(f"Failed to calculate {indicator}: {e}")

            # Ichimoku Cloud
            try:
                df_indicators.ta.ichimoku(append=True)
                indicators_added.extend(['ISA_9', 'ISB_26', 'ICS_26', 'ICB_26', 'ITS_9'])
            except Exception as e:
                logging.warning(f"Failed to calculate Ichimoku: {e}")

            # Support and resistance levels
            try:
                df_indicators['pivot'] = (df_indicators['high'] + df_indicators['low'] + df_indicators['close']) / 3
                df_indicators['r1'] = 2 * df_indicators['pivot'] - df_indicators['low']
                df_indicators['s1'] = 2 * df_indicators['pivot'] - df_indicators['high']
                df_indicators['sup_1'] = df_indicators['low'].rolling(20, min_periods=1).min().shift(1)
                df_indicators['res_1'] = df_indicators['high'].rolling(20, min_periods=1).max().shift(1)
                df_indicators['sup_2'] = df_indicators['low'].rolling(50, min_periods=1).min().shift(1)
                df_indicators['res_2'] = df_indicators['high'].rolling(50, min_periods=1).max().shift(1)
                indicators_added.extend(['pivot', 'r1', 's1', 'sup_1', 'res_1', 'sup_2', 'res_2'])
            except Exception as e:
                logging.warning(f"Failed to calculate support/resistance: {e}")

            # Price action patterns
            try:
                df_indicators['inside_bar'] = ((df_indicators['high'] < df_indicators['high'].shift(1)) & (df_indicators['low'] > df_indicators['low'].shift(1)))
                df_indicators['outside_bar'] = ((df_indicators['high'] > df_indicators['high'].shift(1)) & (df_indicators['low'] < df_indicators['low'].shift(1)))
                indicators_added.extend(['inside_bar', 'outside_bar'])
            except Exception as e:
                logging.warning(f"Failed to calculate price patterns: {e}")

            # Advanced ML features
            try:
                # Price volatility features
                df_indicators['price_range'] = (df_indicators['high'] - df_indicators['low']) / df_indicators['close']
                df_indicators['price_change'] = df_indicators['close'].pct_change()
                df_indicators['volatility_20'] = df_indicators['price_change'].rolling(20).std()
                
                # Momentum features
                df_indicators['momentum_5'] = df_indicators['close'] / df_indicators['close'].shift(5) - 1
                df_indicators['momentum_10'] = df_indicators['close'] / df_indicators['close'].shift(10) - 1
                
                # Mean reversion features
                df_indicators['z_score_20'] = (df_indicators['close'] - df_indicators['close'].rolling(20).mean()) / df_indicators['close'].rolling(20).std()
                
                indicators_added.extend(['price_range', 'price_change', 'volatility_20', 'momentum_5', 'momentum_10', 'z_score_20'])
            except Exception as e:
                logging.warning(f"Failed to calculate ML features: {e}")

            # Remove rows with too many NaN values but keep recent data
            initial_count = len(df_indicators)
            df_indicators = df_indicators.dropna(thresh=len(df_indicators.columns) - 15)  # Allow up to 15 NaN columns
            
            if len(df_indicators) < 50:
                logging.warning(f"Too many NaN values after cleaning: {len(df_indicators)} rows left")
                # Keep recent data even with some NaN values
                df_indicators = df_indicators.tail(100).fillna(method='ffill').fillna(method='bfill')

            logging.info(f"✅ Successfully calculated {len(indicators_added)} indicators for {len(df_indicators)} rows")
            return df_indicators
            
        except Exception as e:
            logging.error(f"❌ Critical error in indicator calculation: {e}")
            # Fallback: return original DataFrame with basic indicators
            return self._calculate_basic_indicators(df)

    def _calculate_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback method for basic indicator calculation"""
        try:
            df_basic = df.copy()
            
            # Basic indicators that rarely fail
            df_basic.ta.ema(length=21, append=True)
            df_basic.ta.ema(length=50, append=True)
            df_basic.ta.rsi(length=14, append=True)
            df_basic.ta.macd(append=True)
            df_basic.ta.bbands(length=20, std=2, append=True)
            df_basic.ta.atr(length=14, append=True)
            
            # Basic support/resistance
            df_basic['sup_1'] = df_basic['low'].rolling(20, min_periods=1).min().shift(1)
            df_basic['res_1'] = df_basic['high'].rolling(20, min_periods=1).max().shift(1)
            
            df_basic = df_basic.dropna()
            logging.info("✅ Basic indicators calculated as fallback")
            return df_basic
            
        except Exception as e:
            logging.error(f"❌ Even basic indicators failed: {e}")
            return None

    def generate_comprehensive_analysis(self, symbol: str, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> Dict:
        """Generate comprehensive technical analysis with robust error handling"""
        if htf_df is None or ltf_df is None or htf_df.empty or ltf_df.empty:
            logging.warning(f"Empty DataFrames provided for {symbol}")
            return None
            
        try:
            # Get the latest data points with bounds checking
            last_htf = htf_df.iloc[-1] if len(htf_df) > 0 else None
            last_ltf = ltf_df.iloc[-1] if len(ltf_df) > 0 else None
            prev_htf = htf_df.iloc[-2] if len(htf_df) > 1 else last_htf
            prev_ltf = ltf_df.iloc[-2] if len(ltf_df) > 1 else last_ltf
            
            if last_htf is None or last_ltf is None:
                return None

            # Multi-timeframe analysis with error handling
            htf_trend = self._analyze_enhanced_trend(last_htf, prev_htf, htf_df)
            ltf_trend = self._analyze_enhanced_trend(last_ltf, prev_ltf, ltf_df)
            
            # Momentum analysis
            momentum = self._analyze_momentum(last_ltf, prev_ltf)
            
            # Key levels with dynamic calculation
            key_levels = self._calculate_dynamic_levels(htf_df, ltf_df, last_ltf['close'])
            
            # Market structure
            market_structure = self._analyze_market_structure(htf_df, ltf_df)
            
            # Volume analysis (handle missing volume)
            volume_analysis = self._analyze_volume(ltf_df)
            
            # Risk assessment
            risk_assessment = self._assess_risk(htf_df, ltf_df)
            
            # ML-based signal strength
            ml_signal = self._calculate_ml_signal(htf_df, ltf_df)
            
            return {
                'symbol': symbol,
                'htf_trend': htf_trend,
                'ltf_trend': ltf_trend,
                'momentum': momentum,
                'key_levels': key_levels,
                'market_structure': market_structure,
                'volume_analysis': volume_analysis,
                'risk_assessment': risk_assessment,
                'ml_signal': ml_signal,
                'volatility': last_ltf.get('ATRr_14', 0.001),
                'current_price': last_ltf['close'],
                'timestamp': datetime.now(UTC).isoformat()
            }
            
        except Exception as e:
            logging.error(f"❌ Error generating technical analysis for {symbol}: {e}")
            # Return basic analysis as fallback
            return self._generate_basic_analysis(symbol, htf_df, ltf_df)

    def _calculate_ml_signal(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> Dict:
        """Calculate machine learning based signal strength"""
        try:
            if ltf_df.empty or len(ltf_df) < 50:
                return {"signal_strength": 0, "confidence": 0, "features": {}}
            
            # Feature engineering for ML signal
            features = {}
            
            # Trend strength features
            if 'ADX_14' in ltf_df.columns:
                features['adx_strength'] = ltf_df['ADX_14'].iloc[-1] / 100.0
            else:
                features['adx_strength'] = 0
                
            # Momentum convergence
            rsi = ltf_df.get('RSI_14', 50)
            macd_hist = ltf_df.get('MACDh_12_26_9', 0)
            stoch_k = ltf_df.get('STOCHk_14_3_3', 50)
            
            # Normalize features
            rsi_signal = abs(rsi.iloc[-1] - 50) / 50.0 if isinstance(rsi, pd.Series) else 0
            macd_signal = abs(macd_hist.iloc[-1]) if isinstance(macd_hist, pd.Series) else 0
            stoch_signal = abs(stoch_k.iloc[-1] - 50) / 50.0 if isinstance(stoch_k, pd.Series) else 0
            
            features['momentum_convergence'] = (rsi_signal + macd_signal + stoch_signal) / 3.0
            
            # Volatility adjusted signal
            volatility = ltf_df['close'].pct_change().std() * 100 if len(ltf_df) > 1 else 1
            features['volatility_factor'] = min(volatility / 2.0, 1.0)  # Normalize to 0-1
            
            # Price position features
            if 'BBU_20_2.0' in ltf_df.columns and 'BBL_20_2.0' in ltf_df.columns:
                bb_upper = ltf_df['BBU_20_2.0'].iloc[-1]
                bb_lower = ltf_df['BBL_20_2.0'].iloc[-1]
                current_price = ltf_df['close'].iloc[-1]
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
                features['bb_position'] = abs(bb_position - 0.5) * 2  # Convert to 0-1 scale
            else:
                features['bb_position'] = 0
                
            # Calculate composite signal strength
            signal_strength = (
                features['adx_strength'] * 0.3 +
                features['momentum_convergence'] * 0.4 +
                features['bb_position'] * 0.2 +
                features['volatility_factor'] * 0.1
            )
            
            # Confidence based on data quality
            confidence = min(len(ltf_df) / 200.0, 1.0)  # Higher confidence with more data
            
            return {
                "signal_strength": round(signal_strength, 3),
                "confidence": round(confidence, 3),
                "features": features
            }
            
        except Exception as e:
            logging.warning(f"ML signal calculation error: {e}")
            return {"signal_strength": 0, "confidence": 0, "features": {}}

    def _generate_basic_analysis(self, symbol: str, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> Dict:
        """Fallback basic analysis"""
        try:
            last_ltf = ltf_df.iloc[-1]
            current_price = last_ltf['close']
            
            return {
                'symbol': symbol,
                'htf_trend': {'direction': 'NEUTRAL', 'strength': 'UNKNOWN', 'adx': 0},
                'ltf_trend': {'direction': 'NEUTRAL', 'strength': 'UNKNOWN', 'adx': 0},
                'momentum': {
                    'rsi': {'value': 50, 'signal': 'NEUTRAL'},
                    'macd': {'signal': 'NEUTRAL', 'histogram': 0},
                    'stochastic': {'k': 50, 'd': 50, 'signal': 'NEUTRAL'}
                },
                'key_levels': {
                    'support_1': current_price * 0.99,
                    'resistance_1': current_price * 1.01,
                    'support_2': current_price * 0.98,
                    'resistance_2': current_price * 1.02
                },
                'market_structure': {'higher_timeframe_structure': 'UNKNOWN'},
                'volume_analysis': {'signal': 'NO_DATA'},
                'risk_assessment': {'risk_level': 'MEDIUM'},
                'ml_signal': {'signal_strength': 0, 'confidence': 0, 'features': {}},
                'volatility': 0.001,
                'current_price': current_price,
                'timestamp': datetime.now(UTC).isoformat()
            }
        except Exception as e:
            logging.error(f"❌ Even basic analysis failed for {symbol}: {e}")
            return None

    def _analyze_enhanced_trend(self, current: pd.Series, previous: pd.Series, df: pd.DataFrame) -> Dict:
        """Enhanced trend analysis with multiple confirmations"""
        try:
            # EMA analysis with fallbacks
            ema_8 = current.get('EMA_8', current['close'])
            ema_21 = current.get('EMA_21', current['close'])
            ema_50 = current.get('EMA_50', current['close'])
            ema_200 = current.get('EMA_200', current['close'])
            
            # EMA alignment score
            ema_alignment = 0
            if ema_8 > ema_21 > ema_50 > ema_200:
                trend_direction = "STRONG_BULLISH"
                ema_alignment = 4
            elif ema_8 < ema_21 < ema_50 < ema_200:
                trend_direction = "STRONG_BEARISH"
                ema_alignment = 4
            elif ema_8 > ema_21 and ema_21 > ema_50:
                trend_direction = "BULLISH"
                ema_alignment = 3
            elif ema_8 < ema_21 and ema_21 < ema_50:
                trend_direction = "BEARISH"
                ema_alignment = 3
            else:
                trend_direction = "NEUTRAL"
                ema_alignment = 1

            # ADX strength with fallback
            adx = current.get('ADX_14', 0)
            if adx > 40:
                trend_strength = "VERY_STRONG"
            elif adx > 25:
                trend_strength = "STRONG"
            elif adx > 20:
                trend_strength = "MODERATE"
            else:
                trend_strength = "WEAK"

            # Ichimoku analysis
            ichimoku_signal = self._analyze_ichimoku(current)
            
            # SuperTrend signal
            supertrend_signal = "BULLISH" if current.get('SUPERT_7_3.0', '') == 'up' else "BEARISH"

            return {
                'direction': trend_direction,
                'strength': trend_strength,
                'adx': adx,
                'ema_alignment': ema_alignment,
                'ichimoku_signal': ichimoku_signal,
                'supertrend_signal': supertrend_signal,
                'price_above_ema200': current['close'] > ema_200
            }
            
        except Exception as e:
            logging.warning(f"Trend analysis error: {e}")
            return {
                'direction': 'NEUTRAL',
                'strength': 'UNKNOWN',
                'adx': 0,
                'ema_alignment': 0,
                'ichimoku_signal': 'NEUTRAL',
                'supertrend_signal': 'NEUTRAL',
                'price_above_ema200': False
            }

    def _analyze_ichimoku(self, data: pd.Series) -> str:
        """Analyze Ichimoku Cloud signals with error handling"""
        try:
            tenkan = data.get('ISA_9', data['close'])
            kijun = data.get('ISB_26', data['close'])
            senkou_a = data.get('ICS_26', data['close'])
            senkou_b = data.get('ICB_26', data['close'])
            chikou = data.get('ITS_9', data['close'])
            price = data['close']
            
            # Cloud analysis
            cloud_bullish = senkou_a > senkou_b
            
            # Signal generation
            if price > max(senkou_a, senkou_b) and tenkan > kijun and chikou > price:
                return "STRONG_BULLISH"
            elif price < min(senkou_a, senkou_b) and tenkan < kijun and chikou < price:
                return "STRONG_BEARISH"
            elif price > max(senkou_a, senkou_b):
                return "BULLISH"
            elif price < min(senkou_a, senkou_b):
                return "BEARISH"
            else:
                return "NEUTRAL"
        except:
            return "NEUTRAL"

    def _analyze_momentum(self, current: pd.Series, previous: pd.Series) -> Dict:
        """Comprehensive momentum analysis with error handling"""
        try:
            rsi = current.get('RSI_14', 50)
            macd = current.get('MACD_12_26_9', 0)
            macd_signal = current.get('MACDs_12_26_9', 0)
            macd_hist = current.get('MACDh_12_26_9', 0)
            stoch_k = current.get('STOCHk_14_3_3', 50)
            stoch_d = current.get('STOCHd_14_3_3', 50)
            cci = current.get('CCI_20_0.015', 0)
            williams = current.get('WILLR_14', -50)
            momentum_val = current.get('MOM_10', 0)

            # RSI analysis
            if rsi > 70:
                rsi_signal = "OVERBOUGHT"
            elif rsi < 30:
                rsi_signal = "OVERSOLD"
            else:
                rsi_signal = "NEUTRAL"

            # MACD analysis
            macd_trend = "BULLISH" if macd_hist > 0 else "BEARISH"
            prev_macd = previous.get('MACD_12_26_9', 0)
            prev_macd_signal = previous.get('MACDs_12_26_9', 0)
            macd_cross = "BULLISH_CROSS" if macd > macd_signal and prev_macd <= prev_macd_signal else "BEARISH_CROSS" if macd < macd_signal and prev_macd >= prev_macd_signal else "NO_CROSS"

            # Stochastic analysis
            stoch_signal = "OVERBOUGHT" if stoch_k > 80 else "OVERSOLD" if stoch_k < 20 else "NEUTRAL"

            # CCI analysis
            cci_signal = "OVERBOUGHT" if cci > 100 else "OVERSOLD" if cci < -100 else "NEUTRAL"

            # Williams %R
            williams_signal = "OVERBOUGHT" if williams > -20 else "OVERSOLD" if williams < -80 else "NEUTRAL"

            # Momentum convergence score
            bullish_signals = 0
            bearish_signals = 0
            
            if rsi_signal == "OVERSOLD":
                bullish_signals += 1
            if rsi_signal == "OVERBOUGHT":
                bearish_signals += 1
                
            if macd_trend == "BULLISH":
                bullish_signals += 1
            if macd_trend == "BEARISH":
                bearish_signals += 1
                
            if stoch_signal == "OVERSOLD":
                bullish_signals += 1
            if stoch_signal == "OVERBOUGHT":
                bearish_signals += 1
                
            momentum_score = bullish_signals - bearish_signals

            return {
                'rsi': {'value': rsi, 'signal': rsi_signal},
                'macd': {'trend': macd_trend, 'cross': macd_cross, 'histogram': macd_hist},
                'stochastic': {'k': stoch_k, 'd': stoch_d, 'signal': stoch_signal},
                'cci': {'value': cci, 'signal': cci_signal},
                'williams': {'value': williams, 'signal': williams_signal},
                'momentum': {'value': momentum_val, 'signal': "BULLISH" if momentum_val > 0 else "BEARISH"},
                'convergence_score': momentum_score,
                'overall_bias': "BULLISH" if momentum_score > 1 else "BEARISH" if momentum_score < -1 else "NEUTRAL"
            }
            
        except Exception as e:
            logging.warning(f"Momentum analysis error: {e}")
            return {
                'rsi': {'value': 50, 'signal': 'NEUTRAL'},
                'macd': {'trend': 'NEUTRAL', 'cross': 'NO_CROSS', 'histogram': 0},
                'stochastic': {'k': 50, 'd': 50, 'signal': 'NEUTRAL'},
                'cci': {'value': 0, 'signal': 'NEUTRAL'},
                'williams': {'value': -50, 'signal': 'NEUTRAL'},
                'momentum': {'value': 0, 'signal': 'NEUTRAL'},
                'convergence_score': 0,
                'overall_bias': "NEUTRAL"
            }

    def _calculate_dynamic_levels(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame, current_price: float) -> Dict:
        """Calculate dynamic support and resistance levels with error handling"""
        try:
            # Recent highs and lows with error handling
            recent_high_20 = ltf_df['high'].tail(20).max() if len(ltf_df) >= 20 else current_price * 1.02
            recent_low_20 = ltf_df['low'].tail(20).min() if len(ltf_df) >= 20 else current_price * 0.98
            recent_high_50 = ltf_df['high'].tail(50).max() if len(ltf_df) >= 50 else current_price * 1.03
            recent_low_50 = ltf_df['low'].tail(50).min() if len(ltf_df) >= 50 else current_price * 0.97

            # Pivot points
            pivot = ltf_df.get('pivot', pd.Series([current_price])).iloc[-1]
            r1 = ltf_df.get('r1', pd.Series([current_price * 1.01])).iloc[-1]
            s1 = ltf_df.get('s1', pd.Series([current_price * 0.99])).iloc[-1]

            # Bollinger Bands
            bb_upper = ltf_df.get('BBU_20_2.0', pd.Series([current_price * 1.02])).iloc[-1]
            bb_lower = ltf_df.get('BBL_20_2.0', pd.Series([current_price * 0.98])).iloc[-1]

            # Fibonacci levels
            range_high = max(recent_high_20, recent_high_50)
            range_low = min(recent_low_20, recent_low_50)
            fib_range = range_high - range_low
            fib_382 = range_high - 0.382 * fib_range
            fib_618 = range_high - 0.618 * fib_range

            # Determine key levels based on proximity
            levels = [recent_high_20, recent_low_20, recent_high_50, recent_low_50, pivot, r1, s1, bb_upper, bb_lower, fib_382, fib_618]
            
            # Find nearest support and resistance
            supports = [level for level in levels if level < current_price]
            resistances = [level for level in levels if level > current_price]
            
            nearest_support = max(supports) if supports else current_price * 0.99
            nearest_resistance = min(resistances) if resistances else current_price * 1.01

            return {
                'support_1': nearest_support,
                'resistance_1': nearest_resistance,
                'support_2': min(supports) if len(supports) > 1 else nearest_support * 0.995,
                'resistance_2': max(resistances) if len(resistances) > 1 else nearest_resistance * 1.005,
                'pivot': pivot,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'fib_382': fib_382,
                'fib_618': fib_618
            }
            
        except Exception as e:
            logging.warning(f"Dynamic levels calculation error: {e}")
            # Fallback levels
            return {
                'support_1': current_price * 0.99,
                'resistance_1': current_price * 1.01,
                'support_2': current_price * 0.98,
                'resistance_2': current_price * 1.02,
                'pivot': current_price,
                'bb_upper': current_price * 1.02,
                'bb_lower': current_price * 0.98,
                'fib_382': current_price * 0.994,
                'fib_618': current_price * 0.988
            }

    def _analyze_market_structure(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> Dict:
        """Analyze market structure for higher timeframe context"""
        try:
            if len(htf_df) < 5 or len(ltf_df) < 10:
                return {'higher_timeframe_structure': 'INSUFFICIENT_DATA', 'is_breaking_structure': False, 'market_phase': 'UNKNOWN'}

            # Higher Highs/Higher Lows analysis
            htf_highs = htf_df['high'].tail(5)
            htf_lows = htf_df['low'].tail(5)
            
            htf_hh = all(htf_highs.iloc[i] > htf_highs.iloc[i-1] for i in range(1, len(htf_highs)))
            htf_ll = all(htf_lows.iloc[i] > htf_lows.iloc[i-1] for i in range(1, len(htf_lows)))
            htf_lh = all(htf_highs.iloc[i] < htf_highs.iloc[i-1] for i in range(1, len(htf_highs)))
            htf_hl = all(htf_lows.iloc[i] < htf_lows.iloc[i-1] for i in range(1, len(htf_lows)))

            if htf_hh and htf_ll:
                structure = "UPTREND"
            elif htf_lh and htf_hl:
                structure = "DOWNTREND"
            else:
                structure = "RANGING"

            return {
                'higher_timeframe_structure': structure,
                'is_breaking_structure': self._check_structure_break(htf_df, ltf_df),
                'market_phase': self._determine_market_phase(htf_df)
            }
            
        except Exception as e:
            logging.warning(f"Market structure analysis error: {e}")
            return {'higher_timeframe_structure': 'UNKNOWN', 'is_breaking_structure': False, 'market_phase': 'UNKNOWN'}

    def _check_structure_break(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> bool:
        """Check if market is breaking structure"""
        try:
            if len(htf_df) < 5 or len(ltf_df) < 10:
                return False
                
            recent_htf_high = htf_df['high'].iloc[-1]
            recent_htf_low = htf_df['low'].iloc[-1]
            ltf_high = ltf_df['high'].tail(5).max()
            ltf_low = ltf_df['low'].tail(5).min()
            
            return ltf_high > recent_htf_high or ltf_low < recent_htf_low
        except:
            return False

    def _determine_market_phase(self, df: pd.DataFrame) -> str:
        """Determine market phase"""
        try:
            if len(df) < 50:
                return "UNKNOWN"
                
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-50]) / df['close'].iloc[-50] * 100
            volatility = df['close'].pct_change().std() * 100
            
            if abs(price_change) < 2 and volatility < 1:
                return "ACCUMULATION"
            elif price_change > 5 and volatility > 1.5:
                return "MARKUP"
            elif abs(price_change) < 3 and volatility > 2:
                return "DISTRIBUTION"
            elif price_change < -5 and volatility > 1.5:
                return "MARKDOWN"
            else:
                return "TRANSITION"
        except:
            return "UNKNOWN"

    def _analyze_volume(self, df: pd.DataFrame) -> Dict:
        """Analyze volume characteristics with handling for missing volume"""
        try:
            if 'volume' not in df.columns or df['volume'].isna().all():
                return {'signal': 'NO_VOLUME_DATA', 'trend': 'UNKNOWN', 'volume_vs_average': 1}
                
            volume_trend = "INCREASING" if df['volume'].iloc[-1] > df['volume'].tail(20).mean() else "DECREASING"
            
            # Handle OBV if available
            obv_trend = "NEUTRAL"
            if 'OBV' in df.columns:
                obv_trend = "BULLISH" if df['OBV'].iloc[-1] > df['OBV'].iloc[-5] else "BEARISH"

            return {
                'volume_trend': volume_trend,
                'obv_signal': obv_trend,
                'volume_vs_average': df['volume'].iloc[-1] / df['volume'].tail(20).mean() if df['volume'].tail(20).mean() > 0 else 1
            }
            
        except Exception as e:
            logging.warning(f"Volume analysis error: {e}")
            return {'signal': 'ERROR', 'trend': 'UNKNOWN', 'volume_vs_average': 1}

    def _assess_risk(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> Dict:
        """Assess market risk conditions"""
        try:
            ltf_volatility = ltf_df['close'].pct_change().std() * 100
            atr = ltf_df.get('ATRr_14', pd.Series([0])).iloc[-1]
            current_range = (ltf_df['high'].iloc[-1] - ltf_df['low'].iloc[-1]) / ltf_df['close'].iloc[-1] * 100
            
            if ltf_volatility > 2 or current_range > 1.5:
                risk_level = "HIGH"
            elif ltf_volatility > 1 or current_range > 0.8:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            return {
                'risk_level': risk_level,
                'volatility_percent': ltf_volatility,
                'atr_value': atr,
                'current_range_percent': current_range
            }
            
        except Exception as e:
            logging.warning(f"Risk assessment error: {e}")
            return {'risk_level': 'MEDIUM', 'volatility_percent': 0, 'atr_value': 0, 'current_range_percent': 0}

# =================================================================================
# --- Smart API Manager with Enhanced Model Selection ---
# =================================================================================

class SmartAPIManager:
    def __init__(self, usage_file: str):
        self.usage_file = usage_file
        self.usage_data = self.load_usage_data()
        self.available_models = self.initialize_available_models()
        self.failed_models = set()

    def load_usage_data(self) -> Dict:
        """Load API usage data"""
        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return self.check_and_reset_daily_usage(data)
            return self.initialize_usage_data()
        except Exception as e:
            logging.error(f"Error loading API usage data: {e}")
            return self.initialize_usage_data()

    def initialize_usage_data(self) -> Dict:
        """Initialize usage data"""
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
        """Check and reset daily usage"""
        today = datetime.now(UTC).date().isoformat()
        last_reset = data.get("last_reset_date", "")
        if last_reset != today:
            for provider in data["providers"]:
                data["providers"][provider]["used_today"] = 0
            data["last_reset_date"] = today
            self.save_usage_data(data)
            logging.info("✅ Daily API usage reset")
        return data

    def save_usage_data(self, data: Dict = None):
        """Save usage data"""
        if data is None:
            data = self.usage_data
        try:
            with open(self.usage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error saving API usage data: {e}")

    def initialize_available_models(self) -> Dict:
        """Initialize available models"""
        return {
            "google_gemini": [GEMINI_MODEL],
            "cloudflare": CLOUDFLARE_MODELS.copy(),
            "groq": GROQ_MODELS.copy()
        }

    def can_use_provider(self, provider: str) -> bool:
        """Check if provider can be used"""
        if provider not in self.usage_data["providers"]:
            return False
        provider_data = self.usage_data["providers"][provider]
        remaining = provider_data["limit"] - provider_data["used_today"]
        return remaining > 0

    def get_available_models_count(self, provider: str) -> int:
        """Get available models count for provider"""
        if not self.can_use_provider(provider):
            return 0
        provider_data = self.usage_data["providers"][provider]
        remaining = provider_data["limit"] - provider_data["used_today"]
        available_models = len(self.available_models[provider])
        return min(remaining, available_models)

    def mark_model_failed(self, provider: str, model_name: str):
        """Mark model as failed"""
        self.failed_models.add((provider, model_name))
        logging.warning(f"❌ Model {provider}/{model_name} added to failed list")

    def is_model_failed(self, provider: str, model_name: str) -> bool:
        """Check if model failed"""
        return (provider, model_name) in self.failed_models

    def _find_model_provider(self, model_name: str) -> Optional[str]:
        """Find which provider a model belongs to"""
        for provider, models in self.available_models.items():
            if model_name in models:
                return provider
        return None

    def select_diverse_models(self, target_total: int = 5, min_required: int = 3) -> List[Tuple[str, str]]:
        """Select diverse models from different providers with intelligent fallback"""
        selected_models = []
        
        # Calculate provider capacity
        provider_capacity = {}
        for provider in ["google_gemini", "cloudflare", "groq"]:
            provider_capacity[provider] = self.get_available_models_count(provider)
            
        logging.info(f"📊 Provider capacity: Gemini={provider_capacity['google_gemini']}, "
                   f"Cloudflare={provider_capacity['cloudflare']}, Groq={provider_capacity['groq']}")

        # Strategy: prioritize diversity across providers
        total_available = sum(provider_capacity.values())
        if total_available == 0:
            logging.error("❌ No providers available")
            return selected_models

        # Model family mapping for intelligent fallback
        model_families = {
            "llama": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", 
                     "@cf/meta/llama-4-scout-17b-16e-instruct", "@cf/meta/llama-3.3-70b-instruct-fp8-fast"],
            "qwen": ["qwen/qwen3-32b"],
            "gemma": ["@cf/google/gemma-3-12b-it"],
            "mistral": ["@cf/mistralai/mistral-small-3.1-24b-instruct"],
            "gemini": ["gemini-2.5-flash"]
        }
        
        # Track used families to ensure diversity
        used_families = set()
        
        # First pass: select one model from each family
        for family, models in model_families.items():
            if len(selected_models) >= target_total:
                break
            
            for model in models:
                # Find which provider this model belongs to
                provider = self._find_model_provider(model)
                if (provider and provider_capacity[provider] > 0 and 
                    not self.is_model_failed(provider, model) and
                    family not in used_families):
                    
                    selected_models.append((provider, model))
                    provider_capacity[provider] -= 1
                    used_families.add(family)
                    logging.info(f"🎯 Selected {provider}/{model} (family: {family})")
                    break

        # Second pass: fill remaining slots with any available models
        providers_order = ["google_gemini", "cloudflare", "groq"]
        round_robin_index = 0
        remaining_target = target_total - len(selected_models)
        
        while remaining_target > 0 and any(provider_capacity[p] > 0 for p in providers_order):
            current_provider = providers_order[round_robin_index % len(providers_order)]
            
            if provider_capacity[current_provider] > 0:
                # Select first available model from this provider that hasn't failed and isn't already selected
                for model_name in self.available_models[current_provider]:
                    if ((current_provider, model_name) not in selected_models and 
                        not self.is_model_failed(current_provider, model_name)):
                        selected_models.append((current_provider, model_name))
                        provider_capacity[current_provider] -= 1
                        remaining_target -= 1
                        logging.info(f"🎯 Added {current_provider}/{model_name} as fallback")
                        break
                    
            round_robin_index += 1
            
            # Safety break
            if round_robin_index > len(providers_order) * 3:
                break

        # Final fallback: if minimum not reached, use any available model even if failed before
        if len(selected_models) < min_required:
            logging.warning(f"⚠️ Only {len(selected_models)} models selected. Activating emergency fallback...")
            for provider in providers_order:
                if self.can_use_provider(provider):
                    for model_name in self.available_models[provider]:
                        if (provider, model_name) not in selected_models:
                            selected_models.append((provider, model_name))
                            logging.info(f"🚨 Emergency fallback: {provider}/{model_name}")
                            if len(selected_models) >= min_required:
                                break
                    if len(selected_models) >= min_required:
                        break

        logging.info(f"🎯 {len(selected_models)} diverse models selected: {selected_models}")
        return selected_models

    def record_api_usage(self, provider: str, count: int = 1):
        """Record API usage"""
        if provider in self.usage_data["providers"]:
            self.usage_data["providers"][provider]["used_today"] += count
            self.save_usage_data()

    def get_usage_summary(self) -> str:
        """Get usage summary"""
        summary = "📊 API Usage Summary:\n"
        for provider, data in self.usage_data["providers"].items():
            remaining = data["limit"] - data["used_today"]
            summary += f"  {provider}: {data['used_today']}/{data['limit']} ({remaining} remaining)\n"
        return summary

# =================================================================================
# --- Enhanced AI Manager with Fixed Error Handling + Token-Aware Gemini + Short-Prompt ---
# =================================================================================

class EnhancedAIManager:
    def __init__(self, gemini_api_key: str, cloudflare_api_key: str, groq_api_key: str, api_manager):
        self.gemini_api_key = gemini_api_key
        self.cloudflare_api_key = cloudflare_api_key
        self.groq_api_key = groq_api_key
        self.api_manager = api_manager
        
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)

    # -----------------------------
    # 🔧 Helper: Gemini model caps
    # -----------------------------
    def _gemini_model_caps(self, model_name: str) -> Dict[str, int]:
        """
        بر اساس نام مدل سقف تقریبی توکن کانتکست رو برمی‌گردونه.
        اگر مدل ناشناخته بود، 8192 رو فرض می‌کنیم.
        """
        name = (model_name or "").lower()
        if "2.5-pro" in name or "2.0-pro" in name:
            return {"context": 32768}
        if "2.5-flash" in name or "flash" in name:
            return {"context": 16384}
        if "1.5-pro" in name or "1.5-flash" in name:
            return {"context": 8192}
        return {"context": 8192}

    # -------------------------------------
    # 🔧 Helper: calc safe max_output_tokens
    # -------------------------------------
    def _calc_gemini_max_tokens(self, model, model_name: str, prompt: str, min_gen: int = 384, pad: int = 256) -> int:
        """
        تعداد توکن‌های پرامپت را می‌شمارد و بر اساس سقف مدل، max_output_tokens امن می‌سازد.
        - min_gen: حداقل خروجی که می‌خواهیم
        - pad: حاشیهٔ امن برای نوسانات شمارش
        """
        try:
            caps = self._gemini_model_caps(model_name)
            ctx = caps.get("context", 8192)
            t = model.count_tokens(prompt)
            prompt_tokens = 0
            if hasattr(t, "total_tokens"):
                prompt_tokens = int(t.total_tokens or 0)
            elif isinstance(t, dict):
                prompt_tokens = int(t.get("total_tokens", 0))
            gen = max(min_gen, ctx - prompt_tokens - pad)
            gen = max(256, min(gen, 1200))
            return gen
        except Exception:
            return 700

    # ---------------------------------
    # 🔧 Helper: make a short prompt
    # ---------------------------------
    def _shorten_prompt_for_gemini(self, prompt: str) -> str:
        """
        یک پرامپت کوتاه‌تر و فشرده از پرامپت اصلی می‌سازد تا احتمال finish_reason=2 کم شود.
        فقط اطلاعات کلیدی را نگه می‌داریم.
        """
        try:
            keep_keys = {
                "SYMBOL", "CURRENT PRICE", "TECHNICAL ANALYSIS SUMMARY",
                "CALCULATION INSTRUCTIONS", "RETURN ONLY THIS EXACT JSON FORMAT", "CRITICAL"
            }
            lines = [ln for ln in prompt.splitlines() if any(k in ln for k in keep_keys)]
            if len(lines) < 12:
                # اگر خیلی کوتاه شد، 80 خط اول را نگه دار
                lines = prompt.splitlines()[:80]
            return "\n".join(lines)
        except Exception:
            return prompt

    def _extract_gemini_text(self, resp) -> Optional[str]:
        """
        Safely extract text from a Gemini response بدون دسترسی مستقیم و کورکورانه به resp.text.
        Order:
          1) resp.to_dict() -> candidates[...].content.parts[].text
          2) resp.candidates -> content.parts[].text (SDK objects)
          3) (no blind resp.text)
        """
        # 1) to_dict
        try:
            d = resp.to_dict()
            cands = d.get("candidates") or []
            for c in cands:
                parts = ((c.get("content") or {}).get("parts")) or []
                for p in parts:
                    t = p.get("text")
                    if t:
                        return t
        except Exception:
            pass

        # 2) SDK attrs
        try:
            if getattr(resp, "candidates", None):
                for c in resp.candidates:
                    if getattr(c, "content", None) and getattr(c.content, "parts", None):
                        for p in c.content.parts:
                            if getattr(p, "text", None):
                                return p.text
        except Exception:
            pass

        # 3) nothing
        return None

    def _create_enhanced_english_prompt(self, symbol: str, technical_analysis: Dict) -> str:
        """Create enhanced English prompt for AI analysis"""
        current_price = technical_analysis.get('current_price', 1.0850)
        
        momentum_data = technical_analysis.get('momentum', {})
        stochastic_data = momentum_data.get('stochastic', {})
        htf_trend = technical_analysis.get('htf_trend', {})
        ltf_trend = technical_analysis.get('ltf_trend', {})
        key_levels = technical_analysis.get('key_levels', {})
        market_structure = technical_analysis.get('market_structure', {})
        risk_assessment = technical_analysis.get('risk_assessment', {})
        ml_signal = technical_analysis.get('ml_signal', {})
        
        return f"""IMPORTANT: You are a professional forex trading analyst. Analyze the technical setup and provide ONLY a valid JSON response.

SYMBOL: {symbol}
CURRENT PRICE: {current_price:.5f}

TECHNICAL ANALYSIS SUMMARY:
- HTF Trend (4H): {htf_trend.get('direction', 'NEUTRAL')} | Strength: {htf_trend.get('strength', 'UNKNOWN')} | ADX: {htf_trend.get('adx', 0):.1f}
- LTF Trend (1H): {ltf_trend.get('direction', 'NEUTRAL')} | EMA Alignment: {htf_trend.get('ema_alignment', 0)}/4
- Momentum Bias: {momentum_data.get('overall_bias', 'NEUTRAL')} | RSI: {momentum_data.get('rsi', {}).get('value', 50):.1f} ({momentum_data.get('rsi', {}).get('signal', 'NEUTRAL')})
- MACD: {momentum_data.get('macd', {}).get('trend', 'NEUTRAL')} | Signal: {momentum_data.get('macd', {}).get('cross', 'NO_CROSS')}
- Stochastic: {stochastic_data.get('k', 50):.1f} ({stochastic_data.get('signal', 'NEUTRAL')})
- ML Signal Strength: {ml_signal.get('signal_strength', 0):.3f} | Confidence: {ml_signal.get('confidence', 0):.3f}
- Key Support: {key_levels.get('support_1', current_price * 0.99):.5f} | Key Resistance: {key_levels.get('resistance_1', current_price * 1.01):.5f}
- Market Structure: {market_structure.get('higher_timeframe_structure', 'UNKNOWN')}
- Risk Level: {risk_assessment.get('risk_level', 'MEDIUM')}
- Volatility: {technical_analysis.get('volatility', 0.001):.5f} (ATR)
- Market Phase: {market_structure.get('market_phase', 'UNKNOWN')}

CALCULATION INSTRUCTIONS:
- Calculate realistic levels based on current price {current_price:.5f} and technical structure
- Use ATR ({risk_assessment.get('atr_value', 0.001):.5f}) for stop loss calculation
- For entry: Use single price, not range
- For stop loss: Calculate based on 1.5x ATR or key levels
- For take profit: Use risk-reward ratio 1.5-2.0
- Consider market structure and phase in your analysis
- Factor in ML signal strength and confidence

RETURN ONLY THIS EXACT JSON FORMAT:
{{
  "SYMBOL": "{symbol}",
  "ACTION": "BUY",
  "CONFIDENCE": 7,
  "ENTRY": "{current_price:.5f}",
  "STOP_LOSS": "{current_price - 0.0020:.5f}",
  "TAKE_PROFIT": "{current_price + 0.0030:.5f}",
  "RISK_REWARD_RATIO": "1.5",
  "ANALYSIS": "Technical analysis based on bullish trend alignment and positive momentum convergence",
  "EXPIRATION_H": 4,
  "TRADE_RATIONALE": "Signal based on EMA alignment, RSI momentum, and key level breakout potential"
}}

CRITICAL:
- ACTION must be "BUY", "SELL", or "HOLD"
- CONFIDENCE must be between 1-10
- All price levels must be single values, not ranges
- Provide realistic levels based on technical context
- Return ONLY the JSON object, no other text
"""

    async def get_enhanced_ai_analysis(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        """Get enhanced AI analysis with multiple models and robust error handling"""
        selected_models = self.api_manager.select_diverse_models(target_total=5, min_required=3)
        
        if len(selected_models) < 3:
            logging.error(f"❌ Cannot find minimum 3 AI models for {symbol}")
            return None
            
        logging.info(f"🎯 Using {len(selected_models)} AI models for {symbol}")

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
                    logging.error(f"❌ Error in {provider}/{model_name} for {symbol}: {str(result)}")
                    logging.error(f"❌ Traceback: {traceback.format_exc()}")
                    self.api_manager.mark_model_failed(provider, model_name)
                    failed_count += 1
                    self.api_manager.record_api_usage(provider)
                elif result is not None:
                    valid_results.append(result)
                    self.api_manager.record_api_usage(provider)
                else:
                    self.api_manager.record_api_usage(provider)

            logging.info(f"📊 Results: {len(valid_results)} successful, {failed_count} failed")
            
            if valid_results:
                return self._combine_signals(symbol, valid_results, len(selected_models))
            else:
                logging.warning(f"⚠️ No valid AI results for {symbol}")
                return None
                
        except Exception as e:
            logging.error(f"❌ Error in AI analysis for {symbol}: {str(e)}")
            logging.error(f"❌ Traceback: {traceback.format_exc()}")
            return None

    async def _get_single_analysis(self, symbol: str, technical_analysis: Dict, provider: str, model_name: str) -> Optional[Dict]:
        """Get analysis from single AI model"""
        try:
            prompt = self._create_enhanced_english_prompt(symbol, technical_analysis)
            
            if provider == "google_gemini":
                return await self._get_gemini_analysis(symbol, prompt, model_name)
            elif provider == "cloudflare":
                return await self._get_cloudflare_analysis(symbol, prompt, model_name)
            elif provider == "groq":
                return await self._get_groq_analysis(symbol, prompt, model_name)
            else:
                return None
                
        except Exception as e:
            logging.error(f"❌ Error in {provider}/{model_name} for {symbol}: {str(e)}")
            logging.error(f"❌ Traceback: {traceback.format_exc()}")
            return None

    # ---------------------------------------------------------
    # ✅ Gemini with token-aware + schema + MIME + plain fallbacks
    # ---------------------------------------------------------
    async def _get_gemini_analysis(self, symbol: str, prompt: str, model_name: str) -> Optional[Dict]:
        """
        Gemini with:
          1) token-aware max_output_tokens
          2) response_schema for structured output (افزایش شانس داشتن Part)
          3) fallback: MIME=application/json بدون schema
          4) fallback نهایی: بدون MIME (plain text) + short prompt
        """
        try:
            model = genai.GenerativeModel(
                model_name,
                system_instruction="You are a forex trading expert. Return ONLY valid JSON. No prose."
            )

            # 1) محاسبهٔ خروجی امن
            max_out = self._calc_gemini_max_tokens(model, model_name, prompt, min_gen=384, pad=256)

            # 2) تلاش اول: Structured Output (schema)
            response_schema = {
                "type": "object",
                "properties": {
                    "SYMBOL": {"type": "string"},
                    "ACTION": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                    "CONFIDENCE": {"type": "number"},
                    "ENTRY": {"type": "string"},
                    "STOP_LOSS": {"type": "string"},
                    "TAKE_PROFIT": {"type": "string"},
                    "RISK_REWARD_RATIO": {"type": "string"},
                    "ANALYSIS": {"type": "string"},
                    "EXPIRATION_H": {"type": "number"},
                    "TRADE_RATIONALE": {"type": "string"}
                },
                "required": ["SYMBOL", "ACTION", "CONFIDENCE"]
            }

            resp1 = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=max_out,
                    response_mime_type="application/json",
                    response_schema=response_schema
                )
            )

            def _resp_is_empty_or_blocked(rsp) -> bool:
                try:
                    d = rsp.to_dict()
                except Exception:
                    d = {}
                pf = (d or {}).get("prompt_feedback") or {}
                if pf.get("block_reason"):
                    return True
                cands = (d or {}).get("candidates") or []
                if not cands:
                    return True
                parts = ((cands[0].get("content") or {}).get("parts")) or []
                return len(parts) == 0

            raw = self._extract_gemini_text(resp1)
            try:
                if getattr(resp1, "candidates", None):
                    fr = getattr(resp1.candidates[0], "finish_reason", None)
                    if fr is not None and fr != 1:
                        logging.warning(f"⚠️ Gemini finish_reason={fr} for {symbol} (schema attempt)")
            except Exception:
                pass

            if raw and not _resp_is_empty_or_blocked(resp1):
                return self._parse_ai_response(raw, symbol, f"Gemini-{model_name}")

            # 3) تلاش دوم: MIME=application/json بدون schema
            resp2 = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.05,
                    max_output_tokens=max_out,
                    response_mime_type="application/json",
                )
            )
            try:
                if getattr(resp2, "candidates", None):
                    fr = getattr(resp2.candidates[0], "finish_reason", None)
                    if fr is not None and fr != 1:
                        logging.warning(f"⚠️ Gemini finish_reason={fr} for {symbol} (json-mime attempt)")
            except Exception:
                pass

            raw = self._extract_gemini_text(resp2)
            if raw:
                return self._parse_ai_response(raw, symbol, f"Gemini-{model_name}")

            # 4) تلاش سوم: بدون MIME (plain) + پرامپت کوتاه‌تر
            short_prompt = self._shorten_prompt_for_gemini(prompt)
            resp3 = await asyncio.to_thread(
                model.generate_content,
                short_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=max(min(1200, max_out + 200), 500),
                )
            )
            try:
                if getattr(resp3, "candidates", None):
                    fr = getattr(resp3.candidates[0], "finish_reason", None)
                    if fr is not None and fr != 1:
                        logging.warning(f"⚠️ Gemini finish_reason={fr} for {symbol} (plain attempt)")
            except Exception:
                pass

            raw = self._extract_gemini_text(resp3)
            if not raw:
                logging.warning(f"❌ Gemini returned no usable content for {symbol} after all fallbacks")
                return None

            return self._parse_ai_response(raw, symbol, f"Gemini-{model_name}")

        except Exception as e:
            logging.error(f"❌ Gemini analysis error for {symbol}: {str(e)}")
            return None

    async def _get_cloudflare_analysis(self, symbol: str, prompt: str, model_name: str) -> Optional[Dict]:
        """Get analysis from Cloudflare AI with improved response handling"""
        if not self.cloudflare_api_key:
            logging.warning(f"❌ Cloudflare API key not available for {symbol}")
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
            
            account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "default_account_id")
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
                        elif "result" in data and isinstance(data["result"], str):
                            content = data["result"]
                        else:
                            content = data
                            
                        if content:
                            return self._parse_ai_response(content, symbol, f"Cloudflare-{model_name}")
                        else:
                            logging.warning(f"❌ Empty content in Cloudflare response for {symbol}")
                            return None
                    else:
                        error_text = await response.text()
                        logging.error(f"❌ Cloudflare API error for {symbol}: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            logging.error(f"❌ Cloudflare/{model_name} analysis error for {symbol}: {str(e)}")
            logging.error(f"❌ Traceback: {traceback.format_exc()}")
            return None

    async def _get_groq_analysis(self, symbol: str, prompt: str, model_name: str) -> Optional[Dict]:
        """Get analysis from Groq API"""
        if not self.groq_api_key:
            logging.warning(f"❌ Groq API key not available for {symbol}")
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
                        else:
                            logging.warning(f"❌ No choices in Groq response for {symbol}: {data}")
                            return None
                    else:
                        error_text = await response.text()
                        logging.error(f"❌ Groq API error for {symbol}: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            logging.error(f"❌ Groq/{model_name} analysis error for {symbol}: {str(e)}")
            logging.error(f"❌ Traceback: {traceback.format_exc()}")
            return None

    def _parse_ai_response(self, response, symbol: str, ai_name: str) -> Optional[Dict]:
        """Parse AI response with enhanced validation and robust error handling"""
        try:
            if isinstance(response, dict):
                cleaned_response = json.dumps(response, ensure_ascii=False)
            else:
                cleaned_response = (response or "").strip()

                cleaned_response = re.sub(r'```json\s*', '', cleaned_response)
                cleaned_response = re.sub(r'```\s*', '', cleaned_response)

                cleaned_response = re.sub(
                    r'<think>.*?</think>',
                    '',
                    cleaned_response,
                    flags=re.DOTALL | re.IGNORECASE
                )
                if cleaned_response.lstrip().lower().startswith('<think>'):
                    brace_idx = cleaned_response.find('{')
                    if brace_idx != -1:
                        cleaned_response = cleaned_response[brace_idx:]

                cleaned_response = re.sub(r'</?[^>]+>', '', cleaned_response)

                cleaned_response = re.sub(
                    r'^\s*(system|assistant|user|inst|instruction|thought)\s*:\s*.*?(?=\{)',
                    '',
                    cleaned_response,
                    flags=re.IGNORECASE | re.DOTALL
                )

                if '{' in cleaned_response:
                    cleaned_response = cleaned_response[cleaned_response.find('{'):]

            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                signal_data = json.loads(json_str)

                if self._validate_signal_data(signal_data, symbol):
                    signal_data['ai_model'] = ai_name
                    signal_data['timestamp'] = datetime.now(UTC).isoformat()

                    signal_data = self._validate_numeric_values(signal_data, symbol)

                    logging.info(f"✅ {ai_name} signal for {symbol}: {signal_data.get('ACTION', 'HOLD')}")
                    return signal_data

            try:
                if isinstance(response, dict):
                    response_preview = json.dumps(response, ensure_ascii=False)[:200]
                else:
                    response_preview = str(response)[:200] if response else "Empty response"
            except:
                response_preview = "Unable to preview response"

            logging.warning(f"❌ {ai_name} response for {symbol} lacks valid JSON format. Response: {response_preview}...")
            return None

        except json.JSONDecodeError as e:
            try:
                if isinstance(response, dict):
                    response_preview = json.dumps(response, ensure_ascii=False)[:200]
                else:
                    response_preview = str(response)[:200] if response else "Empty response"
            except:
                response_preview = "Unable to preview response"

            logging.error(f"❌ JSON error in {ai_name} response for {symbol}: {e}. Response: {response_preview}...")
            return None
        except Exception as e:
            try:
                if isinstance(response, dict):
                    response_preview = json.dumps(response, ensure_ascii=False)[:200]
                else:
                    response_preview = str(response)[:200] if response else "Empty response"
            except:
                response_preview = "Unable to preview response"

            logging.error(f"❌ Error parsing {ai_name} response for {symbol}: {str(e)}. Response: {response_preview}...")
            return None

    def _validate_signal_data(self, signal_data: Dict, symbol: str) -> bool:
        """Validate signal data"""
        required_fields = ['SYMBOL', 'ACTION', 'CONFIDENCE']
        for field in required_fields:
            if field not in signal_data:
                logging.warning(f"❌ Required field {field} missing in signal for {symbol}")
                return False
                
        action = signal_data['ACTION'].upper()
        if action not in ['BUY', 'SELL', 'HOLD']:
            logging.warning(f"❌ Invalid ACTION for {symbol}: {action}")
            return False
            
        try:
            confidence = float(signal_data['CONFIDENCE'])
            if not (1 <= confidence <= 10):
                logging.warning(f"❌ CONFIDENCE out of range for {symbol}: {confidence}")
                return False
        except (ValueError, TypeError):
            logging.warning(f"❌ Invalid CONFIDENCE for {symbol}: {signal_data['CONFIDENCE']}")
            return False
            
        return True

    def _validate_numeric_values(self, signal_data: Dict, symbol: str) -> Dict:
        """Validate and fix numeric values"""
        numeric_fields = ['ENTRY', 'STOP_LOSS', 'TAKE_PROFIT', 'EXPIRATION_H', 'RISK_REWARD_RATIO']
        
        for field in numeric_fields:
            if field in signal_data:
                value = signal_data[field]
                if value is None or value == "null" or str(value).strip() == "":
                    if field == 'EXPIRATION_H':
                        signal_data[field] = 4
                    elif field == 'RISK_REWARD_RATIO':
                        signal_data[field] = "1.5"
                    else:
                        signal_data[field] = "N/A"
                        
        return signal_data

    def _combine_signals(self, symbol: str, valid_results: List[Dict], total_models: int) -> Optional[Dict]:
        """Combine signal results intelligently"""
        if not valid_results:
            return None
            
        action_counts = {}
        for result in valid_results:
            action = result['ACTION'].upper()
            action_counts[action] = action_counts.get(action, 0) + 1
            
        total_valid = len(valid_results)
        max_agreement = max(action_counts.values())
        
        if max_agreement >= 3:
            agreement_type = 'STRONG_CONSENSUS'
        elif max_agreement == 2:
            agreement_type = 'MEDIUM_CONSENSUS'
        else:
            agreement_type = 'WEAK_CONSENSUS'
            
        majority_action = max(action_counts, key=action_counts.get)
        agreeing_results = [r for r in valid_results if r['ACTION'].upper() == majority_action]
        
        combined = {
            'SYMBOL': symbol,
            'ACTION': majority_action,
            'AGREEMENT_LEVEL': max_agreement,
            'AGREEMENT_TYPE': agreement_type,
            'VALID_MODELS': total_valid,
            'TOTAL_MODELS': total_models,
            'timestamp': datetime.now(UTC).isoformat()
        }
        
        if agreeing_results:
            confidences = [float(r.get('CONFIDENCE', 5)) for r in agreeing_results]
            combined['CONFIDENCE'] = round(sum(confidences) / len(confidences), 1)
            
        if agreeing_results:
            first_valid = agreeing_results[0]
            for field in ['ENTRY', 'STOP_LOSS', 'TAKE_PROFIT', 'RISK_REWARD_RATIO', 'EXPIRATION_H', 'ANALYSIS']:
                if field in first_valid and first_valid[field] not in [None, "null", ""]:
                    combined[field] = first_valid[field]
                else:
                    if field == 'ANALYSIS':
                        combined[field] = f"{majority_action} signal based on agreement of {max_agreement} out of {total_models} AI models"
                    elif field == 'EXPIRATION_H':
                        combined[field] = 4
                    elif field == 'RISK_REWARD_RATIO':
                        combined[field] = "1.5"
                        
        return combined

# =================================================================================
# --- Gemini Direct Signal Agent (Non-Intrusive Add-on) ---
# =================================================================================
# این کلاسِ جدید مستقلاً با Gemini کار می‌کند و به هیچ بخش قبلی دست نمی‌زند.
# اگر خواستی مستقیم از جمینی سیگنال بگیری (بدون رأی‌گیری چندمدلی)، از این کلاس استفاده کن.

class GeminiDirectSignalAgent:
    """
    A thin, robust wrapper that queries Gemini once and returns a strict-JSON trade signal.
    It **does not modify** existing managers. Safe to plug anywhere.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = GEMINI_MODEL):
        self.model_name = model_name
        self.available = False
        try:
            k = api_key or os.getenv("GOOGLE_API_KEY")
            if k:
                genai.configure(api_key=k)
                self.available = True
            else:
                logging.warning("⚠️ GeminiDirectSignalAgent: GOOGLE_API_KEY not set.")
        except Exception as e:
            logging.error(f"GeminiDirectSignalAgent init error: {e}")

    def _schema(self):
        # حداقل اسکیمای قابل اتکا
        return {
            "type": "object",
            "properties": {
                "SYMBOL": {"type": "string"},
                "ACTION": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                "CONFIDENCE": {"type": "number"},
                "ENTRY": {"type": "string"},
                "STOP_LOSS": {"type": "string"},
                "TAKE_PROFIT": {"type": "string"},
                "RISK_REWARD_RATIO": {"type": "string"},
                "ANALYSIS": {"type": "string"},
                "EXPIRATION_H": {"type": "number"},
                "TRADE_RATIONALE": {"type": "string"}
            },
            "required": ["SYMBOL", "ACTION", "CONFIDENCE"]
        }

    def _make_prompt(self, symbol: str, ta: Dict) -> str:
        # پرامپت کوتاه و تمرکز روی خروجی JSON
        price = ta.get("current_price", 1.0)
        risk = ta.get("risk_assessment", {})
        ml = ta.get("ml_signal", {})
        key = ta.get("key_levels", {})
        return (
            "You are a strict-trader agent. Output ONLY JSON, no prose.\n\n"
            f"SYMBOL={symbol}\nPRICE={price:.6f}\n"
            f"RISK_LEVEL={risk.get('risk_level','MEDIUM')}\n"
            f"ATR={risk.get('atr_value',0.001)}\n"
            f"ML_STRENGTH={ml.get('signal_strength',0)} CONF={ml.get('confidence',0)}\n"
            f"SUP1={key.get('support_1',price*0.99)} RES1={key.get('resistance_1',price*1.01)}\n\n"
            "Return JSON with fields listed in the schema: SYMBOL,ACTION,CONFIDENCE,ENTRY,STOP_LOSS,TAKE_PROFIT,"
            "RISK_REWARD_RATIO,ANALYSIS,EXPIRATION_H,TRADE_RATIONALE.\n"
        )

    async def fetch_signal(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        if not self.available:
            return None
        try:
            model = genai.GenerativeModel(
                self.model_name,
                system_instruction="Return ONLY valid JSON. No extra text."
            )
            prompt = self._make_prompt(symbol, technical_analysis)

            # تلاش اول: با schema و MIME
            try:
                resp = await asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.05,
                        max_output_tokens=800,
                        response_mime_type="application/json",
                        response_schema=self._schema()
                    )
                )
                text = self._extract_text(resp)
                data = self._clean_parse(text)
                if data:
                    return data
            except Exception as e:
                logging.warning(f"GeminiDirect(schema) failed: {e}")

            # تلاش دوم: فقط MIME application/json
            try:
                resp = await asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.05,
                        max_output_tokens=800,
                        response_mime_type="application/json",
                    )
                )
                text = self._extract_text(resp)
                data = self._clean_parse(text)
                if data:
                    return data
            except Exception as e:
                logging.warning(f"GeminiDirect(json-mime) failed: {e}")

            # تلاش سوم: Plain + تمیزکاری
            try:
                resp = await asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=900
                    )
                )
                text = self._extract_text(resp)
                data = self._clean_parse(text)
                if data:
                    return data
            except Exception as e:
                logging.warning(f"GeminiDirect(plain) failed: {e}")

            return None
        except Exception as e:
            logging.error(f"GeminiDirectSignalAgent error: {e}")
            return None

    def _extract_text(self, resp) -> Optional[str]:
        # مثل EnhancedAIManager._extract_gemini_text ولی مختصر
        try:
            d = resp.to_dict()
            for c in d.get("candidates", []):
                for p in (c.get("content", {}).get("parts") or []):
                    if p.get("text"):
                        return p["text"]
        except Exception:
            pass
        try:
            if getattr(resp, "candidates", None):
                for c in resp.candidates:
                    if getattr(c, "content", None):
                        for p in getattr(c.content, "parts", []):
                            if getattr(p, "text", None):
                                return p.text
        except Exception:
            pass
        return None

    def _clean_parse(self, text: Optional[str]) -> Optional[Dict]:
        if not text:
            return None
        s = text.strip()
        s = re.sub(r'```json\s*', '', s)
        s = re.sub(r'```\s*', '', s)
        s = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL | re.IGNORECASE)
        s = re.sub(r'</?[^>]+>', '', s)
        if '{' in s:
            s = s[s.find('{'):]
        m = re.search(r'\{.*\}', s, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict) and "SYMBOL" in data and "ACTION" in data and "CONFIDENCE" in data:
                return data
        except Exception:
            return None
        return None


# =================================================================================
# --- Risk Manager & Execution Heuristics (Add-on Systems) ---
# =================================================================================

class EnhancedRiskManager:
    """
    Position sizing + risk guardrails.
    - Max risk per trade (percent of equity)
    - Volatility-based stop check (ATR sanity)
    - Kelly-capped sizing
    """
    def __init__(self,
                 equity: float = 10000.0,
                 risk_per_trade_pct: float = 1.0,
                 max_leverage: float = 30.0,
                 kelly_cap: float = 0.5):
        self.equity = equity
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_leverage = max_leverage
        self.kelly_cap = kelly_cap

    def size_by_atr(self, price: float, stop: float, atr: float, rr: float = 1.5) -> Dict:
        if not price or not stop or atr is None or atr <= 0:
            return {"units": 0, "risk_amount": 0, "leverage": 0}
        risk_amount = self.equity * (self.risk_per_trade_pct / 100.0)
        per_unit_risk = abs(price - stop)
        units = risk_amount / max(per_unit_risk, 1e-8)
        notional = units * price
        leverage = notional / max(self.equity, 1e-8)
        if leverage > self.max_leverage:
            scale = self.max_leverage / leverage
            units *= scale
            leverage = self.max_leverage
        return {"units": max(0, units), "risk_amount": risk_amount, "leverage": leverage}

    def kelly_adjust(self, winrate: float, rr: float, units: float) -> float:
        # Kelly fraction = p - (1-p)/R
        p = min(max(winrate, 0.0), 1.0)
        R = max(rr, 1e-6)
        k = p - (1 - p) / R
        k = max(0.0, min(k, self.kelly_cap))
        return units * (0.25 + 0.75 * k)  # clamp toward conservative base

class TradeFilter:
    """
    Regime filters and cool-down:
    - Avoid high-impact news windows (pluggable)
    - Cooldown after stop-outs
    - Spread/slippage guard (inputs pluggable)
    """
    def __init__(self):
        self.last_stopout_time = {}
        self.cooldown_minutes = 60

    def can_trade(self, symbol: str, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now(UTC)
        t = self.last_stopout_time.get(symbol)
        if t and (now - t).total_seconds() < self.cooldown_minutes * 60:
            return False
        return True

    def mark_stopout(self, symbol: str):
        self.last_stopout_time[symbol] = datetime.now(UTC)


# =================================================================================
# --- Main Forex Analyzer Class (extended with optional add-ons) ---
# =================================================================================

class ImprovedForexAnalyzer:
    def __init__(self):
        self.api_manager = SmartAPIManager(USAGE_TRACKER_FILE)
        self.technical_analyzer = AdvancedTechnicalAnalyzer()
        self.ai_manager = EnhancedAIManager(google_api_key, CLOUDFLARE_AI_API_KEY, GROQ_API_KEY, self.api_manager)
        self.data_fetcher = EnhancedDataFetcher()
        self.performance_monitor = PerformanceMonitor()

        # افزوده‌های اختیاری
        self.gemini_direct = GeminiDirectSignalAgent(google_api_key, GEMINI_MODEL)
        self.risk_manager = EnhancedRiskManager()
        self.trade_filter = TradeFilter()

    async def analyze_pair(self, pair: str) -> Optional[Dict]:
        """Complete analysis of a currency pair with comprehensive error handling"""
        logging.info(f"🔍 Starting analysis for {pair}")
        start_time = time.time()
        
        try:
            logging.info(self.api_manager.get_usage_summary())
            
            # Get market data with enhanced fetcher
            htf_df = await self.data_fetcher.get_market_data(pair, HIGH_TIMEFRAME)
            ltf_df = await self.data_fetcher.get_market_data(pair, LOW_TIMEFRAME)
            
            if htf_df is None or ltf_df is None:
                logging.warning(f"⚠️ Market data retrieval failed for {pair}")
                self.performance_monitor.record_failure()
                return None
                
            logging.info(f"✅ Retrieved data: HTF={len(htf_df)} rows, LTF={len(ltf_df)} rows")
            
            # Technical analysis with fallback
            htf_df_processed = self.technical_analyzer.calculate_enhanced_indicators(htf_df)
            ltf_df_processed = self.technical_analyzer.calculate_enhanced_indicators(ltf_df)
            
            if htf_df_processed is None or ltf_df_processed is None:
                logging.warning(f"⚠️ Technical analysis failed for {pair}")
                # Try basic analysis as fallback
                htf_df_processed = self.technical_analyzer._calculate_basic_indicators(htf_df)
                ltf_df_processed = self.technical_analyzer._calculate_basic_indicators(ltf_df)
                if htf_df_processed is None or ltf_df_processed is None:
                    self.performance_monitor.record_failure()
                    return None
                    
            technical_analysis = self.technical_analyzer.generate_comprehensive_analysis(
                pair, htf_df_processed, ltf_df_processed
            )
            
            if not technical_analysis:
                logging.warning(f"⚠️ Technical analysis generation failed for {pair}")
                self.performance_monitor.record_failure()
                return None

            # Optional pre-trade filter (cooldown, etc.)
            if not self.trade_filter.can_trade(pair):
                logging.info(f"⏸️ Cooldown active for {pair}, skipping trade signal.")
                self.performance_monitor.record_success()
                return None
                
            # AI analysis (ensemble)
            ai_analysis = await self.ai_manager.get_enhanced_ai_analysis(pair, technical_analysis)

            # اگر جمینی مستقیم هم خواستی موازی تست کنی (بدون جایگزینی):
            direct = await self.gemini_direct.fetch_signal(pair, technical_analysis) if self.gemini_direct else None
            if direct and ai_analysis is None:
                ai_analysis = direct  # فقط اگر چیزی از ensemble نیامد

            if ai_analysis:
                # محاسبه سایز پوزیشن (اختیاری)
                price = technical_analysis.get('current_price', 0.0)
                atr = technical_analysis.get('risk_assessment', {}).get('atr_value', 0.001) or 0.001
                try:
                    sl = float(ai_analysis.get('STOP_LOSS')) if ai_analysis.get('STOP_LOSS') not in [None, "N/A"] else (price - 1.5 * atr)
                except Exception:
                    sl = price - 1.5 * atr
                rr = float(ai_analysis.get('RISK_REWARD_RATIO', "1.5")) if isinstance(ai_analysis.get('RISK_REWARD_RATIO'), (int, float, str)) else 1.5
                sizing = self.risk_manager.size_by_atr(price, sl, atr, rr=rr)

                # Kelly adjust with a conservative assumed win-rate (قابل تنظیم)
                winrate_guess = 0.52
                sized_units = self.risk_manager.kelly_adjust(winrate_guess, rr, sizing["units"])
                ai_analysis["POSITION_UNITS"] = round(sized_units, 4)
                ai_analysis["LEVERAGE_EST"] = round(sizing["leverage"], 2)
                ai_analysis["RISK_USD"] = round(sizing["risk_amount"], 2)

                analysis_duration = time.time() - start_time
                self.performance_monitor.record_analysis_time(pair, analysis_duration)
                self.performance_monitor.record_success()
                
                logging.info(f"✅ Signal for {pair}: {ai_analysis['ACTION']} "
                           f"(Agreement: {ai_analysis.get('AGREEMENT_LEVEL', 0)}/{ai_analysis.get('TOTAL_MODELS', 0)})")
                return ai_analysis
                
            self.performance_monitor.record_failure()
            logging.info(f"🔍 No trading signal for {pair}")
            return None
            
        except Exception as e:
            self.performance_monitor.record_failure()
            logging.error(f"❌ Error analyzing {pair}: {str(e)}")
            logging.error(f"❌ Traceback: {traceback.format_exc()}")
            return None

    async def analyze_all_pairs(self, pairs: List[str]) -> List[Dict]:
        """Analyze all currency pairs"""
        logging.info(f"🚀 Starting analysis for {len(pairs)} currency pairs")
        
        tasks = [self.analyze_pair(pair) for pair in pairs]
        results = await asyncio.gather(*tasks)
        
        valid_signals = [r for r in results if r is not None]
        logging.info(f"📊 Analysis complete. {len(valid_signals)} valid signals")
        
        # Log performance statistics
        perf_stats = self.performance_monitor.get_performance_stats()
        logging.info(f"📈 Performance Statistics: {json.dumps(perf_stats, indent=2)}")
        
        return valid_signals

    def save_signals(self, signals: List[Dict]):
        """Save signals to files in root directory"""
        import os
        
        current_dir = os.getcwd()
        logging.info(f"📁 Current directory for file saving: {current_dir}")
        
        if not signals:
            logging.info("📝 No signals to save")
            # Create empty files in root
            empty_data = []
            try:
                files_to_create = [
                    "strong_consensus_signals.json",
                    "medium_consensus_signals.json", 
                    "weak_consensus_signals.json"
                ]
                for filename in files_to_create:
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(empty_data, f, indent=2, ensure_ascii=False)
                    logging.info(f"💾 Empty file created: {filename}")
            except Exception as e:
                logging.error(f"❌ Error creating empty files: {e}")
            return

        # Categorize signals
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
                
        # Save to root directory
        try:
            # Strong signals file
            with open("strong_consensus_signals.json", 'w', encoding='utf-8') as f:
                json.dump(strong_signals, f, indent=2, ensure_ascii=False)
            logging.info(f"💾 {len(strong_signals)} strong signals saved in root")
            
            # Medium signals file
            with open("medium_consensus_signals.json", 'w', encoding='utf-8') as f:
                json.dump(medium_signals, f, indent=2, ensure_ascii=False)
            logging.info(f"💾 {len(medium_signals)} medium signals saved in root")
            
            # Weak signals file
            with open("weak_consensus_signals.json", 'w', encoding='utf-8') as f:
                json.dump(weak_signals, f, indent=2, ensure_ascii=False)
            logging.info(f"💾 {len(weak_signals)} weak signals saved in root")
            
            # Verify file creation
            for filename in ["strong_consensus_signals.json", "medium_consensus_signals.json", "weak_consensus_signals.json"]:
                if os.path.exists(filename):
                    logging.info(f"✅ File {filename} successfully created")
                else:
                    logging.error(f"❌ File {filename} not created!")
                    
        except Exception as e:
            logging.error(f"❌ Error saving signals: {e}")

            
  # =================================================================================
# --- Installation Helper ---
# =================================================================================

def install_required_packages():
    """Install required packages if missing"""
    required_packages = ['yfinance', 'pandas-ta', 'aiohttp', 'scipy']
    
    for package in required_packages:
        try:
            if package == 'yfinance':
                import yfinance
            elif package == 'pandas-ta':
                import pandas_ta
            elif package == 'aiohttp':
                import aiohttp
            elif package == 'scipy':
                import scipy
            print(f"✅ {package} is already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully!")


# =================================================================================
# --- Main Function ---
# =================================================================================

async def main():
    """Main program execution function"""
    logging.info("🎯 Starting Forex Analysis System (Enhanced AI Engine)")
    
    # Install required packages
    install_required_packages()
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Forex Analysis System with AI')
    parser.add_argument("--pair", type=str, help="Analyze specific currency pair")
    parser.add_argument("--all", action="store_true", help="Analyze all currency pairs") 
    parser.add_argument("--pairs", type=str, help="Analyze specified currency pairs")
    parser.add_argument("--equity", type=float, help="Account equity for sizing (USD)")
    parser.add_argument("--risk_pct", type=float, help="Risk per trade percent (default 1.0)")
    parser.add_argument("--kelly_cap", type=float, help="Kelly cap (0..1, default 0.5)")
    parser.add_argument("--max_lev", type=float, help="Max leverage (default 30)")
    
    args = parser.parse_args()
    
    if args.pair:
        pairs_to_analyze = [args.pair]
    elif args.pairs:
        pairs_to_analyze = [p.strip() for p in args.pairs.split(',')]
    elif args.all:
        pairs_to_analyze = CURRENCY_PAIRS_TO_ANALYZE
    else:
        pairs_to_analyze = CURRENCY_PAIRS_TO_ANALYZE[:3]  # Default to first 3 pairs
        logging.info(f"🔍 Using default currency pairs: {', '.join(pairs_to_analyze)}")
    
    logging.info(f"🎯 Currency pairs to analyze: {', '.join(pairs_to_analyze)}")
    
    analyzer = ImprovedForexAnalyzer()

    # Optional CLI overrides for risk manager
    if args.equity is not None:
        analyzer.risk_manager.equity = float(args.equity)
    if args.risk_pct is not None:
        analyzer.risk_manager.risk_per_trade_pct = float(args.risk_pct)
    if args.kelly_cap is not None:
        analyzer.risk_manager.kelly_cap = float(args.kelly_cap)
    if args.max_lev is not None:
        analyzer.risk_manager.max_leverage = float(args.max_lev)

    signals = await analyzer.analyze_all_pairs(pairs_to_analyze)
    
    # Save signals
    analyzer.save_signals(signals)
    
    # Display results
    logging.info("📈 Results Summary:")
    strong_count = len([s for s in signals if s.get('AGREEMENT_TYPE') == 'STRONG_CONSENSUS'])
    medium_count = len([s for s in signals if s.get('AGREEMENT_TYPE') == 'MEDIUM_CONSENSUS'])
    weak_count = len([s for s in signals if s.get('AGREEMENT_TYPE') == 'WEAK_CONSENSUS'])
    
    logging.info(f"🎯 Strong signals: {strong_count}")
    logging.info(f"📊 Medium signals: {medium_count}") 
    logging.info(f"📈 Weak signals: {weak_count}")
    
    for signal in signals:
        action_icon = "🟢" if signal['ACTION'] == 'BUY' else "🔴" if signal['ACTION'] == 'SELL' else "⚪"
        logging.info(f"  {action_icon} {signal['SYMBOL']}: {signal['ACTION']} (Confidence: {signal.get('CONFIDENCE', 0)}/10)"
                     f" | Units: {signal.get('POSITION_UNITS','-')} | Lev≈{signal.get('LEVERAGE_EST','-')} | Risk ${signal.get('RISK_USD','-')}")
    
    # Display data source statistics
    data_source_stats = analyzer.data_fetcher.get_data_source_stats()
    logging.info("📊 Data Source Statistics:")
    for source, count in data_source_stats.items():
        logging.info(f"  {source}: {count} pairs")
    
    # Display performance statistics
    perf_stats = analyzer.performance_monitor.get_performance_stats()
    logging.info("🚀 Performance Statistics:")
    logging.info(f"  Total Analyses: {perf_stats['total_analyses']}")
    logging.info(f"  Success Rate: {perf_stats['success_rate']}%")
    logging.info(f"  Avg Analysis Time: {perf_stats['avg_analysis_time_sec']}s")
    logging.info(f"  Avg API Response Time: {perf_stats['avg_api_response_time_sec']}s")
    
    # Display final API status
    analyzer.api_manager.save_usage_data()
    logging.info(analyzer.api_manager.get_usage_summary())
    logging.info("🏁 System execution completed")


if __name__ == "__main__":
    asyncio.run(main())      
