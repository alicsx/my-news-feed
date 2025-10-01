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

# =================================================================================
# --- Advanced Main Configuration Section ---
# =================================================================================

# API Keys
google_api_key = os.getenv("GOOGLE_API_KEY")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
CLOUDFLARE_AI_API_KEY = os.getenv("CLOUDFLARE_AI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not all([google_api_key, TWELVEDATA_API_KEY]):
    raise ValueError("Please set API keys: GOOGLE_API_KEY, TWELVEDATA_API_KEY")

# Main system configuration
HIGH_TIMEFRAME = "4h"
LOW_TIMEFRAME = "1h"
CANDLES_TO_FETCH = 500
CURRENCY_PAIRS_TO_ANALYZE = [
    "EUR/USD", "GBP/USD", "USD/CHF", "EUR/JPY",
    "AUD/JPY", "GBP/JPY", "EUR/AUD", "NZD/CAD"
]

CACHE_FILE = "signal_cache.json"
USAGE_TRACKER_FILE = "api_usage_tracker.json"
LOG_FILE = "trading_log.log"

# Updated AI models with more diversity
GEMINI_MODEL = 'gemini-2.0-flash-exp'

# Enhanced Cloudflare models
CLOUDFLARE_MODELS = [
    "@cf/meta/llama-3-8b-instruct",
    "@cf/mistralai/mistral-7b-instruct-v0.1", 
    "@cf/qwen/qwen1.5-7b-chat-awq",
    "@cf/google/gemma-2-9b-it",
    "@cf/microsoft/codestral-22b-v0.1"
]

# Enhanced Groq models
GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "llama-3.2-3b-preview",
    "llama-3.2-1b-preview"
]

# Daily API limits
API_DAILY_LIMITS = {
    "google_gemini": 1500,
    "cloudflare": 10000,
    "groq": 10000
}

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
# --- Enhanced Technical Analysis Class with Robust Error Handling ---
# =================================================================================

class EnhancedTechnicalAnalyzer:
    def __init__(self):
        self.indicators_config = {
            'trend': ['ema_8', 'ema_21', 'ema_50', 'ema_200', 'wma_34', 'hma_55', 'adx_14', 'ichimoku'],
            'momentum': ['rsi_14', 'stoch_14_3_3', 'macd', 'cci_20', 'williams_14', 'momentum_10'],
            'volatility': ['bb_20_2', 'bb_20_1.5', 'atr_14', 'kc_20_2'],
            'volume': ['obv', 'cmf_20', 'vwap'],
            'advanced': ['supertrend', 'parabolic_sar', 'donchian_20', 'pivot_points']
        }

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
                df_indicators['inside_bar'] = ((df_indicators['high'] < df_indicators['high'].shift(1)) & 
                                              (df_indicators['low'] > df_indicators['low'].shift(1)))
                df_indicators['outside_bar'] = ((df_indicators['high'] > df_indicators['high'].shift(1)) & 
                                               (df_indicators['low'] < df_indicators['low'].shift(1)))
                indicators_added.extend(['inside_bar', 'outside_bar'])
            except Exception as e:
                logging.warning(f"Failed to calculate price patterns: {e}")

            # Remove rows with too many NaN values but keep recent data
            initial_count = len(df_indicators)
            df_indicators = df_indicators.dropna(thresh=len(df_indicators.columns) - 10)  # Allow up to 10 NaN columns
            
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

            return {
                'symbol': symbol,
                'htf_trend': htf_trend,
                'ltf_trend': ltf_trend,
                'momentum': momentum,
                'key_levels': key_levels,
                'market_structure': market_structure,
                'volume_analysis': volume_analysis,
                'risk_assessment': risk_assessment,
                'volatility': last_ltf.get('ATRr_14', 0.001),
                'current_price': last_ltf['close'],
                'timestamp': datetime.now(UTC).isoformat()
            }
        except Exception as e:
            logging.error(f"❌ Error generating technical analysis for {symbol}: {e}")
            # Return basic analysis as fallback
            return self._generate_basic_analysis(symbol, htf_df, ltf_df)

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
            
            if rsi_signal == "OVERSOLD": bullish_signals += 1
            if rsi_signal == "OVERBOUGHT": bearish_signals += 1
            if macd_trend == "BULLISH": bullish_signals += 1
            if macd_trend == "BEARISH": bearish_signals += 1
            if stoch_signal == "OVERSOLD": bullish_signals += 1
            if stoch_signal == "OVERBOUGHT": bearish_signals += 1

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
            levels = [recent_high_20, recent_low_20, recent_high_50, recent_low_50, 
                     pivot, r1, s1, bb_upper, bb_lower, fib_382, fib_618]
            
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
# --- Smart API Manager ---
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

    def select_diverse_models(self, target_total: int = 5, min_required: int = 3) -> List[Tuple[str, str]]:
        """Select diverse models from different providers"""
        selected_models = []
        
        # Calculate provider capacity
        provider_capacity = {}
        for provider in ["google_gemini", "cloudflare", "groq"]:
            provider_capacity[provider] = self.get_available_models_count(provider)

        logging.info(f"📊 Provider capacity: Gemini={provider_capacity['google_gemini']}, "
                    f"Cloudflare={provider_capacity['cloudflare']}, Groq={provider_capacity['groq']}")

        # Strategy: diverse selection from all providers
        total_available = sum(provider_capacity.values())
        if total_available == 0:
            logging.error("❌ No providers available")
            return selected_models

        # Balanced distribution between providers
        providers_order = ["google_gemini", "cloudflare", "groq"]
        round_robin_index = 0
        remaining_target = min(target_total, total_available)

        while remaining_target > 0 and any(provider_capacity[p] > 0 for p in providers_order):
            current_provider = providers_order[round_robin_index % len(providers_order)]
            if provider_capacity[current_provider] > 0:
                # Select first available model from this provider that hasn't failed
                for model_name in self.available_models[current_provider]:
                    if (current_provider, model_name) not in selected_models and not self.is_model_failed(current_provider, model_name):
                        selected_models.append((current_provider, model_name))
                        provider_capacity[current_provider] -= 1
                        remaining_target -= 1
                        break
            round_robin_index += 1
            
            # Break if no addition after full rotation
            if round_robin_index > len(providers_order) * 2:
                break

        # Fallback if minimum not reached
        if len(selected_models) < min_required:
            logging.warning(f"⚠️ Only {len(selected_models)} models selected. Activating fallback...")
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
            summary += f" {provider}: {data['used_today']}/{data['limit']} ({remaining} remaining)\n"
        return summary

# =================================================================================
# --- Enhanced AI Manager with Fixed Error Handling ---
# =================================================================================

class EnhancedAIManager:
    def __init__(self, gemini_api_key: str, cloudflare_api_key: str, groq_api_key: str, api_manager):
        self.gemini_api_key = gemini_api_key
        self.cloudflare_api_key = cloudflare_api_key
        self.groq_api_key = groq_api_key
        self.api_manager = api_manager
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)

    def _create_enhanced_english_prompt(self, symbol: str, technical_analysis: Dict) -> str:
        """Create enhanced English prompt for AI analysis"""
        
        current_price = technical_analysis.get('current_price', 1.0850)
        
        # Use .get() with default values to avoid KeyErrors
        momentum_data = technical_analysis.get('momentum', {})
        stochastic_data = momentum_data.get('stochastic', {})
        htf_trend = technical_analysis.get('htf_trend', {})
        ltf_trend = technical_analysis.get('ltf_trend', {})
        key_levels = technical_analysis.get('key_levels', {})
        market_structure = technical_analysis.get('market_structure', {})
        risk_assessment = technical_analysis.get('risk_assessment', {})
        
        return f"""IMPORTANT: You are a professional forex trading analyst. Analyze the technical setup and provide ONLY a valid JSON response.

SYMBOL: {symbol}
CURRENT PRICE: {current_price:.5f}

TECHNICAL ANALYSIS SUMMARY:
- HTF Trend (4H): {htf_trend.get('direction', 'NEUTRAL')} | Strength: {htf_trend.get('strength', 'UNKNOWN')} | ADX: {htf_trend.get('adx', 0):.1f}
- LTF Trend (1H): {ltf_trend.get('direction', 'NEUTRAL')} | EMA Alignment: {htf_trend.get('ema_alignment', 0)}/4
- Momentum Bias: {momentum_data.get('overall_bias', 'NEUTRAL')} | RSI: {momentum_data.get('rsi', {}).get('value', 50):.1f} ({momentum_data.get('rsi', {}).get('signal', 'NEUTRAL')})
- MACD: {momentum_data.get('macd', {}).get('trend', 'NEUTRAL')} | Signal: {momentum_data.get('macd', {}).get('cross', 'NO_CROSS')}
- Stochastic: {stochastic_data.get('k', 50):.1f} ({stochastic_data.get('signal', 'NEUTRAL')})
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
        """Get enhanced AI analysis with multiple models"""
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
            return self._combine_signals(symbol, valid_results, len(selected_models))
            
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

    async def _get_gemini_analysis(self, symbol: str, prompt: str, model_name: str) -> Optional[Dict]:
        """Get analysis from Gemini"""
        try:
            model = genai.GenerativeModel(model_name)
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=500,
                )
            )
            return self._parse_ai_response(response.text, symbol, f"Gemini-{model_name}")
        except Exception as e:
            logging.error(f"❌ Gemini analysis error for {symbol}: {str(e)}")
            logging.error(f"❌ Traceback: {traceback.format_exc()}")
            return None

    async def _get_cloudflare_analysis(self, symbol: str, prompt: str, model_name: str) -> Optional[Dict]:
        """Get analysis from Cloudflare AI"""
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
                        else:
                            logging.warning(f"❌ Unexpected Cloudflare response structure for {symbol}: {data}")
                            return None
                            
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

    def _parse_ai_response(self, response: str, symbol: str, ai_name: str) -> Optional[Dict]:
        """Parse AI response with enhanced validation"""
        try:
            cleaned_response = response.strip()
            # Remove markdown and code blocks
            cleaned_response = re.sub(r'```json\s*', '', cleaned_response)
            cleaned_response = re.sub(r'```\s*', '', cleaned_response)
            
            # Find JSON
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                signal_data = json.loads(json_str)
                
                if self._validate_signal_data(signal_data, symbol):
                    signal_data['ai_model'] = ai_name
                    signal_data['timestamp'] = datetime.now(UTC).isoformat()
                    # Validate numeric values
                    signal_data = self._validate_numeric_values(signal_data, symbol)
                    logging.info(f"✅ {ai_name} signal for {symbol}: {signal_data.get('ACTION', 'HOLD')}")
                    return signal_data
                    
            logging.warning(f"❌ {ai_name} response for {symbol} lacks valid JSON format. Response: {response[:200]}...")
            return None
            
        except json.JSONDecodeError as e:
            logging.error(f"❌ JSON error in {ai_name} response for {symbol}: {e}. Response: {response[:200]}...")
            return None
        except Exception as e:
            logging.error(f"❌ Error parsing {ai_name} response for {symbol}: {str(e)}. Response: {response[:200]}...")
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
                    # Default values for essential fields
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

        # Count signals
        action_counts = {}
        for result in valid_results:
            action = result['ACTION'].upper()
            action_counts[action] = action_counts.get(action, 0) + 1

        total_valid = len(valid_results)
        max_agreement = max(action_counts.values())

        # Determine agreement type
        if max_agreement >= 3:
            agreement_type = 'STRONG_CONSENSUS'
        elif max_agreement == 2:
            agreement_type = 'MEDIUM_CONSENSUS'
        else:
            agreement_type = 'WEAK_CONSENSUS'

        # Select majority action
        majority_action = max(action_counts, key=action_counts.get)
        agreeing_results = [r for r in valid_results if r['ACTION'].upper() == majority_action]

        # Combine agreeing signals
        combined = {
            'SYMBOL': symbol,
            'ACTION': majority_action,
            'AGREEMENT_LEVEL': max_agreement,
            'AGREEMENT_TYPE': agreement_type,
            'VALID_MODELS': total_valid,
            'TOTAL_MODELS': total_models,
            'timestamp': datetime.now(UTC).isoformat()
        }

        # Average confidence
        if agreeing_results:
            confidences = [float(r.get('CONFIDENCE', 5)) for r in agreeing_results]
            combined['CONFIDENCE'] = round(sum(confidences) / len(confidences), 1)

        # Use values from first valid signal
        if agreeing_results:
            first_valid = agreeing_results[0]
            for field in ['ENTRY', 'STOP_LOSS', 'TAKE_PROFIT', 'RISK_REWARD_RATIO', 'EXPIRATION_H', 'ANALYSIS']:
                if field in first_valid and first_valid[field] not in [None, "null", ""]:
                    combined[field] = first_valid[field]
                else:
                    # Default values
                    if field == 'ANALYSIS':
                        combined[field] = f"{majority_action} signal based on agreement of {max_agreement} out of {total_models} AI models"
                    elif field == 'EXPIRATION_H':
                        combined[field] = 4
                    elif field == 'RISK_REWARD_RATIO':
                        combined[field] = "1.5"

        return combined

# =================================================================================
# --- Main Forex Analyzer Class with Enhanced Error Handling ---
# =================================================================================

class ImprovedForexAnalyzer:
    def __init__(self):
        self.api_manager = SmartAPIManager(USAGE_TRACKER_FILE)
        self.technical_analyzer = EnhancedTechnicalAnalyzer()
        self.ai_manager = EnhancedAIManager(google_api_key, CLOUDFLARE_AI_API_KEY, GROQ_API_KEY, self.api_manager)

    async def analyze_pair(self, pair: str) -> Optional[Dict]:
        """Complete analysis of a currency pair with comprehensive error handling"""
        logging.info(f"🔍 Starting analysis for {pair}")
        try:
            logging.info(self.api_manager.get_usage_summary())
            
            # Get market data with retry mechanism
            htf_df = await self.get_market_data_with_retry(pair, HIGH_TIMEFRAME)
            ltf_df = await self.get_market_data_with_retry(pair, LOW_TIMEFRAME)
            
            if htf_df is None or ltf_df is None:
                logging.warning(f"⚠️ Market data retrieval failed for {pair}")
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
                    return None

            technical_analysis = self.technical_analyzer.generate_comprehensive_analysis(
                pair, htf_df_processed, ltf_df_processed
            )

            if not technical_analysis:
                logging.warning(f"⚠️ Technical analysis generation failed for {pair}")
                return None

            # AI analysis
            ai_analysis = await self.ai_manager.get_enhanced_ai_analysis(pair, technical_analysis)
            
            if ai_analysis:
                logging.info(f"✅ Signal for {pair}: {ai_analysis['ACTION']} "
                           f"(Agreement: {ai_analysis.get('AGREEMENT_LEVEL', 0)}/{ai_analysis.get('TOTAL_MODELS', 0)})")
                return ai_analysis

            logging.info(f"🔍 No trading signal for {pair}")
            return None
            
        except Exception as e:
            logging.error(f"❌ Error analyzing {pair}: {str(e)}")
            logging.error(f"❌ Traceback: {traceback.format_exc()}")
            return None

    async def get_market_data_with_retry(self, symbol: str, interval: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Get market data with retry mechanism"""
        for attempt in range(max_retries):
            try:
                df = await self.get_market_data(symbol, interval)
                if df is not None and not df.empty and len(df) > 50:
                    return df
                logging.warning(f"Attempt {attempt + 1} failed for {symbol} - insufficient data")
                await asyncio.sleep(2)  # Wait before retry
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} error for {symbol}: {str(e)}")
                await asyncio.sleep(2)
        
        logging.error(f"❌ All {max_retries} attempts failed for {symbol}")
        return None

    async def get_market_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Get market data from Twelve Data API"""
        try:
            url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={CANDLES_TO_FETCH}&apikey={TWELVEDATA_API_KEY}'
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=45) as response:
                    if response.status == 200:
                        data = await response.json()
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
                                logging.info(f"✅ Retrieved {len(df)} candles for {symbol} ({interval})")
                                return df
                            else:
                                logging.warning(f"⚠️ Insufficient valid data after cleaning for {symbol}: {len(df)} rows")
                                return None
                        else:
                            logging.warning(f"⚠️ No values in response for {symbol}")
                            return None
                    else:
                        logging.warning(f"⚠️ API response {response.status} for {symbol}")
                        return None
                        
        except Exception as e:
            logging.error(f"❌ Market data error for {symbol}: {str(e)}")
            return None

    async def analyze_all_pairs(self, pairs: List[str]) -> List[Dict]:
        """Analyze all currency pairs"""
        logging.info(f"🚀 Starting analysis for {len(pairs)} currency pairs")
        tasks = [self.analyze_pair(pair) for pair in pairs]
        results = await asyncio.gather(*tasks)
        valid_signals = [r for r in results if r is not None]
        logging.info(f"📊 Analysis complete. {len(valid_signals)} valid signals")
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
# --- Main Function ---
# =================================================================================

async def main():
    """Main program execution function"""
    logging.info("🎯 Starting Forex Analysis System (Enhanced AI Engine)")
    
    import argparse
    parser = argparse.ArgumentParser(description='Forex Analysis System with AI')
    parser.add_argument("--pair", type=str, help="Analyze specific currency pair")
    parser.add_argument("--all", action="store_true", help="Analyze all currency pairs") 
    parser.add_argument("--pairs", type=str, help="Analyze specified currency pairs")
    
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
        logging.info(f" {action_icon} {signal['SYMBOL']}: {signal['ACTION']} (Confidence: {signal.get('CONFIDENCE', 0)}/10)")

    # Display final API status
    analyzer.api_manager.save_usage_data()
    logging.info(analyzer.api_manager.get_usage_summary())
    logging.info("🏁 System execution completed")

if __name__ == "__main__":
    asyncio.run(main())
