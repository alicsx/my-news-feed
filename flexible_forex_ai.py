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
CANDLES_TO_FETCH = 500  # Increased for better analysis
CURRENCY_PAIRS_TO_ANALYZE = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", 
    "GBP/JPY", "EUR/JPY", "AUD/JPY", "NZD/USD", "USD/CAD"
]

CACHE_FILE = "signal_cache.json"
USAGE_TRACKER_FILE = "api_usage_tracker.json"
LOG_FILE = "trading_log.log"

# Updated AI models with more diversity
GEMINI_MODEL = 'gemini-2.5-flash-exp'

# Enhanced Cloudflare models
CLOUDFLARE_MODELS = [
    "@cf/meta/llama-4-scout-17b-16e-instruct",
    "@cf/google/gemma-3-12b-it", 
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
    "@cf/qwen/qwq-32b",
    "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b"
]

# Enhanced Groq models
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768"
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
# --- Enhanced Technical Analysis Class ---
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
        """Calculate comprehensive technical indicators"""
        if df is None or df.empty or len(df) < 100:
            return None
            
        try:
            # Ensure numeric columns
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna()

            # Trend indicators
            df.ta.ema(length=8, append=True)
            df.ta.ema(length=21, append=True)
            df.ta.ema(length=50, append=True)
            df.ta.ema(length=200, append=True)
            df.ta.wma(length=34, append=True)
            df.ta.hma(length=55, append=True)
            df.ta.adx(length=14, append=True)
            
            # Momentum indicators
            df.ta.rsi(length=14, append=True)
            df.ta.stoch(append=True)
            df.ta.macd(append=True)
            df.ta.cci(length=20, append=True)
            df.ta.willr(length=14, append=True)
            df.ta.mom(length=10, append=True)
            
            # Volatility indicators
            df.ta.bbands(length=20, std=2, append=True)
            df.ta.bbands(length=20, std=1.5, append=True)
            df.ta.atr(length=14, append=True)
            df.ta.kc(length=20, scalar=2, append=True)
            
            # Volume indicators
            df.ta.obv(append=True)
            df.ta.cmf(length=20, append=True)
            df.ta.vwap(append=True)
            
            # Advanced indicators
            df.ta.supertrend(append=True)
            df.ta.psar(append=True)
            df.ta.donchian(lower_length=20, upper_length=20, append=True)
            
            # Pivot points
            df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
            df['r1'] = 2 * df['pivot'] - df['low']
            df['s1'] = 2 * df['pivot'] - df['high']
            
            # Support and resistance levels
            df['sup_1'] = df['low'].rolling(20).min().shift(1)
            df['res_1'] = df['high'].rolling(20).max().shift(1)
            df['sup_2'] = df['low'].rolling(50).min().shift(1)
            df['res_2'] = df['high'].rolling(50).max().shift(1)
            
            # Ichimoku Cloud
            df.ta.ichimoku(append=True)
            
            # Price action patterns
            df['inside_bar'] = ((df['high'] < df['high'].shift(1)) & 
                              (df['low'] > df['low'].shift(1)))
            df['outside_bar'] = ((df['high'] > df['high'].shift(1)) & 
                               (df['low'] < df['low'].shift(1)))
            
            df.dropna(inplace=True)
            return df
            
        except Exception as e:
            logging.error(f"Error calculating indicators: {e}")
            return None

    def generate_comprehensive_analysis(self, symbol: str, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> Dict:
        """Generate comprehensive technical analysis"""
        if htf_df.empty or ltf_df.empty:
            return None

        last_htf = htf_df.iloc[-1]
        last_ltf = ltf_df.iloc[-1]
        prev_htf = htf_df.iloc[-2] if len(htf_df) > 1 else last_htf
        prev_ltf = ltf_df.iloc[-2] if len(ltf_df) > 1 else last_ltf

        # Multi-timeframe analysis
        htf_trend = self._analyze_enhanced_trend(last_htf, prev_htf, htf_df)
        ltf_trend = self._analyze_enhanced_trend(last_ltf, prev_ltf, ltf_df)
        
        # Momentum analysis
        momentum = self._analyze_momentum(last_ltf, prev_ltf)
        
        # Key levels with dynamic calculation
        key_levels = self._calculate_dynamic_levels(htf_df, ltf_df, last_ltf['close'])
        
        # Market structure
        market_structure = self._analyze_market_structure(htf_df, ltf_df)
        
        # Volume analysis
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
            'volatility': last_ltf.get('ATRr_14', 0),
            'timestamp': datetime.now(UTC).isoformat()
        }

    def _analyze_enhanced_trend(self, current: pd.Series, previous: pd.Series, df: pd.DataFrame) -> Dict:
        """Enhanced trend analysis with multiple confirmations"""
        # EMA analysis
        ema_8 = current.get('EMA_8', 0)
        ema_21 = current.get('EMA_21', 0)
        ema_50 = current.get('EMA_50', 0)
        ema_200 = current.get('EMA_200', 0)
        
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

        # ADX strength
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

    def _analyze_ichimoku(self, data: pd.Series) -> str:
        """Analyze Ichimoku Cloud signals"""
        try:
            tenkan = data.get('ISA_9', 0)
            kijun = data.get('ISB_26', 0)
            senkou_a = data.get('ICS_26', 0)
            senkou_b = data.get('ICB_26', 0)
            chikou = data.get('ITS_9', 0)
            price = data['close']

            # Cloud analysis
            if senkou_a > senkou_b:
                cloud_bullish = True
            else:
                cloud_bullish = False

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
        """Comprehensive momentum analysis"""
        rsi = current.get('RSI_14', 50)
        macd = current.get('MACD_12_26_9', 0)
        macd_signal = current.get('MACDs_12_26_9', 0)
        macd_hist = current.get('MACDh_12_26_9', 0)
        stoch_k = current.get('STOCHk_14_3_3', 50)
        stoch_d = current.get('STOCHd_14_3_3', 50)
        cci = current.get('CCI_20_0.015', 0)
        williams = current.get('WILLR_14', -50)
        momentum = current.get('MOM_10', 0)

        # RSI analysis
        if rsi > 70:
            rsi_signal = "OVERBOUGHT"
        elif rsi < 30:
            rsi_signal = "OVERSOLD"
        else:
            rsi_signal = "NEUTRAL"

        # MACD analysis
        macd_trend = "BULLISH" if macd_hist > 0 else "BEARISH"
        macd_cross = "BULLISH_CROSS" if macd > macd_signal and previous.get('MACD_12_26_9', 0) <= previous.get('MACDs_12_26_9', 0) else "BEARISH_CROSS" if macd < macd_signal and previous.get('MACD_12_26_9', 0) >= previous.get('MACDs_12_26_9', 0) else "NO_CROSS"

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
            'momentum': {'value': momentum, 'signal': "BULLISH" if momentum > 0 else "BEARISH"},
            'convergence_score': momentum_score,
            'overall_bias': "BULLISH" if momentum_score > 1 else "BEARISH" if momentum_score < -1 else "NEUTRAL"
        }

    def _calculate_dynamic_levels(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame, current_price: float) -> Dict:
        """Calculate dynamic support and resistance levels"""
        # Recent highs and lows
        recent_high_20 = ltf_df['high'].tail(20).max()
        recent_low_20 = ltf_df['low'].tail(20).min()
        recent_high_50 = ltf_df['high'].tail(50).max()
        recent_low_50 = ltf_df['low'].tail(50).min()

        # Pivot points
        pivot = ltf_df['pivot'].iloc[-1]
        r1 = ltf_df['r1'].iloc[-1]
        s1 = ltf_df['s1'].iloc[-1]

        # Bollinger Bands
        bb_upper = ltf_df.get('BBU_20_2.0', pd.Series([current_price * 1.02])).iloc[-1]
        bb_lower = ltf_df.get('BBL_20_2.0', pd.Series([current_price * 0.98])).iloc[-1]

        # Fibonacci levels (simplified)
        range_high = max(recent_high_20, recent_high_50)
        range_low = min(recent_low_20, recent_low_50)
        fib_range = range_high - range_low
        
        fib_236 = range_high - 0.236 * fib_range
        fib_382 = range_high - 0.382 * fib_range
        fib_500 = range_high - 0.5 * fib_range
        fib_618 = range_high - 0.618 * fib_range
        fib_786 = range_high - 0.786 * fib_range

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

    def _analyze_market_structure(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> Dict:
        """Analyze market structure for higher timeframe context"""
        # Higher Highs/Higher Lows analysis
        htf_highs = htf_df['high'].tail(10)
        htf_lows = htf_df['low'].tail(10)
        
        ltf_highs = ltf_df['high'].tail(20)
        ltf_lows = ltf_df['low'].tail(20)

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

    def _check_structure_break(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> bool:
        """Check if market is breaking structure"""
        if len(htf_df) < 5 or len(ltf_df) < 10:
            return False
            
        recent_htf_high = htf_df['high'].iloc[-1]
        recent_htf_low = htf_df['low'].iloc[-1]
        ltf_high = ltf_df['high'].tail(5).max()
        ltf_low = ltf_df['low'].tail(5).min()
        
        return ltf_high > recent_htf_high or ltf_low < recent_htf_low

    def _determine_market_phase(self, df: pd.DataFrame) -> str:
        """Determine market phase (accumulation, markup, distribution, markdown)"""
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

    def _analyze_volume(self, df: pd.DataFrame) -> Dict:
        """Analyze volume characteristics"""
        if 'volume' not in df.columns:
            return {'signal': 'NO_DATA', 'trend': 'UNKNOWN'}
            
        volume_trend = "INCREASING" if df['volume'].iloc[-1] > df['volume'].tail(20).mean() else "DECREASING"
        obv_trend = "BULLISH" if df.get('OBV', pd.Series([0])).iloc[-1] > df.get('OBV', pd.Series([0])).iloc[-5] else "BEARISH"
        
        return {
            'volume_trend': volume_trend,
            'obv_signal': obv_trend,
            'volume_vs_average': df['volume'].iloc[-1] / df['volume'].tail(20).mean() if df['volume'].tail(20).mean() > 0 else 1
        }

    def _assess_risk(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> Dict:
        """Assess market risk conditions"""
        ltf_volatility = ltf_df['close'].pct_change().std() * 100
        htf_volatility = htf_df['close'].pct_change().std() * 100
        
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

# =================================================================================
# --- Enhanced AI Manager with Improved English Prompts ---
# =================================================================================

class EnhancedAIManager:
    def __init__(self, gemini_api_key: str, cloudflare_api_key: str, groq_api_key: str, api_manager):
        self.gemini_api_key = gemini_api_key
        self.cloudflare_api_key = cloudflare_api_key
        self.groq_api_key = groq_api_key
        self.api_manager = api_manager
        genai.configure(api_key=gemini_api_key)

    def _create_enhanced_english_prompt(self, symbol: str, technical_analysis: Dict) -> str:
        """Create enhanced English prompt for AI analysis"""
        
        current_price = 1.0850  # Sample price - should use real data in practice
        
        return f"""IMPORTANT: You are a professional forex trading analyst. Analyze the technical setup and provide ONLY a valid JSON response.

SYMBOL: {symbol}
CURRENT PRICE: ~{current_price}

TECHNICAL ANALYSIS SUMMARY:
- HTF Trend (4H): {technical_analysis['htf_trend']['direction']} | Strength: {technical_analysis['htf_trend']['strength']} | ADX: {technical_analysis['htf_trend'].get('adx', 0):.1f}
- LTF Trend (1H): {technical_analysis['ltf_trend']['direction']} | EMA Alignment: {technical_analysis['htf_trend'].get('ema_alignment', 0)}/4
- Momentum Bias: {technical_analysis['momentum']['overall_bias']} | RSI: {technical_analysis['momentum']['rsi']['value']:.1f} ({technical_analysis['momentum']['rsi']['signal']})
- MACD: {technical_analysis['momentum']['macd']['trend']} | Signal: {technical_analysis['momentum']['macd']['cross']}
- Key Support: {technical_analysis['key_levels']['support_1']:.5f} | Key Resistance: {technical_analysis['key_levels']['resistance_1']:.5f}
- Market Structure: {technical_analysis['market_structure']['higher_timeframe_structure']}
- Risk Level: {technical_analysis['risk_assessment']['risk_level']}
- Volume Trend: {technical_analysis['volume_analysis']['volume_trend']}

CALCULATION INSTRUCTIONS:
- Calculate realistic levels based on current price ~{current_price} and technical structure
- Use ATR ({technical_analysis['risk_assessment']['atr_value']:.5f}) for stop loss calculation
- For entry: Use single price, not range
- For stop loss: Calculate based on ATR or key levels
- For take profit: Use risk-reward ratio 1.5-2.0

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
                    logging.error(f"Error in {provider}/{model_name} for {symbol}: {result}")
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
            logging.error(f"Error in AI analysis for {symbol}: {e}")
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
            logging.warning(f"Error in {provider}/{model_name} for {symbol}: {e}")
            return None

    async def _get_gemini_analysis(self, symbol: str, prompt: str, model_name: str) -> Optional[Dict]:
        """Get analysis from Gemini"""
        try:
            model = genai.GenerativeModel(model_name)
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                request_options={'timeout': 60}
            )
            return self._parse_ai_response(response.text, symbol, f"Gemini-{model_name}")
        except Exception as e:
            logging.warning(f"Gemini analysis error for {symbol}: {e}")
            return None

    async def _get_cloudflare_analysis(self, symbol: str, prompt: str, model_name: str) -> Optional[Dict]:
        """Get analysis from Cloudflare AI"""
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
            logging.warning(f"Cloudflare/{model_name} analysis error for {symbol}: {e}")
            return None

    async def _get_groq_analysis(self, symbol: str, prompt: str, model_name: str) -> Optional[Dict]:
        """Get analysis from Groq API"""
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
            logging.warning(f"Groq/{model_name} analysis error for {symbol}: {e}")
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
                    
            logging.warning(f"❌ {ai_name} response for {symbol} lacks valid JSON format")
            return None
            
        except json.JSONDecodeError as e:
            logging.error(f"JSON error in {ai_name} response for {symbol}: {e}")
            return None
        except Exception as e:
            logging.error(f"Error parsing {ai_name} response for {symbol}: {e}")
            return None

    def _validate_signal_data(self, signal_data: Dict, symbol: str) -> bool:
        """Validate signal data"""
        required_fields = ['SYMBOL', 'ACTION', 'CONFIDENCE']
        for field in required_fields:
            if field not in signal_data:
                logging.warning(f"Required field {field} missing in signal for {symbol}")
                return False
                
        action = signal_data['ACTION'].upper()
        if action not in ['BUY', 'SELL', 'HOLD']:
            logging.warning(f"Invalid ACTION for {symbol}: {action}")
            return False
            
        try:
            confidence = float(signal_data['CONFIDENCE'])
            if not (1 <= confidence <= 10):
                logging.warning(f"CONFIDENCE out of range for {symbol}: {confidence}")
                return False
        except (ValueError, TypeError):
            logging.warning(f"Invalid CONFIDENCE for {symbol}: {signal_data['CONFIDENCE']}")
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
# --- Smart API Manager (Keep existing implementation) ---
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
# --- Main Forex Analyzer Class ---
# =================================================================================

class ImprovedForexAnalyzer:
    def __init__(self):
        self.api_manager = SmartAPIManager(USAGE_TRACKER_FILE)
        self.technical_analyzer = EnhancedTechnicalAnalyzer()
        self.ai_manager = EnhancedAIManager(google_api_key, CLOUDFLARE_AI_API_KEY, GROQ_API_KEY, self.api_manager)

    async def analyze_pair(self, pair: str) -> Optional[Dict]:
        """Complete analysis of a currency pair"""
        logging.info(f"🔍 Starting analysis for {pair}")
        try:
            logging.info(self.api_manager.get_usage_summary())
            
            # Get market data
            htf_df = await self.get_market_data(pair, HIGH_TIMEFRAME)
            ltf_df = await self.get_market_data(pair, LOW_TIMEFRAME)
            
            if htf_df is None or ltf_df is None:
                logging.warning(f"⚠️ Market data retrieval failed for {pair}")
                return None

            # Technical analysis
            htf_df_processed = self.technical_analyzer.calculate_enhanced_indicators(htf_df)
            ltf_df_processed = self.technical_analyzer.calculate_enhanced_indicators(ltf_df)
            
            if htf_df_processed is None or ltf_df_processed is None:
                logging.warning(f"⚠️ Technical analysis failed for {pair}")
                return None

            technical_analysis = self.technical_analyzer.generate_comprehensive_analysis(
                pair, htf_df_processed, ltf_df_processed
            )

            if not technical_analysis:
                logging.warning(f"⚠️ Technical analysis generation failed for {pair}")
                return None

            # AI analysis
            ai_analysis = await self.ai_manager.get_enhanced_ai_analysis(pair, technical_analysis)
            
            if ai_analysis and ai_analysis.get('ACTION') != 'HOLD':
                logging.info(f"✅ Signal for {pair}: {ai_analysis['ACTION']} "
                           f"(Agreement: {ai_analysis.get('AGREEMENT_LEVEL', 0)}/{ai_analysis.get('TOTAL_MODELS', 0)})")
                return ai_analysis

            logging.info(f"🔍 No trading signal for {pair}")
            return None
            
        except Exception as e:
            logging.error(f"❌ Error analyzing {pair}: {e}")
            return None

    async def get_market_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Get market data"""
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
            logging.error(f"Market data error for {symbol}: {e}")
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
        pairs_to_analyze = CURRENCY_PAIRS_TO_ANALYZE[:2]
        logging.info(f"🔍 Using 2 main currency pairs")

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
