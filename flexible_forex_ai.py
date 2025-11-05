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

# FIXED: Use ONLY free Gemini models
GEMINI_FREE_MODELS = [
    'gemini-1.5-flash',
    'gemini-2.0-flash-exp',
    'gemini-1.5-pro'
]

GEMINI_MODEL = 'gemini-1.5-flash'  # Default free model
GEMINI_FALLBACK_MODEL = 'gemini-1.5-pro'  # Fallback free model

# Enhanced Cloudflare models - Diverse AI families (updated & non-deprecated)
CLOUDFLARE_MODELS = [
    "@cf/meta/llama-3.1-8b-instruct",               # Meta Llama
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",     # Meta Llama large fast
    "@cf/meta/llama-4-scout-17b-16e-instruct",      # Meta Llama scout
    "@cf/google/gemma-3-12b-it",                    # Google Gemma 3
    "@cf/mistral/mistral-small-3.1-24b-instruct",   # Mistral
    "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b", # DeepSeek
    "@cf/qwen/qwq-32b"                              # Qwen reasoning
]

# Enhanced Groq models - Diverse AI families
GROQ_MODELS = [
    "llama-3.3-70b-versatile",                  # Meta Llama
    "qwen-2.5-32b",                             # Qwen
    "mixtral-8x7b-32768",                       # Mistral
    "gemma2-9b-it",                             # Google Gemma
    "deepseek-r1-distill-qwen-32b"              # DeepSeek
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
# --- NEW: AI Model Diversity Manager ---
# =================================================================================

class AIModelDiversityManager:
    """Manager to ensure diverse AI model selection across different families"""
    
    def __init__(self):
        self.model_families = {
            'llama': ['llama', 'meta-llama'],
            'gemini': ['gemini'],
            'qwen': ['qwen'],
            'mistral': ['mistral', 'mixtral'],
            'gemma': ['gemma'],
            'deepseek': ['deepseek'],
            'claude': ['claude']
        }
        
    def get_model_family(self, model_name: str) -> str:
        """Identify the AI family of a model"""
        model_lower = model_name.lower()
        for family, keywords in self.model_families.items():
            if any(keyword in model_lower for keyword in keywords):
                return family
        return 'unknown'
    
    def select_diverse_models(self, available_models: List[Tuple[str, str]], target_count: int = 5) -> List[Tuple[str, str]]:
        """Select models from diverse AI families"""
        diversity_manager = AIModelDiversityManager()
        selected_models = []
        used_families = set()
        
        # First pass: try to get one model from each family
        for provider, model in available_models:
            if len(selected_models) >= target_count:
                break
            family = diversity_manager.get_model_family(model)
            if family not in used_families:
                selected_models.append((provider, model))
                used_families.add(family)
                logging.info(f"🧠 Selected diverse model: {provider}/{model} (Family: {family})")
        
        # Second pass: fill remaining slots with any available models
        for provider, model in available_models:
            if len(selected_models) >= target_count:
                break
            if (provider, model) not in selected_models:
                selected_models.append((provider, model))
                family = diversity_manager.get_model_family(model)
                logging.info(f"🧠 Added additional model: {provider}/{model} (Family: {family})")
        
        # Log diversity summary
        family_count = {}
        for provider, model in selected_models:
            family = diversity_manager.get_model_family(model)
            family_count[family] = family_count.get(family, 0) + 1
        
        logging.info(f"🧭 Model Diversity: {family_count}")
        return selected_models

# =================================================================================
# --- Enhanced Free Tier Model Filter System ---
# =================================================================================

class FreeTierModelFilter:
    """Filter to only use free tier models for Gemini"""
    
    @staticmethod
    def filter_gemini_models(available_models: List[str]) -> List[str]:
        """Filter Gemini models to only include free tier ones"""
        free_models = []
        for model in available_models:
            # Only allow free tier models - updated for newer Gemini models
            if any(free_model in model.lower() for free_model in ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash-exp']):
                free_models.append(model)
        
        # If no free models found, use our predefined free models
        if not free_models:
            free_models = GEMINI_FREE_MODELS.copy()
            
        logging.info(f"🧠 Free tier Gemini models available: {free_models}")
        return free_models
    
    @staticmethod
    def is_free_tier_model(model_name: str) -> bool:
        """Check if a model is in free tier"""
        return any(free_model in model_name.lower() for free_model in 
                  ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash-exp'])

# =================================================================================
# --- NEW: Advanced Dynamic Model Discovery System ---
# =================================================================================

class AdvancedDynamicModelDiscoverer:
    """Enhanced model discovery with diversity tracking and family classification"""
    
    def __init__(self):
        self.available_models = {
            "google_gemini": [],
            "cloudflare": [],
            "groq": []
        }
        self.fallback_models = {
            "google_gemini": GEMINI_FREE_MODELS,  # Only free models
            "cloudflare": CLOUDFLARE_MODELS,
            "groq": GROQ_MODELS
        }
        self.model_filter = FreeTierModelFilter()
        self.diversity_manager = AIModelDiversityManager()
        
    async def discover_models(self) -> Dict[str, List[str]]:
        """Discover available models from all providers with enhanced diversity"""
        discovery_tasks = [
            self._discover_gemini_models(),
            self._discover_cloudflare_models(),
            self._discover_groq_models()
        ]
        
        results = await asyncio.gather(*discovery_tasks, return_exceptions=True)
        
        # Process results with enhanced error handling
        for i, result in enumerate(results):
            provider = list(self.available_models.keys())[i]
            if isinstance(result, Exception):
                logging.warning(f"❗ Model discovery failed for {provider}: {result}")
                # Use fallback models but ensure diversity
                self.available_models[provider] = self._ensure_fallback_diversity(provider)
            elif result:
                self.available_models[provider] = result
            else:
                self.available_models[provider] = self._ensure_fallback_diversity(provider)
                
        # Apply free tier filter for Gemini
        self.available_models["google_gemini"] = self.model_filter.filter_gemini_models(
            self.available_models["google_gemini"]
        )
        
        # Log model diversity
        self._log_model_diversity()
                   
        return self.available_models
    
    def _ensure_fallback_diversity(self, provider: str) -> List[str]:
        """Ensure fallback models maintain diversity"""
        fallback_models = self.fallback_models[provider].copy()
        
        # For Gemini, only return free models
        if provider == "google_gemini":
            return self.model_filter.filter_gemini_models(fallback_models)
            
        return fallback_models
    
    def _log_model_diversity(self):
        """Log the diversity of discovered models"""
        for provider, models in self.available_models.items():
            families = {}
            for model in models:
                family = self.diversity_manager.get_model_family(model)
                families[family] = families.get(family, 0) + 1
            logging.info(f"🧭 {provider} model diversity: {families}")
                
        total_models = sum(len(models) for models in self.available_models.values())
        logging.info(f"🧠 Total discovered models: Gemini({len(self.available_models['google_gemini'])}), "
                   f"Cloudflare({len(self.available_models['cloudflare'])}), "
                   f"Groq({len(self.available_models['groq'])})")

    async def _discover_gemini_models(self) -> List[str]:
        """Discover available Gemini models with free tier filtering"""
        if not google_api_key:
            return self.fallback_models["google_gemini"]
            
        try:
            genai.configure(api_key=google_api_key)
            models = genai.list_models()
            
            available_models = []
            for model in models:
                model_name = model.name
                # Filter for relevant models and free tier only
                if ('gemini' in model_name.lower() and 
                    'generateContent' in model.supported_generation_methods and
                    self.model_filter.is_free_tier_model(model_name)):
                    
                    available_models.append(model_name.split('/')[-1])  # Extract model name only
            
            # Prioritize our preferred free models
            preferred_models = GEMINI_FREE_MODELS
            for preferred in preferred_models:
                if preferred in available_models:
                    available_models.remove(preferred)
                    available_models.insert(0, preferred)
                    
            return available_models if available_models else self.fallback_models["google_gemini"]
            
        except Exception as e:
            logging.warning(f"❗ Gemini model discovery failed: {e}")
            return self.fallback_models["google_gemini"]
    
    async def _discover_cloudflare_models(self) -> List[str]:
        """Discover available Cloudflare models with enhanced error handling"""
        if not CLOUDFLARE_AI_API_KEY:
            return self.fallback_models["cloudflare"]
            
        try:
            headers = {
                "Authorization": f"Bearer {CLOUDFLARE_AI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
            if not account_id:
                logging.warning("⚠️ CLOUDFLARE_ACCOUNT_ID is not set; using fallback model catalog.")
                return self.fallback_models["cloudflare"]

            url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model.get("id") for model in data.get("result", []) if model.get("id")]
                        
                        # Prioritize diverse models
                        diverse_models = self._prioritize_diverse_models(models, "cloudflare")
                        return diverse_models if diverse_models else self.fallback_models["cloudflare"]
                    else:
                        logging.warning(f"❗ Cloudflare API returned status {response.status}")
                        return self.fallback_models["cloudflare"]
                        
        except Exception as e:
            logging.warning(f"❗ Cloudflare model discovery failed: {e}")
            return self.fallback_models["cloudflare"]
    
    async def _discover_groq_models(self) -> List[str]:
        """Discover available Groq models with enhanced error handling"""
        if not GROQ_API_KEY:
            return self.fallback_models["groq"]
            
        try:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            url = "https://api.groq.com/openai/v1/models"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model["id"] for model in data.get("data", [])]
                        
                        # Prioritize diverse models
                        diverse_models = self._prioritize_diverse_models(models, "groq")
                        return diverse_models if diverse_models else self.fallback_models["groq"]
                    else:
                        logging.warning(f"❗ Groq API returned status {response.status}")
                        return self.fallback_models["groq"]
                        
        except Exception as e:
            logging.warning(f"❗ Groq model discovery failed: {e}")
            return self.fallback_models["groq"]
    
    def _prioritize_diverse_models(self, models: List[str], provider: str) -> List[str]:
        """Prioritize models to ensure diversity across AI families"""
        if not models:
            return models
            
        # Classify models by family
        families = {}
        for model in models:
            family = self.diversity_manager.get_model_family(model)
            if family not in families:
                families[family] = []
            families[family].append(model)
        
        # Build diverse list - one from each family first
        diverse_models = []
        for family in families:
            if families[family]:
                diverse_models.append(families[family][0])
        
        # Add remaining models
        for family in families:
            for model in families[family][1:]:
                diverse_models.append(model)
        
        return diverse_models

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
        self.model_performance = {}
        
    def record_analysis_time(self, symbol: str, duration: float):
        """Record analysis duration"""
        self.analysis_times.append((symbol, duration))
        
    def record_api_time(self, provider: str, duration: float):
        """Record API response time"""
        self.api_response_times.append((provider, duration))
        
    def record_model_performance(self, provider: str, model: str, success: bool, response_time: float):
        """Record individual model performance"""
        key = f"{provider}/{model}"
        if key not in self.model_performance:
            self.model_performance[key] = {'success': 0, 'failures': 0, 'total_time': 0, 'count': 0}
        
        if success:
            self.model_performance[key]['success'] += 1
        else:
            self.model_performance[key]['failures'] += 1
            
        self.model_performance[key]['total_time'] += response_time
        self.model_performance[key]['count'] += 1
        
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
        
        # Model performance summary
        model_stats = {}
        for model, data in self.model_performance.items():
            total_calls = data['success'] + data['failures']
            success_rate_model = (data['success'] / total_calls * 100) if total_calls > 0 else 0
            avg_time = data['total_time'] / data['count'] if data['count'] > 0 else 0
            model_stats[model] = {
                'success_rate': round(success_rate_model, 1),
                'avg_response_time': round(avg_time, 2),
                'total_calls': total_calls
            }
        
        return {
            "total_analyses": total_analyses,
            "success_rate": round(success_rate, 2),
            "avg_analysis_time_sec": round(avg_analysis_time, 2),
            "avg_api_response_time_sec": round(avg_api_time, 2),
            "model_performance": model_stats,
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
                logging.warning(f"❗ {source} failed for {symbol}: {str(e)}")
                continue
                
        logging.error(f"❗ All data sources failed for {symbol}")
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
                logging.warning(f"❗ TwelveData error on attempt {attempt + 1} for {symbol}: {str(e)}")
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
                                logging.warning(f"🧱 Rate limit hit for {symbol}, will use fallback")
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
            logging.warning(f"⏱️ TwelveData timeout for {symbol}")
            return DataFetchResult(False, None, DataSource.TWELVEDATA, symbol, "Timeout")
        except Exception as e:
            logging.warning(f"❗ TwelveData exception for {symbol}: {str(e)}")
            return DataFetchResult(False, None, DataSource.TWELVEDATA, symbol, str(e))

    async def _get_yahoo_data(self, symbol: str, interval: str) -> DataFetchResult:
        """Get data from Yahoo Finance as fallback"""
        try:
            # Convert symbol to Yahoo format
            yahoo_symbol = YAHOO_SYMBOLS.get(symbol, symbol.replace("/", "") + "=X")
            
            # Map intervals
            interval_map = {
                "1h": "1h",
                "4h": "1h",  # fetch 1h and resample to 4h
                "1d": "1d",
                "1m": "1m",
                "5m": "5m",
                "15m": "15m"
            }
            
            yf_interval = interval_map.get(interval, "1h")
            period = "60d" if interval in ["1h", "4h"] else "120d"
            
            # Download data using async thread (yfinance.history has no 'timeout' kwarg)
            ticker = yf.Ticker(yahoo_symbol)
            df = await asyncio.to_thread(
                ticker.history, 
                period=period, 
                interval=yf_interval
            )
            
            if df.empty:
                return DataFetchResult(False, None, DataSource.YAHOO, symbol, "Empty DataFrame from Yahoo")
                
            # Resample to 4H if needed
            if interval == "4h":
                df = df.resample("4H").agg({
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum"
                }).dropna()

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
            logging.warning(f"❗ Yahoo Finance error for {symbol}: {str(e)}")
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
            
            logging.info(f"🧪 Synthetic data generated for {symbol} ({interval}): {len(df)} candles")
            return DataFetchResult(True, df, DataSource.SYNTHETIC, symbol)
            
        except Exception as e:
            logging.warning(f"❗ Synthetic data generation failed for {symbol}: {str(e)}")
            return DataFetchResult(False, None, DataSource.SYNTHETIC, symbol, str(e))

    def get_data_source_stats(self) -> Dict:
        """Get statistics about data sources used"""
        stats = {}
        for source in DataSource:
            count = list(self.last_data_source.values()).count(source)
            stats[source.value] = count
        return stats

# =================================================================================
# --- Smart API Manager with Enhanced Model Selection & Dynamic Discovery ---
# =================================================================================

class SmartAPIManager:
    def __init__(self, usage_file: str, model_discoverer: AdvancedDynamicModelDiscoverer):
        self.usage_file = usage_file
        self.model_discoverer = model_discoverer
        self.usage_data = self.load_usage_data()
        self.available_models = {}
        self.failed_models = set()
        self.models_initialized = False
        self.gemini_disabled = False
        self.diversity_manager = AIModelDiversityManager()

    async def initialize_models(self):
        """Initialize available models dynamically"""
        if not self.models_initialized:
            self.available_models = await self.model_discoverer.discover_models()
            self.models_initialized = True
            logging.info("🧠 AI Models initialized dynamically with diversity")

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
        available_models = len(self.available_models.get(provider, []))
        return min(remaining, available_models)

    def mark_model_failed(self, provider: str, model_name: str):
        """Mark model as failed"""
        self.failed_models.add((provider, model_name))
        logging.warning(f"❗ Model {provider}/{model_name} added to failed list")

    def is_model_failed(self, provider: str, model_name: str) -> bool:
        """Check if model failed"""
        return (provider, model_name) in self.failed_models

    def _find_model_provider(self, model_name: str) -> Optional[str]:
        """Find which provider a model belongs to"""
        for provider, models in self.available_models.items():
            if model_name in models:
                return provider
        return None

    def disable_gemini(self):
        """Disable Gemini temporarily due to quota limits"""
        self.gemini_disabled = True
        logging.warning("🚫 Gemini temporarily disabled due to quota limits")

    def is_gemini_available(self) -> bool:
        """Check if Gemini has available quota"""
        if self.gemini_disabled:
            return False
        if not self.can_use_provider("google_gemini"):
            return False
        gemini_models = self.available_models.get("google_gemini", [])
        return len(gemini_models) > 0

    def select_diverse_models(self, target_total: int = 5, min_required: int = 3) -> List[Tuple[str, str]]:
        """NEW: Select diverse models from different AI families with enhanced logic"""
        selected_models = []
        
        # Calculate provider capacity with Gemini availability check
        provider_capacity = {}
        for provider in ["google_gemini", "cloudflare", "groq"]:
            if provider == "google_gemini" and not self.is_gemini_available():
                provider_capacity[provider] = 0
            else:
                provider_capacity[provider] = self.get_available_models_count(provider)
            
        logging.info(f"📊 Provider capacity: Gemini={provider_capacity['google_gemini']}, "
                   f"Cloudflare={provider_capacity['cloudflare']}, Groq={provider_capacity['groq']}")

        # NEW: Get all available models across providers
        all_available_models = []
        for provider in ["google_gemini", "cloudflare", "groq"]:
            if provider_capacity[provider] > 0:
                for model in self.available_models.get(provider, []):
                    # For Gemini, only use free tier models
                    if provider == "google_gemini" and not FreeTierModelFilter.is_free_tier_model(model):
                        continue
                    if not self.is_model_failed(provider, model):
                        all_available_models.append((provider, model))
        
        # NEW: Use diversity manager to select 5 diverse models
        if all_available_models:
            selected_models = self.diversity_manager.select_diverse_models(all_available_models, target_total)
            
            # If we don't have enough diverse models, fill with any available
            if len(selected_models) < target_total:
                remaining_needed = target_total - len(selected_models)
                for provider, model in all_available_models:
                    if (provider, model) not in selected_models and remaining_needed > 0:
                        selected_models.append((provider, model))
                        remaining_needed -= 1
                        logging.info(f"🧠 Added fill model: {provider}/{model}")
        else:
            logging.warning("⚠️ No available models found for selection")

        # FINAL FALLBACK: If still not enough, use synthetic decision maker
        if len(selected_models) == 0:
            logging.error("❗ No AI models available. Using synthetic decision maker.")
            # This ensures we always have at least one "model"
            selected_models.append(("synthetic", "technical_analyzer"))
            
        logging.info(f"🧠 {len(selected_models)} diverse models selected: {selected_models}")
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
# --- Enhanced AI Manager with Diverse Model Support ---
# =================================================================================

class EnhancedAIManager:
    def __init__(self, gemini_api_key: str, cloudflare_api_key: str, groq_api_key: str, api_manager):
        self.gemini_api_key = gemini_api_key
        self.cloudflare_api_key = cloudflare_api_key
        self.groq_api_key = groq_api_key
        self.api_manager = api_manager
        self.performance_monitor = PerformanceMonitor()
        
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)

    def _create_optimized_prompt(self, symbol: str, technical_analysis: Dict) -> str:
        """Create prompt that encourages decisive signals"""
        current_price = technical_analysis.get('current_price', 1.0850)
        trend = technical_analysis.get('htf_trend', {})
        momentum = technical_analysis.get('momentum', {})
        key_levels = technical_analysis.get('key_levels', {})
        risk = technical_analysis.get('risk_assessment', {})
        
        return f"""As a professional forex trader, analyze {symbol} and provide a TRADING DECISION.

CRITICAL INSTRUCTIONS:
- Be DECISIVE: Prefer BUY/SELL over HOLD when there's reasonable evidence
- Only use HOLD if market conditions are completely unclear
- Consider risk-reward ratios above 1.5 as acceptable

MARKET DATA:
Price: {current_price:.5f}
Trend: {trend.get('direction', 'NEUTRAL')} (Strength: {trend.get('strength', 'UNKNOWN')})
RSI: {momentum.get('rsi', {}).get('value', 50):.1f} ({momentum.get('rsi', {}).get('signal', 'NEUTRAL')})
Support: {key_levels.get('support_1', current_price * 0.99):.5f}
Resistance: {key_levels.get('resistance_1', current_price * 1.01):.5f}
Volatility: {risk.get('volatility_percent', 0):.2f}%
Risk Level: {risk.get('risk_level', 'MEDIUM')}

TRADING GUIDELINES:
- BUY if trend is bullish and RSI is not overbought
- SELL if trend is bearish and RSI is not oversold  
- HOLD only if trend is completely neutral

Return ONLY this JSON format (NO other text):
{{
  "SYMBOL": "{symbol}",
  "ACTION": "BUY|SELL|HOLD",
  "CONFIDENCE": 1-10,
  "ENTRY": "{current_price:.5f}",
  "STOP_LOSS": "calculated_price",
  "TAKE_PROFIT": "calculated_price",
  "RISK_REWARD_RATIO": "1.5-3.0",
  "ANALYSIS": "brief_reasoning"
}}"""

    async def _timed(self, coro, provider: str, model: str):
        """Measure latency and capture result/exceptions"""
        t0 = time.time()
        try:
            out = await coro
            return out, time.time() - t0, True, provider, model
        except Exception as e:
            return e, time.time() - t0, False, provider, model

    async def get_enhanced_ai_analysis(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        """Get enhanced AI analysis with diverse model ensemble"""
        await self.api_manager.initialize_models()
        selected_models = self.api_manager.select_diverse_models(target_total=5, min_required=3)
        
        if len(selected_models) < 3:
            logging.error(f"❗ Cannot find minimum 3 AI models for {symbol}")
            return None
            
        logging.info(f"🧠 Using {len(selected_models)} diverse AI models for {symbol}")

        tasks = []
        for provider, model_name in selected_models:
            # Handle synthetic fallback
            if provider == "synthetic":
                task = self._timed(self._get_synthetic_analysis(symbol, technical_analysis), provider, model_name)
            else:
                task = self._timed(self._get_single_analysis(symbol, technical_analysis, provider, model_name), provider, model_name)
            tasks.append(task)

        try:
            results = await asyncio.gather(*tasks, return_exceptions=False)
            
            valid_results = []
            failed_count = 0
            
            for result, elapsed, ok, provider, model_name in results:
                if ok and not isinstance(result, Exception) and result is not None:
                    valid_results.append(result)
                    if provider != "synthetic":
                        self.api_manager.record_api_usage(provider)
                        self.performance_monitor.record_model_performance(provider, model_name, True, elapsed)
                else:
                    failed_count += 1
                    if provider != "synthetic":
                        self.api_manager.record_api_usage(provider)
                        self.performance_monitor.record_model_performance(provider, model_name, False, elapsed)
                        # If explicit error, mark failed model (helps skip deprecated/invalid models)
                        if isinstance(result, Exception):
                            self.api_manager.mark_model_failed(provider, model_name)

            logging.info(f"📊 Results: {len(valid_results)} successful, {failed_count} failed")
            
            if valid_results:
                combined_signal = self._combine_signals(symbol, valid_results, len(selected_models))
                if combined_signal:
                    return combined_signal
                else:
                    logging.warning(f"⚠️ Signal combination failed for {symbol}")
            else:
                logging.warning(f"⚠️ No valid AI results for {symbol}")
                
            # Ultimate fallback: use technical analysis only
            return await self._get_technical_fallback(symbol, technical_analysis)
                
        except Exception as e:
            logging.error(f"❗ Error in AI analysis for {symbol}: {str(e)}")
            return await self._get_technical_fallback(symbol, technical_analysis)

    async def _retry_with_alternate_same_family(self, symbol: str, technical_analysis: dict, provider: str, model_name: str):
        """
        یک بار تلاش مجدد با مدل هم‌خانواده از provider دیگر (اولویت با Groq).
        اگر چیزی پیدا نشد یا دوباره fail شد، None برمی‌گرداند.
        """
        family = self.api_manager.diversity_manager.get_model_family(model_name)

        # ترتیب اولویت: Groq -> Gemini -> Cloudflare (هر کدام غیر از provider فعلی)
        provider_order = ["groq", "google_gemini", "cloudflare"]
        for alt_provider in provider_order:
            if alt_provider == provider:
                continue
            for alt_model in self.api_manager.available_models.get(alt_provider, []):
                if self.api_manager.diversity_manager.get_model_family(alt_model) == family \
                   and not self.api_manager.is_model_failed(alt_provider, alt_model):
                    logging.info(f"🔁 Retrying {symbol} with alternate {alt_provider}/{alt_model} (family={family})")
                    try:
                        # مستقیم همان مسیر کال اصلی هر provider را صدا بزنیم تا ریکرسیو نشود
                        if alt_provider == "google_gemini":
                            return await self._get_gemini_analysis_optimized(symbol, self._create_optimized_prompt(symbol, technical_analysis), alt_model)
                        elif alt_provider == "cloudflare":
                            return await self._get_cloudflare_analysis(symbol, self._create_optimized_prompt(symbol, technical_analysis), alt_model)
                        elif alt_provider == "groq":
                            return await self._get_groq_analysis(symbol, self._create_optimized_prompt(symbol, technical_analysis), alt_model)
                    except Exception as e2:
                        logging.error(f"Alternate model also failed: {alt_provider}/{alt_model} -> {e2}")
                        self.api_manager.mark_model_failed(alt_provider, alt_model)
                        continue
        return None
    async def _get_single_analysis(self, symbol: str, technical_analysis: Dict, provider: str, model_name: str) -> Optional[Dict]:
        """Get analysis from single AI model with enhanced error handling"""
        try:
            prompt = self._create_optimized_prompt(symbol, technical_analysis)
            
            if provider == "google_gemini":
                return await self._get_gemini_analysis_optimized(symbol, prompt, model_name)
            elif provider == "cloudflare":
                return await self._get_cloudflare_analysis(symbol, prompt, model_name)
            elif provider == "groq":
                return await self._get_groq_analysis(symbol, prompt, model_name)
            else:
                return None
                
        except Exception as e:
            logging.error(f"❗ Error in {provider}/{model_name} for {symbol}: {str(e)}")
            self.api_manager.mark_model_failed(provider, model_name)
    # 🔽 افزوده جدید: تلاشِ یک‌باره با مدل هم‌خانواده از provider دیگر (اولویت با Groq)
        return await self._retry_with_alternate_same_family(symbol, technical_analysis, provider, model_name)
            

    async def _get_synthetic_analysis(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        """Synthetic analysis based on technical indicators only"""
        try:
            trend = technical_analysis.get('htf_trend', {})
            momentum = technical_analysis.get('momentum', {})
            current_price = technical_analysis.get('current_price', 1.0)
            
            # Simple decision logic based on technicals
            trend_direction = trend.get('direction', 'NEUTRAL')
            rsi = momentum.get('rsi', {}).get('value', 50)
            rsi_signal = momentum.get('rsi', {}).get('signal', 'NEUTRAL')
            
            action = "HOLD"
            confidence = 5
            
            if trend_direction == "BULLISH" and rsi_signal != "OVERBOUGHT" and rsi < 70:
                action = "BUY"
                confidence = 7
            elif trend_direction == "BEARISH" and rsi_signal != "OVERSOLD" and rsi > 30:
                action = "SELL" 
                confidence = 7
            elif trend_direction in ["STRONG_BULLISH", "STRONG_BEARISH"]:
                action = "BUY" if "BULL" in trend_direction else "SELL"
                confidence = 8
                
            return {
                "SYMBOL": symbol,
                "ACTION": action,
                "CONFIDENCE": confidence,
                "ENTRY": f"{current_price:.5f}",
                "STOP_LOSS": f"{current_price * 0.995:.5f}" if action == "BUY" else f"{current_price * 1.005:.5f}",
                "TAKE_PROFIT": f"{current_price * 1.01:.5f}" if action == "BUY" else f"{current_price * 0.99:.5f}",
                "RISK_REWARD_RATIO": "1.8",
                "ANALYSIS": f"Synthetic signal based on {trend_direction} trend and RSI {rsi:.1f}",
                "ai_model": "SYNTHETIC_TECHNICAL"
            }
            
        except Exception as e:
            logging.error(f"❗ Synthetic analysis error for {symbol}: {e}")
            return None

    async def _get_technical_fallback(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        """Ultimate fallback using pure technical analysis"""
        try:
            return await self._get_synthetic_analysis(symbol, technical_analysis)
        except Exception as e:
            logging.error(f"❗ Technical fallback also failed for {symbol}: {e}")
            return None
            
    async def _get_gemini_analysis_optimized(self, symbol: str, prompt: str, model_name: str) -> Optional[Dict]:
        """Optimized Gemini analysis with proper error handling"""
        try:
            model = genai.GenerativeModel(model_name)
            
            generation_config = genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=500,
                top_p=0.8,
                top_k=40
            )
            
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=generation_config
            )
            
            text = self._extract_response_text(response)
            if text:
                return self._parse_ai_response(text, symbol, f"Gemini-{model_name}")
            
            return None
            
        except Exception as e:
            logging.error(f"❗ Gemini analysis error for {symbol}: {str(e)}")
            return None

    def _extract_response_text(self, response) -> Optional[str]:
        """Safely extract text from AI response with enhanced error handling"""
        try:
            # Method 1: Direct text attribute
            if hasattr(response, 'text') and response.text:
                return response.text
                
            # Method 2: Try to access parts directly
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        text_parts = []
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                text_parts.append(part.text)
                        if text_parts:
                            return ''.join(text_parts)
            
            # Method 3: Use string representation as last resort
            return str(response)
            
        except Exception as e:
            logging.warning(f"Error extracting response text: {e}")
            return None

    async def _get_cloudflare_analysis(self, symbol: str, prompt: str, model_name: str) -> Optional[Dict]:
        """Get analysis from Cloudflare AI"""
        if not self.cloudflare_api_key:
            logging.warning(f"⚠️ Cloudflare API key not available for {symbol}")
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
                        "content": "Return ONLY valid JSON format. No additional text."
                    },
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
            
            account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
            if not account_id:
                logging.warning("⚠️ CLOUDFLARE_ACCOUNT_ID is not set; skipping Cloudflare call.")
                return None

            url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_name}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=60) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = ""
                        
                        if "result" in data and isinstance(data["result"], dict) and "response" in data["result"]:
                            content = data["result"]["response"]
                        elif "response" in data:
                            content = data["response"]
                        elif "result" in data and isinstance(data["result"], str):
                            content = data["result"]
                        else:
                            content = json.dumps(data, ensure_ascii=False)
                            
                        if content:
                            return self._parse_ai_response(content, symbol, f"Cloudflare-{model_name}")
                        else:
                            logging.warning(f"⚠️ Empty content in Cloudflare response for {symbol}")
                            return None
                    else:
                        error_text = await response.text()
                        logging.error(f"❗ Cloudflare API error for {symbol}: {response.status} - {error_text}")
                        # Special handling for deprecated model / wrong route
                        if response.status == 410 or "deprecated" in error_text.lower():
                            self.api_manager.mark_model_failed("cloudflare", model_name)
                            logging.warning(f"Model deprecated on Cloudflare: {model_name}; removed from rotation.")
                        if "No route for that URI" in error_text or '"code":7000' in error_text:
                            logging.warning(f"Cloudflare routing/model name issue for {model_name}. Check model id & account_id.")
                        return None
                        
        except Exception as e:
            logging.error(f"❗ Cloudflare/{model_name} analysis error for {symbol}: {str(e)}")
            return None

    async def _get_groq_analysis(self, symbol: str, prompt: str, model_name: str) -> Optional[Dict]:
        """Get analysis from Groq API"""
        if not self.groq_api_key:
            logging.warning(f"⚠️ Groq API key not available for {symbol}")
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
                        "content": "Return ONLY valid JSON format. No additional text."
                    },
                    {"role": "user", "content": prompt}
                ],
                "model": model_name,
                "temperature": 0.1,
                "max_tokens": 600,
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
                            logging.warning(f"⚠️ No choices in Groq response for {symbol}: {data}")
                            return None
                    else:
                        error_text = await response.text()
                        logging.error(f"❗ Groq API error for {symbol}: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            logging.error(f"❗ Groq/{model_name} analysis error for {symbol}: {str(e)}")
            return None

    def _parse_ai_response(self, response, symbol: str, ai_name: str) -> Optional[Dict]:
        """Parse AI response with enhanced validation and text extraction fallback"""
        try:
            if isinstance(response, dict):
                cleaned_response = json.dumps(response, ensure_ascii=False)
            else:
                cleaned_response = (response or "").strip()

            # Clean response
            cleaned_response = re.sub(r'```json\s*', '', cleaned_response)
            cleaned_response = re.sub(r'```\s*', '', cleaned_response)
            cleaned_response = re.sub(r'<think>.*?</think>', '', cleaned_response, flags=re.DOTALL)
            cleaned_response = re.sub(r'</?[^>]+>', '', cleaned_response)

            # Extract JSON
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

            # NEW: If JSON parsing fails, try text extraction
            logging.warning(f"⚠️ {ai_name} response for {symbol} lacks valid JSON format, trying text extraction...")
            return self._extract_signal_from_text(cleaned_response, symbol, ai_name)

        except json.JSONDecodeError as e:
            logging.warning(f"⚠️ JSON error in {ai_name} response for {symbol}, trying text extraction: {e}")
            return self._extract_signal_from_text(cleaned_response, symbol, ai_name)
        except Exception as e:
            logging.error(f"❗ Error parsing {ai_name} response for {symbol}: {str(e)}")
            return None

    def _extract_signal_from_text(self, text: str, symbol: str, ai_name: str) -> Optional[Dict]:
        """NEW: Extract trading signal from unstructured text response"""
        try:
            text_upper = text.upper()
            action = "HOLD"
            
            # Extract action from text
            if "BUY" in text_upper and "SELL" not in text_upper:
                action = "BUY"
            elif "SELL" in text_upper and "BUY" not in text_upper:
                action = "SELL"
            
            # Extract confidence
            confidence_match = re.search(r'CONFIDENCE[:\s]*(\d+)', text_upper)
            confidence = int(confidence_match.group(1)) if confidence_match else 5
            
            # Create basic signal
            signal_data = {
                "SYMBOL": symbol,
                "ACTION": action,
                "CONFIDENCE": confidence,
                "ENTRY": "N/A",
                "STOP_LOSS": "N/A", 
                "TAKE_PROFIT": "N/A",
                "RISK_REWARD_RATIO": "1.8",
                "ANALYSIS": f"Extracted from text: {text[:100]}...",
                "ai_model": ai_name + "-TEXT_EXTRACTED"
            }
            
            logging.info(f"✅ {ai_name} text-extracted signal for {symbol}: {action}")
            return signal_data
            
        except Exception as e:
            logging.warning(f"⚠️ Text extraction failed for {symbol}: {e}")
            return None

    def _validate_signal_data(self, signal_data: Dict, symbol: str) -> bool:
        """Validate signal data"""
        required_fields = ['SYMBOL', 'ACTION', 'CONFIDENCE']
        for field in required_fields:
            if field not in signal_data:
                logging.warning(f"⚠️ Required field {field} missing in signal for {symbol}")
                return False
                
        action = signal_data['ACTION'].upper()
        if action not in ['BUY', 'SELL', 'HOLD']:
            logging.warning(f"⚠️ Invalid ACTION for {symbol}: {action}")
            return False
            
        try:
            confidence = float(signal_data['CONFIDENCE'])
            if not (1 <= confidence <= 10):
                logging.warning(f"⚠️ CONFIDENCE out of range for {symbol}: {confidence}")
                return False
        except (ValueError, TypeError):
            logging.warning(f"⚠️ Invalid CONFIDENCE for {symbol}: {signal_data['CONFIDENCE']}")
            return False
            
        return True

    def _validate_numeric_values(self, signal_data: Dict, symbol: str) -> Dict:
        """Validate and fix numeric values"""
        numeric_fields = ['ENTRY', 'STOP_LOSS', 'TAKE_PROFIT', 'RISK_REWARD_RATIO']
        
        for field in numeric_fields:
            if field in signal_data:
                value = signal_data[field]
                if value is None or value == "null" or str(value).strip() == "":
                    if field == 'RISK_REWARD_RATIO':
                        signal_data[field] = "1.8"
                    else:
                        signal_data[field] = "N/A"
                elif field == 'CONFIDENCE':
                    try:
                        signal_data[field] = float(value)
                    except:
                        signal_data[field] = 5.0
                        
        return signal_data

    def _combine_signals(self, symbol: str, valid_results: List[Dict], total_models: int) -> Optional[Dict]:
        """Combine signal results with robust error handling"""
        if not valid_results:
            return None
            
        action_counts = {}
        confidence_sum = {}
        
        for result in valid_results:
            action = result['ACTION'].upper()
            action_counts[action] = action_counts.get(action, 0) + 1
            
            # FIXED: Better confidence handling
            try:
                confidence_val = float(result.get('CONFIDENCE', 5))
                confidence_sum[action] = confidence_sum.get(action, 0) + confidence_val
            except (ValueError, TypeError):
                confidence_sum[action] = confidence_sum.get(action, 0) + 5  # Default value
        
        logging.info(f"📊 Signal combination for {symbol}: {action_counts}")
        
        total_valid = len(valid_results)
        max_agreement = max(action_counts.values()) if action_counts else 0
        
        if max_agreement >= 3:
            agreement_type = 'STRONG_CONSENSUS'
        elif max_agreement == 2:
            agreement_type = 'MEDIUM_CONSENSUS'
        else:
            agreement_type = 'WEAK_CONSENSUS'
            
        majority_action = max(action_counts, key=action_counts.get) if action_counts else 'HOLD'
        
        # Calculate average confidence for majority action
        avg_confidence = 5.0  # Default
        if majority_action in confidence_sum and max_agreement > 0:
            avg_confidence = confidence_sum[majority_action] / max_agreement
        
        combined = {
            'SYMBOL': symbol,
            'ACTION': majority_action,
            'CONFIDENCE': round(avg_confidence, 1),
            'AGREEMENT_LEVEL': max_agreement,
            'AGREEMENT_TYPE': agreement_type,
            'VALID_MODELS': total_valid,
            'TOTAL_MODELS': total_models,
            'timestamp': datetime.now(UTC).isoformat()
        }
        
        # Add details from the first valid result of majority action
        majority_results = [r for r in valid_results if r['ACTION'].upper() == majority_action]
        if majority_results:
            first_result = majority_results[0]
            for field in ['ENTRY', 'STOP_LOSS', 'TAKE_PROFIT', 'RISK_REWARD_RATIO', 'ANALYSIS']:
                if field in first_result and first_result[field] not in [None, "null", ""]:
                    combined[field] = first_result[field]
        
        # Ensure all required fields have values
        if 'ENTRY' not in combined:
            combined['ENTRY'] = "N/A"
        if 'STOP_LOSS' not in combined:
            combined['STOP_LOSS'] = "N/A"
        if 'TAKE_PROFIT' not in combined:
            combined['TAKE_PROFIT'] = "N/A"
        if 'RISK_REWARD_RATIO' not in combined:
            combined['RISK_REWARD_RATIO'] = "1.8"
        if 'ANALYSIS' not in combined:
            combined['ANALYSIS'] = f"{majority_action} signal based on agreement of {max_agreement} out of {total_models} AI models"
            
        return combined

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
            advanced_indicators = ['SUPERT_7_3.0', 'SUPERTd_7_3.0', 'PSARl_0.02_0.2', 'DCP_20', 'DCM_20', 'DCU_20']
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

            # Ichimoku Cloud (correct column mapping)
            try:
                df_indicators.ta.ichimoku(append=True)
                indicators_added.extend(['ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'ICS_26'])
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
            logging.error(f"❗ Critical error in indicator calculation: {e}")
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
            logging.error(f"❗ Even basic indicators failed: {e}")
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
            logging.error(f"❗ Error generating technical analysis for {symbol}: {e}")
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
            logging.error(f"❗ Even basic analysis failed for {symbol}: {e}")
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

            # Ichimoku analysis (fixed mapping inside helper)
            ichimoku_signal = self._analyze_ichimoku(current)
            
            # SuperTrend signal (use direction column)
            st_dir = current.get('SUPERTd_7_3.0', 0)
            supertrend_signal = "BULLISH" if st_dir > 0 else "BEARISH" if st_dir < 0 else "NEUTRAL"

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
        """Analyze Ichimoku Cloud signals with error handling (correct column names)"""
        try:
            tenkan    = data.get('ITS_9',  data['close'])   # Tenkan
            kijun     = data.get('IKS_26', data['close'])   # Kijun
            senkou_a  = data.get('ISA_9',  data['close'])   # Span A
            senkou_b  = data.get('ISB_26', data['close'])   # Span B
            chikou    = data.get('ICS_26', data['close'])   # Chikou
            price     = data['close']
            
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
# --- Enhanced Trade Filter with Flexible Settings ---
# =================================================================================

class EnhancedTradeFilter:
    """
    Advanced trade filtering with flexible market condition analysis
    """
    def __init__(self):
        self.last_stopout_time = {}
        self.cooldown_minutes = 120
        
        # FLEXIBLE volatility ranges for better signal generation
        self.min_volatility = 0.05    # Reduced from 0.3% - allow lower volatility
        self.max_volatility = 10.0    # Increased from 3.0% - allow higher volatility

    def can_trade(self, symbol: str, technical_analysis: Dict, now: Optional[datetime] = None) -> bool:
        """Check if trading is allowed with flexible volatility ranges"""
        now = now or datetime.now(UTC)
        
        # Cooldown check
        if not self._check_cooldown(symbol, now):
            logging.info(f"🛑 {symbol} in cooldown period")
            return False
            
        # Volatility check (now more flexible)
        volatility_ok, volatility_value = self._check_volatility(technical_analysis)
        if not volatility_ok:
            logging.info(f"📊 {symbol} volatility {volatility_value:.2f}% outside extended range [{self.min_volatility}%-{self.max_volatility}%]")
            return False
            
        # Market hours consideration (optional)
        if not self._check_market_hours(now):
            logging.warning(f"⏰ {symbol} outside optimal trading hours - but continuing")
            # return False  # Comment out for more signals
            
        # Trend strength check (relaxed)
        if not self._check_trend_strength(technical_analysis):
            logging.warning(f"ℹ️ {symbol} trend weak but continuing analysis")
            # return False  # Comment out for more signals
            
        logging.info(f"✅ Trade filter PASSED for {symbol} (volatility: {volatility_value:.2f}%)")
        return True

    def _check_cooldown(self, symbol: str, now: datetime) -> bool:
        """Check if symbol is in cooldown period"""
        last_stopout = self.last_stopout_time.get(symbol)
        if last_stopout and (now - last_stopout).total_seconds() < self.cooldown_minutes * 60:
            return False
        return True

    def _check_volatility(self, technical_analysis: Dict) -> Tuple[bool, float]:
        """Check if volatility is within extended acceptable range"""
        volatility = technical_analysis.get('risk_assessment', {}).get('volatility_percent', 0)
        return self.min_volatility <= volatility <= self.max_volatility, volatility

    def _check_market_hours(self, now: datetime) -> bool:
        """Check if current time is within optimal trading hours - more flexible"""
        hour = now.hour
        # Extended trading hours
        return 0 <= hour <= 24  # All hours

    def _check_trend_strength(self, technical_analysis: Dict) -> bool:
        """More flexible trend strength check"""
        trend_strength = technical_analysis.get('htf_trend', {}).get('strength', 'WEAK')
        # Allow all trend strengths
        return True

    def mark_stopout(self, symbol: str):
        """Record stopout for cooldown period"""
        self.last_stopout_time[symbol] = datetime.now(UTC)
        logging.info(f"💥 Stopout recorded for {symbol}, cooldown activated")

# =================================================================================
# --- Advanced Risk Manager ---
# =================================================================================

class AdvancedRiskManager:
    """
    Advanced position sizing and risk management
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

    def calculate_intelligent_stop_loss(self, action: str, current_price: float, 
                                      technical_analysis: Dict, atr: float) -> float:
        """Calculate intelligent stop loss using multiple methods"""
        key_levels = technical_analysis.get('key_levels', {})
        
        if action == "BUY":
            # Method 1: Below nearest support
            sl_support = key_levels.get('support_1', current_price * 0.99)
            
            # Method 2: ATR-based (2 ATR below)
            sl_atr = current_price - (2 * atr)
            
            # Choose the most conservative (highest) stop loss for BUY
            stop_loss = max(sl_support, sl_atr)
            
        else:  # SELL
            # Method 1: Above nearest resistance
            sl_resistance = key_levels.get('resistance_1', current_price * 1.01)
            
            # Method 2: ATR-based (2 ATR above)
            sl_atr = current_price + (2 * atr)
            
            # Choose the most conservative (lowest) stop loss for SELL
            stop_loss = min(sl_resistance, sl_atr)
            
        return round(stop_loss, 5)

    def calculate_intelligent_take_profit(self, action: str, entry_price: float, stop_loss: float,
                                        technical_analysis: Dict, atr: float) -> float:
        """NEW: Calculate intelligent take profit with minimum 1.5:1 RR ratio"""
        key_levels = technical_analysis.get('key_levels', {})
        risk_amount = abs(entry_price - stop_loss)
        
        # Ensure minimum risk-reward ratio of 1.5:1
        min_rr_ratio = 1.5
        
        if action == "BUY":
            # Method 1: Risk-Reward Ratio with minimum
            tp_rr = entry_price + (risk_amount * min_rr_ratio)
            
            # Method 2: Key resistance level
            tp_resistance = key_levels.get('resistance_1', entry_price * 1.03)
            
            # Method 3: ATR-based (3 ATR above)
            tp_atr = entry_price + (3 * atr)
            
            # Choose the most conservative that meets minimum RR
            candidate_tps = [tp for tp in [tp_rr, tp_resistance, tp_atr] 
                            if (tp - entry_price) >= (risk_amount * min_rr_ratio)]
            
            take_profit = min(candidate_tps) if candidate_tps else tp_rr
            
        else:  # SELL
            # Method 1: Risk-Reward Ratio with minimum
            tp_rr = entry_price - (risk_amount * min_rr_ratio)
            
            # Method 2: Key support level
            tp_support = key_levels.get('support_1', entry_price * 0.97)
            
            # Method 3: ATR-based (3 ATR below)
            tp_atr = entry_price - (3 * atr)
            
            # Choose the most conservative that meets minimum RR
            candidate_tps = [tp for tp in [tp_rr, tp_support, tp_atr] 
                            if (entry_price - tp) >= (risk_amount * min_rr_ratio)]
            
            take_profit = max(candidate_tps) if candidate_tps else tp_rr
        
        return round(take_profit, 5)

    def validate_signal_risk(self, signal: Dict, technical_analysis: Dict) -> Dict:
        """Validate and enhance signal with proper risk management"""
        if not signal or signal.get('ACTION') == 'HOLD':
            return signal
            
        current_price = technical_analysis.get('current_price', 0)
        atr = technical_analysis.get('volatility', 0.001) or 0.001
        action = signal.get('ACTION')
        
        # Calculate intelligent stop loss and take profit
        stop_loss = self.calculate_intelligent_stop_loss(action, current_price, technical_analysis, atr)
        take_profit = self.calculate_intelligent_take_profit(action, current_price, stop_loss, technical_analysis, atr)
        
        # Update signal with calculated levels
        signal['STOP_LOSS'] = f"{stop_loss:.5f}"
        signal['TAKE_PROFIT'] = f"{take_profit:.5f}"
        
        # Calculate actual risk-reward ratio
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        actual_rr = reward / risk if risk > 0 else 1.8
        signal['ACTUAL_RR_RATIO'] = round(actual_rr, 2)
        # Reflect in RISK_REWARD_RATIO too for transparency
        signal['RISK_REWARD_RATIO'] = str(round(actual_rr, 2))
        
        return signal

# =================================================================================
# --- Signal Quality Scorer ---
# =================================================================================

class SignalQualityScorer:
    """Evaluate signal quality based on multiple factors"""
    
    def __init__(self):
        self.weights = {
            'ai_agreement': 0.25,
            'technical_alignment': 0.20,
            'risk_reward': 0.15,
            'trend_strength': 0.15,
            'volatility_appropriateness': 0.10,
            'momentum_confirmation': 0.10,
            'market_structure': 0.05
        }
    
    def calculate_signal_score(self, signal: Dict, technical_analysis: Dict) -> float:
        """Calculate comprehensive signal quality score (0-100)"""
        if signal.get('ACTION') == 'HOLD':
            return 0.0
            
        scores = {}
        
        # 1. AI Agreement Score
        agreement_level = signal.get('AGREEMENT_LEVEL', 0)
        total_models = signal.get('TOTAL_MODELS', 1)
        scores['ai_agreement'] = (agreement_level / total_models) * 100
        
        # 2. Technical Alignment Score
        scores['technical_alignment'] = self._calculate_technical_alignment(signal, technical_analysis)
        
        # 3. Risk-Reward Score
        rr_ratio = float(signal.get('ACTUAL_RR_RATIO', 1.0))
        scores['risk_reward'] = min(rr_ratio * 33.33, 100)
        
        # 4. Trend Strength Score
        trend_strength = technical_analysis.get('htf_trend', {}).get('strength', 'WEAK')
        strength_scores = {'VERY_STRONG': 100, 'STRONG': 80, 'MODERATE': 60, 'WEAK': 30, 'UNKNOWN': 0}
        scores['trend_strength'] = strength_scores.get(trend_strength, 0)
        
        # 5. Volatility Appropriateness Score
        scores['volatility_appropriateness'] = self._calculate_volatility_score(technical_analysis)
        
        # 6. Momentum Confirmation Score
        scores['momentum_confirmation'] = self._calculate_momentum_score(signal, technical_analysis)
        
        # 7. Market Structure Score
        scores['market_structure'] = self._calculate_market_structure_score(technical_analysis)
        
        # Calculate weighted total score
        total_score = 0
        for factor, weight in self.weights.items():
            total_score += scores.get(factor, 0) * weight
            
        return round(total_score, 1)
    
    def _calculate_technical_alignment(self, signal: Dict, technical_analysis: Dict) -> float:
        """Calculate technical alignment score"""
        action = signal.get('ACTION')
        trend_direction = technical_analysis.get('htf_trend', {}).get('direction', 'NEUTRAL')
        momentum_bias = technical_analysis.get('momentum', {}).get('overall_bias', 'NEUTRAL')
        market_structure = technical_analysis.get('market_structure', {}).get('higher_timeframe_structure', 'UNKNOWN')
        
        alignment_score = 0
        
        # Trend alignment (40 points max)
        if (action == 'BUY' and 'BULLISH' in trend_direction) or (action == 'SELL' and 'BEARISH' in trend_direction):
            alignment_score += 40
        elif trend_direction == 'NEUTRAL':
            alignment_score += 20
        else:  # Counter-trend
            alignment_score += 10
            
        # Momentum alignment (30 points max)
        if (action == 'BUY' and momentum_bias == 'BULLISH') or (action == 'SELL' and momentum_bias == 'BEARISH'):
            alignment_score += 30
        elif momentum_bias == 'NEUTRAL':
            alignment_score += 15
            
        # Market structure alignment (30 points max)
        if (action == 'BUY' and market_structure == 'UPTREND') or (action == 'SELL' and market_structure == 'DOWNTREND'):
            alignment_score += 30
        elif market_structure == 'RANGING':
            alignment_score += 20
        elif market_structure == 'UNKNOWN':
            alignment_score += 10
            
        return min(alignment_score, 100)
    
    def _calculate_volatility_score(self, technical_analysis: Dict) -> float:
        """NEW: Calculate volatility appropriateness score with better ranges"""
        volatility = technical_analysis.get('risk_assessment', {}).get('volatility_percent', 0)
        
        # More realistic volatility ranges for forex
        if 0.1 <= volatility <= 2.0:  # Good volatility for trading
            return 100
        elif 0.05 <= volatility < 0.1 or 2.0 < volatility <= 3.0:  # Acceptable but not ideal
            return 70
        elif 0.02 <= volatility < 0.05 or 3.0 < volatility <= 5.0:  # Marginal
            return 40
        else:  # Too low or too high
            return 20
    
    def _calculate_momentum_score(self, signal: Dict, technical_analysis: Dict) -> float:
        """Calculate momentum confirmation score"""
        momentum = technical_analysis.get('momentum', {})
        action = signal.get('ACTION')
        
        score = 50  # Base score
        
        # RSI alignment
        rsi_signal = momentum.get('rsi', {}).get('signal', 'NEUTRAL')
        if (action == 'BUY' and rsi_signal == 'OVERSOLD') or (action == 'SELL' and rsi_signal == 'OVERBOUGHT'):
            score += 20
        elif rsi_signal == 'NEUTRAL':
            score += 10
            
        # MACD alignment
        macd_trend = momentum.get('macd', {}).get('trend', 'NEUTRAL')
        if (action == 'BUY' and macd_trend == 'BULLISH') or (action == 'SELL' and macd_trend == 'BEARISH'):
            score += 15
            
        return min(score, 100)
    
    def _calculate_market_structure_score(self, technical_analysis: Dict) -> float:
        """Calculate market structure score"""
        market_structure = technical_analysis.get('market_structure', {})
        structure = market_structure.get('higher_timeframe_structure', 'UNKNOWN')
        is_breaking = market_structure.get('is_breaking_structure', False)
        
        if structure in ['UPTREND', 'DOWNTREND'] and not is_breaking:
            return 100
        elif structure in ['UPTREND', 'DOWNTREND'] and is_breaking:
            return 60
        elif structure == 'RANGING':
            return 40
        else:
            return 20

# =================================================================================
# --- Gemini Direct Signal Agent (Enhanced) ---
# =================================================================================

class GeminiDirectSignalAgent:
    """
    Enhanced direct Gemini signal agent with improved error handling
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = GEMINI_MODEL):
        self.model_name = model_name
        self.fallback_model = GEMINI_FALLBACK_MODEL
        self.available = False
        try:
            k = api_key or os.getenv("GOOGLE_API_KEY")
            if k:
                genai.configure(api_key=k)
                self.available = True
                logging.info(f"✅ GeminiDirectSignalAgent initialized with {model_name}")
            else:
                logging.warning("⚠️ GeminiDirectSignalAgent: GOOGLE_API_KEY not set.")
        except Exception as e:
            logging.error(f"❗ GeminiDirectSignalAgent init error: {e}")

    def _create_enhanced_prompt(self, symbol: str, ta: Dict) -> str:
        """Create enhanced prompt for better signal generation"""
        price = ta.get("current_price", 1.0)
        trend = ta.get("htf_trend", {})
        risk = ta.get("risk_assessment", {})
        ml = ta.get("ml_signal", {})
        key = ta.get("key_levels", {})
        momentum = ta.get("momentum", {})
        
        return f"""As an expert forex trading analyst, analyze {symbol} and provide ONLY JSON output.

CRITICAL: Be DECISIVE - prefer BUY/SELL over HOLD unless market is completely unclear.

MARKET DATA:
- Price: {price:.5f}
- Trend: {trend.get('direction', 'NEUTRAL')} ({trend.get('strength', 'UNKNOWN')})
- RSI: {momentum.get('rsi', {}).get('value', 50):.1f} ({momentum.get('rsi', {}).get('signal', 'NEUTRAL')})
- Key Support: {key.get('support_1', price*0.99):.5f}
- Key Resistance: {key.get('resistance_1', price*1.01):.5f}
- Volatility: {risk.get('volatility_percent', 0):.2f}%
- Risk Level: {risk.get('risk_level', 'MEDIUM')}
- ML Signal Strength: {ml.get('signal_strength', 0):.2f}

TRADING RULES:
1. Only use HOLD if trend is completely neutral and no clear direction
2. BUY if trend is bullish and RSI < 70
3. SELL if trend is bearish and RSI > 30
4. Minimum risk-reward ratio: 1.5

Return EXACT JSON format:
{{
  "SYMBOL": "{symbol}",
  "ACTION": "BUY|SELL|HOLD",
  "CONFIDENCE": 1-10,
  "ENTRY": "exact_price",
  "STOP_LOSS": "exact_price",
  "TAKE_PROFIT": "exact_price", 
  "RISK_REWARD_RATIO": "number",
  "ANALYSIS": "brief_technical_reasoning",
  "EXPIRATION_H": 4,
  "TRADE_RATIONALE": "detailed_explanation"
}}"""

    async def fetch_signal(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        """Fetch direct signal from Gemini with enhanced reliability"""
        if not self.available:
            return None
            
        try:
            prompt = self._create_enhanced_prompt(symbol, technical_analysis)
            
            # Try primary model first
            model = genai.GenerativeModel(self.model_name)
            
            try:
                response = await asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=600,
                    )
                )
                
                text = self._extract_text(response)
                data = self._clean_parse(text)
                if data and self._validate_direct_signal(data, symbol):
                    logging.info(f"✅ GeminiDirect signal for {symbol}: {data.get('ACTION', 'HOLD')}")
                    return data
                    
            except Exception as e:
                logging.warning(f"⚠️ GeminiDirect primary model failed: {e}")
                # Try fallback model
                if self.model_name != self.fallback_model:
                    logging.info(f"🧪 Trying fallback model: {self.fallback_model}")
                    model = genai.GenerativeModel(self.fallback_model)
                    response = await asyncio.to_thread(
                        model.generate_content,
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.1,
                            max_output_tokens=600,
                        )
                    )
                    text = self._extract_text(response)
                    data = self._clean_parse(text)
                    if data and self._validate_direct_signal(data, symbol):
                        logging.info(f"✅ GeminiDirect (fallback) signal for {symbol}: {data.get('ACTION', 'HOLD')}")
                        return data

            return None
            
        except Exception as e:
            logging.error(f"❗ GeminiDirectSignalAgent error for {symbol}: {e}")
            return None

    def _extract_text(self, resp) -> Optional[str]:
        """Enhanced text extraction from Gemini response"""
        try:
            # Method 1: Direct text attribute
            if hasattr(resp, 'text') and resp.text:
                return resp.text
                
            # Method 2: Candidates with parts
            if hasattr(resp, 'candidates') and resp.candidates:
                for candidate in resp.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        text_parts = []
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                text_parts.append(part.text)
                        if text_parts:
                            return ''.join(text_parts)
                            
            # Method 3: Try to_dict method
            if hasattr(resp, 'to_dict'):
                resp_dict = resp.to_dict()
                if 'candidates' in resp_dict:
                    for candidate in resp_dict['candidates']:
                        if 'content' in candidate and 'parts' in candidate['content']:
                            text_parts = []
                            for part in candidate['content']['parts']:
                                if 'text' in part:
                                    text_parts.append(part['text'])
                            if text_parts:
                                return ''.join(text_parts)
                                
            return None
            
        except Exception as e:
            logging.warning(f"Error extracting Gemini response text: {e}")
            return None

    def _clean_parse(self, text: Optional[str]) -> Optional[Dict]:
        """Enhanced JSON parsing with validation"""
        if not text:
            return None
            
        try:
            # Clean the text
            cleaned = text.strip()
            cleaned = re.sub(r'```json\s*', '', cleaned)
            cleaned = re.sub(r'```\s*', '', cleaned)
            cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            cleaned = re.sub(r'</?[^>]+>', '', cleaned)
            
            # Find JSON
            if '{' in cleaned:
                cleaned = cleaned[cleaned.find('{'):]
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if not json_match:
                return None
                
            data = json.loads(json_match.group(0))
            return data
            
        except Exception as e:
            logging.warning(f"GeminiDirect JSON parsing error: {e}")
            return None

    def _validate_direct_signal(self, signal_data: Dict, symbol: str) -> bool:
        """Validate direct signal data"""
        required = ['SYMBOL', 'ACTION', 'CONFIDENCE']
        for field in required:
            if field not in signal_data:
                logging.warning(f"⚠️ GeminiDirect missing field {field} for {symbol}")
                return False
                
        action = signal_data['ACTION'].upper()
        if action not in ['BUY', 'SELL', 'HOLD']:
            logging.warning(f"⚠️ GeminiDirect invalid ACTION for {symbol}: {action}")
            return False
            
        try:
            confidence = float(signal_data['CONFIDENCE'])
            if not (1 <= confidence <= 10):
                logging.warning(f"⚠️ GeminiDirect CONFIDENCE out of range for {symbol}: {confidence}")
                return False
        except (ValueError, TypeError):
            logging.warning(f"⚠️ GeminiDirect invalid CONFIDENCE for {symbol}: {signal_data['CONFIDENCE']}")
            return False
            
        return True

# =================================================================================
# --- Main Forex Analyzer Class (Fully Enhanced) ---
# =================================================================================

class ImprovedForexAnalyzer:
    def __init__(self, strict_filters: bool = False):
        self.model_discoverer = AdvancedDynamicModelDiscoverer()
        self.api_manager = SmartAPIManager(USAGE_TRACKER_FILE, self.model_discoverer)
        self.technical_analyzer = AdvancedTechnicalAnalyzer()
        self.ai_manager = EnhancedAIManager(google_api_key, CLOUDFLARE_AI_API_KEY, GROQ_API_KEY, self.api_manager)
        self.data_fetcher = EnhancedDataFetcher()
        self.performance_monitor = PerformanceMonitor()

        # Enhanced components
        self.gemini_direct = GeminiDirectSignalAgent(google_api_key, GEMINI_MODEL)
        self.risk_manager = AdvancedRiskManager()
        self.signal_scorer = SignalQualityScorer()
        self.trade_filter = EnhancedTradeFilter()
        self.strict_filters = strict_filters

    async def analyze_pair(self, pair: str) -> Optional[Dict]:
        """Complete analysis with flexible filtering and robust fallbacks"""
        logging.info(f"🚀 Starting enhanced analysis for {pair}")
        start_time = time.time()
        
        try:
            logging.info(self.api_manager.get_usage_summary())
            
            # Get market data
            htf_df = await self.data_fetcher.get_market_data(pair, HIGH_TIMEFRAME)
            ltf_df = await self.data_fetcher.get_market_data(pair, LOW_TIMEFRAME)
            
            if htf_df is None or ltf_df is None:
                logging.warning(f"⚠️ Market data retrieval failed for {pair}")
                self.performance_monitor.record_failure()
                return None
                
            logging.info(f"✅ Retrieved data: HTF={len(htf_df)} rows, LTF={len(ltf_df)} rows")
            
            # Technical analysis
            htf_df_processed = self.technical_analyzer.calculate_enhanced_indicators(htf_df)
            ltf_df_processed = self.technical_analyzer.calculate_enhanced_indicators(ltf_df)
            
            if htf_df_processed is None or ltf_df_processed is None:
                logging.warning(f"⚠️ Technical analysis failed for {pair}")
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

            # Trade filter check (flexible)
            if self.strict_filters:
                if not self.trade_filter.can_trade(pair, technical_analysis):
                    logging.info(f"🛑 Trade filter blocked {pair}")
                    self.performance_monitor.record_success()
                    return None
                volatility_ok = True
            else:
                # Just log the volatility info but don't block
                volatility_ok, volatility_value = self.trade_filter._check_volatility(technical_analysis)
                if not volatility_ok:
                    logging.warning(f"⚠️ {pair} volatility {volatility_value:.2f}% outside optimal range but continuing...")
                else:
                    logging.info(f"✅ {pair} volatility {volatility_value:.2f}% within acceptable range")
                
            # AI analysis (ensemble with robust fallbacks)
            ai_analysis = await self.ai_manager.get_enhanced_ai_analysis(pair, technical_analysis)

            # Gemini direct as additional backup
            if not ai_analysis and self.gemini_direct.available:
                logging.info(f"🧪 Trying GeminiDirect as additional backup for {pair}")
                direct_signal = await self.gemini_direct.fetch_signal(pair, technical_analysis)
                if direct_signal:
                    ai_analysis = direct_signal
                    ai_analysis['AGREEMENT_LEVEL'] = 1
                    ai_analysis['AGREEMENT_TYPE'] = 'DIRECT_SIGNAL'
                    ai_analysis['VALID_MODELS'] = 1
                    ai_analysis['TOTAL_MODELS'] = 1

            if ai_analysis:
                # Enhanced risk management and position sizing
                ai_analysis = self.risk_manager.validate_signal_risk(ai_analysis, technical_analysis)
                
                # Calculate signal quality score
                quality_score = self.signal_scorer.calculate_signal_score(ai_analysis, technical_analysis)
                ai_analysis['QUALITY_SCORE'] = quality_score
                
                # Add technical context
                ai_analysis['TECHNICAL_CONTEXT'] = {
                    'trend': technical_analysis.get('htf_trend', {}).get('direction', 'NEUTRAL'),
                    'rsi': technical_analysis.get('momentum', {}).get('rsi', {}).get('value', 50),
                    'key_support': technical_analysis.get('key_levels', {}).get('support_1', 0),
                    'key_resistance': technical_analysis.get('key_levels', {}).get('resistance_1', 0),
                    'volatility': technical_analysis.get('risk_assessment', {}).get('volatility_percent', 0),
                    'volatility_status': 'OPTIMAL' if (self.strict_filters or volatility_ok) else 'EXTENDED_RANGE'
                }
                
                analysis_duration = time.time() - start_time
                self.performance_monitor.record_analysis_time(pair, analysis_duration)
                self.performance_monitor.record_success()
                
                logging.info(f"✅ Enhanced signal for {pair}: {ai_analysis['ACTION']} "
                           f"(Quality: {quality_score}/100, Agreement: {ai_analysis.get('AGREEMENT_LEVEL', 0)}/{ai_analysis.get('TOTAL_MODELS', 0)})")
                return ai_analysis
                
            self.performance_monitor.record_failure()
            logging.info(f"ℹ️ No trading signal for {pair}")
            return None
            
        except Exception as e:
            self.performance_monitor.record_failure()
            logging.error(f"❗ Error analyzing {pair}: {str(e)}")
            logging.error(f"❗ Traceback: {traceback.format_exc()}")
            return None

    async def analyze_all_pairs(self, pairs: List[str]) -> List[Dict]:
        """Analyze all currency pairs with enhanced processing"""
        logging.info(f"🗂️ Starting enhanced analysis for {len(pairs)} currency pairs")
        
        # Initialize models first
        await self.api_manager.initialize_models()
        
        # Disable Gemini if there are quota issues (uncomment if needed)
        # self.api_manager.disable_gemini()
        
        tasks = [self.analyze_pair(pair) for pair in pairs]
        results = await asyncio.gather(*tasks)
        
        valid_signals = [r for r in results if r is not None]
        
        # Sort by quality score
        valid_signals.sort(key=lambda x: x.get('QUALITY_SCORE', 0), reverse=True)
        
        logging.info(f"📊 Enhanced analysis complete. {len(valid_signals)} valid signals")
        
        # Log performance statistics
        perf_stats = self.performance_monitor.get_performance_stats()
        logging.info(f"📈 Performance Statistics: {json.dumps(perf_stats, indent=2)}")
        
        return valid_signals

    def save_signals(self, signals: List[Dict]):
        """Save signals to files with enhanced categorization"""
        import os
        
        current_dir = os.getcwd()
        logging.info(f"📁 Current directory for file saving: {current_dir}")
        
        if not signals:
            logging.info("🗒️ No signals to save")
            # Create empty files
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
                    logging.info(f"📝 Empty file created: {filename}")
            except Exception as e:
                logging.error(f"❗ Error creating empty files: {e}")
            return

        # Enhanced categorization with quality scoring
        strong_signals = []
        medium_signals = []
        weak_signals = []
        
        for signal in signals:
            agreement_type = signal.get('AGREEMENT_TYPE', '')
            quality_score = signal.get('QUALITY_SCORE', 0)
            
            # Enhanced categorization logic
            if (agreement_type == 'STRONG_CONSENSUS' and quality_score >= 70) or quality_score >= 80:
                strong_signals.append(signal)
            elif (agreement_type == 'MEDIUM_CONSENSUS' and quality_score >= 50) or quality_score >= 60:
                medium_signals.append(signal)
            else:
                weak_signals.append(signal)
                
        # Save to files with enhanced information
        try:
            # Strong signals
            with open("strong_consensus_signals.json", 'w', encoding='utf-8') as f:
                json.dump(strong_signals, f, indent=2, ensure_ascii=False)
            logging.info(f"📝 {len(strong_signals)} strong signals saved")
            
            # Medium signals  
            with open("medium_consensus_signals.json", 'w', encoding='utf-8') as f:
                json.dump(medium_signals, f, indent=2, ensure_ascii=False)
            logging.info(f"📝 {len(medium_signals)} medium signals saved")
            
            # Weak signals
            with open("weak_consensus_signals.json", 'w', encoding='utf-8') as f:
                json.dump(weak_signals, f, indent=2, ensure_ascii=False)
            logging.info(f"📝 {len(weak_signals)} weak signals saved")
            
            # Create summary file
            summary = {
                "timestamp": datetime.now(UTC).isoformat(),
                "total_signals": len(signals),
                "strong_signals": len(strong_signals),
                "medium_signals": len(medium_signals),
                "weak_signals": len(weak_signals),
                "performance_stats": self.performance_monitor.get_performance_stats()
            }
            
            with open("analysis_summary.json", 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logging.info("📝 Analysis summary saved")
            
            # Verify file creation
            for filename in ["strong_consensus_signals.json", "medium_consensus_signals.json", 
                           "weak_consensus_signals.json", "analysis_summary.json"]:
                if os.path.exists(filename):
                    logging.info(f"✅ File {filename} successfully created")
                else:
                    logging.error(f"❗ File {filename} not created!")
                    
        except Exception as e:
            logging.error(f"❗ Error saving signals: {e}")

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
    logging.info("🧠 Starting Enhanced Forex Analysis System (Advanced AI Engine)")
    
    # Install required packages
    install_required_packages()
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Forex Analysis System with AI')
    parser.add_argument("--pair", type=str, help="Analyze specific currency pair")
    parser.add_argument("--all", action="store_true", help="Analyze all currency pairs") 
    parser.add_argument("--pairs", type=str, help="Analyze specified currency pairs")
    parser.add_argument("--equity", type=float, default=10000.0, help="Account equity for sizing (USD)")
    parser.add_argument("--risk_pct", type=float, default=1.0, help="Risk per trade percent (default 1.0)")
    parser.add_argument("--kelly_cap", type=float, default=0.5, help="Kelly cap (0..1, default 0.5)")
    parser.add_argument("--max_lev", type=float, default=30.0, help="Max leverage (default 30)")
    parser.add_argument("--strict", action="store_true", help="Use strict trade filters")
    
    args = parser.parse_args()
    
    if args.pair:
        pairs_to_analyze = [args.pair]
    elif args.pairs:
        pairs_to_analyze = [p.strip() for p in args.pairs.split(',')]
    elif args.all:
        pairs_to_analyze = CURRENCY_PAIRS_TO_ANALYZE
    else:
        pairs_to_analyze = CURRENCY_PAIRS_TO_ANALYZE[:4]  # Default to first 4 pairs
        logging.info(f"🧪 Using default currency pairs: {', '.join(pairs_to_analyze)}")
    
    logging.info(f"🧠 Currency pairs to analyze: {', '.join(pairs_to_analyze)}")
    
    # Use strict filters only if explicitly requested
    analyzer = ImprovedForexAnalyzer(strict_filters=args.strict)

    # Configure risk manager with CLI arguments
    analyzer.risk_manager.equity = args.equity
    analyzer.risk_manager.risk_per_trade_pct = args.risk_pct
    analyzer.risk_manager.kelly_cap = args.kelly_cap
    analyzer.risk_manager.max_leverage = args.max_lev

    logging.info(f"✨ Risk Configuration: Equity=${args.equity:.0f}, Risk={args.risk_pct}%, "
               f"Kelly Cap={args.kelly_cap}, Max Leverage={args.max_lev}x")
    logging.info(f"✨ Filter Mode: {'STRICT' if args.strict else 'FLEXIBLE'}")

    # Optional: Disable Gemini if having quota issues
    # analyzer.api_manager.disable_gemini()
    # logging.info("🚫 Gemini disabled due to quota issues")

    signals = await analyzer.analyze_all_pairs(pairs_to_analyze)
    
    # Save signals
    analyzer.save_signals(signals)
    
    # Display results
    logging.info("📊 Enhanced Results Summary:")
    strong_count = len([s for s in signals if s.get('AGREEMENT_TYPE') == 'STRONG_CONSENSUS' or s.get('QUALITY_SCORE', 0) >= 70])
    medium_count = len([s for s in signals if s.get('AGREEMENT_TYPE') == 'MEDIUM_CONSENSUS' or 50 <= s.get('QUALITY_SCORE', 0) < 70])
    weak_count = len([s for s in signals if s.get('AGREEMENT_TYPE') == 'WEAK_CONSENSUS' or s.get('QUALITY_SCORE', 0) < 50])
    
    avg_quality = sum(s.get('QUALITY_SCORE', 0) for s in signals) / len(signals) if signals else 0
    
    logging.info(f"🟢 Strong signals: {strong_count}")
    logging.info(f"🟡 Medium signals: {medium_count}") 
    logging.info(f"🔴 Weak signals: {weak_count}")
    logging.info(f"📈 Average Quality Score: {avg_quality:.1f}/100")
    
    for signal in signals:
        action_icon = "🟢" if signal['ACTION'] == 'BUY' else "🔴" if signal['ACTION'] == 'SELL' else "🟦"
        quality_score = signal.get('QUALITY_SCORE', 0)
        quality_icon = "🔥" if quality_score >= 80 else "✅" if quality_score >= 60 else "⚠️"
        
        logging.info(f"  {action_icon} {quality_icon} {signal['SYMBOL']}: {signal['ACTION']} "
                   f"(Quality: {quality_score}/100, Conf: {signal.get('CONFIDENCE', 0)}/10)"
                   f" | Entry: {signal.get('ENTRY', 'N/A')} | SL: {signal.get('STOP_LOSS', 'N/A')} | TP: {signal.get('TAKE_PROFIT', 'N/A')}"
                   f" | RR: {signal.get('ACTUAL_RR_RATIO', 'N/A')}:1")
    
    # Display enhanced statistics
    data_source_stats = analyzer.data_fetcher.get_data_source_stats()
    logging.info("📦 Data Source Statistics:")
    for source, count in data_source_stats.items():
        logging.info(f"  {source}: {count} pairs")
    
    
    # Performance statistics
    perf_stats = analyzer.performance_monitor.get_performance_stats()
    logging.info("⚙️ Enhanced Performance Statistics:")
    logging.info(f"  Total Analyses: {perf_stats['total_analyses']}")
    logging.info(f"  Success Rate: {perf_stats['success_rate']}%")
    logging.info(f"  Avg Analysis Time: {perf_stats['avg_analysis_time_sec']}s")
    logging.info(f"  Avg API Response Time: {perf_stats['avg_api_response_time_sec']}s")
    
    # Final API status
    analyzer.api_manager.save_usage_data()
    logging.info(analyzer.api_manager.get_usage_summary())
    
    if signals:
        logging.info("🏁 Enhanced system execution completed successfully with signals!")
    else:
        logging.info("🏁 Enhanced system executed but no signals generated")

if __name__ == "__main__":
    asyncio.run(main())           
