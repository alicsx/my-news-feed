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

# Enhanced Model Configuration - More Diverse Models
GEMINI_FREE_MODELS = [
    'gemini-1.5-flash',
    'gemini-1.5-pro'
]

# Enhanced Cloudflare models with more diversity
CLOUDFLARE_MODELS = [
    "@cf/meta/llama-4-scout-17b-16e-instruct",
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast", 
    "@cf/meta/llama-3.1-8b-instruct-fast",
    "@cf/google/gemma-3-12b-it",
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
    "@cf/qwen/qwen1.5-14b-chat-awq",
    "@cf/deepseek-ai/deepseek-math-7b-instruct"
]

# Enhanced Groq models with more diversity
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b", 
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "deepseek-r1-distill-qwen-32b",
    "deepseek-r1-distill-llama-70b"
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
# --- NEW: Enhanced Model Diversity System ---
# =================================================================================

class ModelDiversityManager:
    """Manage model diversity across different providers and architectures"""
    
    def __init__(self):
        self.model_categories = {
            "large_llama": ["llama-3.3-70b", "llama-4-scout", "llama-4-maverick"],
            "medium_llama": ["llama-3.1-8b", "llama-3.3-70b-instruct"],
            "qwen_models": ["qwen3-32b", "qwen1.5", "qwen2.5"],
            "gemini_models": ["gemini-1.5-flash", "gemini-1.5-pro"],
            "mistral_models": ["mistral-small", "mistral-7b"],
            "deepseek_models": ["deepseek-r1", "deepseek-math", "deepseek-coder"],
            "gemma_models": ["gemma2", "gemma-3"],
            "other_models": ["mixtral", "claude", "command-r"]
        }
        
    def get_model_category(self, model_name: str) -> str:
        """Categorize model by architecture/family"""
        model_lower = model_name.lower()
        for category, patterns in self.model_categories.items():
            for pattern in patterns:
                if pattern in model_lower:
                    return category
        return "other_models"
    
    def ensure_diversity(self, selected_models: List[Tuple[str, str]], target_count: int = 5) -> List[Tuple[str, str]]:
        """Ensure selected models represent different architectures"""
        if len(selected_models) >= target_count:
            return selected_models
            
        # Analyze current diversity
        current_categories = set()
        for provider, model in selected_models:
            category = self.get_model_category(model)
            current_categories.add(category)
        
        # If we have good diversity already, return
        if len(current_categories) >= 3 and len(selected_models) >= target_count - 1:
            return selected_models
            
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
            # Only allow free tier models (Gemini 1.5 Flash and Pro are free)
            if any(free_model in model.lower() for free_model in ['gemini-1.5-flash', 'gemini-1.5-pro']):
                free_models.append(model)
        
        # If no free models found, use our predefined free models
        if not free_models:
            free_models = GEMINI_FREE_MODELS.copy()
            
        logging.info(f"🎯 Free tier Gemini models available: {free_models}")
        return free_models
    
    @staticmethod
    def is_free_tier_model(model_name: str) -> bool:
        """Check if a model is in free tier"""
        return any(free_model in model_name.lower() for free_model in ['gemini-1.5-flash', 'gemini-1.5-pro'])

# =================================================================================
# --- Enhanced Dynamic Model Discovery System ---
# =================================================================================

class EnhancedDynamicModelDiscoverer:
    """Discover available models from AI providers dynamically with enhanced diversity"""
    
    def __init__(self):
        self.available_models = {
            "google_gemini": [],
            "cloudflare": [],
            "groq": []
        }
        self.fallback_models = {
            "google_gemini": GEMINI_FREE_MODELS,
            "cloudflare": CLOUDFLARE_MODELS,
            "groq": GROQ_MODELS
        }
        self.model_filter = FreeTierModelFilter()
        self.diversity_manager = ModelDiversityManager()
        
    async def discover_models(self) -> Dict[str, List[str]]:
        """Discover available models from all providers with enhanced diversity"""
        discovery_tasks = [
            self._discover_gemini_models(),
            self._discover_cloudflare_models(),
            self._discover_groq_models()
        ]
        
        results = await asyncio.gather(*discovery_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            provider = list(self.available_models.keys())[i]
            if isinstance(result, Exception):
                logging.warning(f"❌ Model discovery failed for {provider}: {result}")
                # Use fallback models
                self.available_models[provider] = self.fallback_models[provider]
            elif result:
                self.available_models[provider] = result
            else:
                self.available_models[provider] = self.fallback_models[provider]
                
        # Apply free tier filter for Gemini
        self.available_models["google_gemini"] = self.model_filter.filter_gemini_models(
            self.available_models["google_gemini"]
        )
        
        # Log model diversity
        self._log_model_diversity()
                
        logging.info(f"🎯 Discovered models: Gemini({len(self.available_models['google_gemini'])}), "
                   f"Cloudflare({len(self.available_models['cloudflare'])}), "
                   f"Groq({len(self.available_models['groq'])})")
                   
        return self.available_models
    
    def _log_model_diversity(self):
        """Log the diversity of available models"""
        all_models = []
        for provider, models in self.available_models.items():
            for model in models:
                all_models.append((provider, model))
        
        categories = {}
        for provider, model in all_models:
            category = self.diversity_manager.get_model_category(model)
            if category not in categories:
                categories[category] = []
            categories[category].append(f"{provider}/{model}")
        
        logging.info("📊 Model Diversity Analysis:")
        for category, models in categories.items():
            logging.info(f"  {category}: {len(models)} models")
    
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
            logging.warning(f"❌ Gemini model discovery failed: {e}")
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
            
            account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "default_account_id")
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
                        logging.warning(f"❌ Cloudflare API returned status {response.status}")
                        return self.fallback_models["cloudflare"]
                        
        except Exception as e:
            logging.warning(f"❌ Cloudflare model discovery failed: {e}")
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
                        logging.warning(f"❌ Groq API returned status {response.status}")
                        return self.fallback_models["groq"]
                        
        except Exception as e:
            logging.warning(f"❌ Groq model discovery failed: {e}")
            return self.fallback_models["groq"]
    
    def _prioritize_diverse_models(self, models: List[str], provider: str) -> List[str]:
        """Prioritize models to ensure diversity"""
        if not models:
            return models
            
        # Categorize models
        categorized = {}
        for model in models:
            category = self.diversity_manager.get_model_category(model)
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(model)
        
        # Build diverse list - take 1-2 from each category
        diverse_models = []
        for category, category_models in categorized.items():
            # Take up to 2 models from each category
            diverse_models.extend(category_models[:2])
        
        # If we have too many, trim
        if len(diverse_models) > 8:
            diverse_models = diverse_models[:8]
            
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
        self.model_performance = {}  # Track performance by model
        
    def record_analysis_time(self, symbol: str, duration: float):
        """Record analysis duration"""
        self.analysis_times.append((symbol, duration))
        
    def record_api_time(self, provider: str, duration: float):
        """Record API response time"""
        self.api_response_times.append((provider, duration))
        
    def record_model_performance(self, provider: str, model: str, success: bool, response_time: float):
        """Record model performance metrics"""
        key = f"{provider}/{model}"
        if key not in self.model_performance:
            self.model_performance[key] = {
                'total_requests': 0,
                'successful_requests': 0,
                'total_response_time': 0,
                'last_used': None
            }
        
        self.model_performance[key]['total_requests'] += 1
        self.model_performance[key]['total_response_time'] += response_time
        self.model_performance[key]['last_used'] = datetime.now(UTC)
        
        if success:
            self.model_performance[key]['successful_requests'] += 1
        
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
        
        # Model performance stats
        model_stats = {}
        for model, data in self.model_performance.items():
            success_rate_model = (data['successful_requests'] / data['total_requests'] * 100) if data['total_requests'] > 0 else 0
            avg_response_time = data['total_response_time'] / data['total_requests'] if data['total_requests'] > 0 else 0
            model_stats[model] = {
                'success_rate': round(success_rate_model, 2),
                'avg_response_time': round(avg_response_time, 2),
                'total_requests': data['total_requests']
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

    def get_best_performing_models(self, count: int = 5) -> List[Tuple[str, float]]:
        """Get best performing models based on success rate and speed"""
        scored_models = []
        
        for model, data in self.model_performance.items():
            if data['total_requests'] < 3:  # Minimum requests for reliability
                continue
                
            success_rate = data['successful_requests'] / data['total_requests']
            avg_response_time = data['total_response_time'] / data['total_requests']
            
            # Score: success rate (70%) + speed (30%)
            speed_score = max(0, 1 - (avg_response_time / 10))  # Normalize speed
            score = (success_rate * 0.7) + (speed_score * 0.3)
            
            scored_models.append((model, score))
        
        # Sort by score descending
        scored_models.sort(key=lambda x: x[1], reverse=True)
        return scored_models[:count]

# =================================================================================
# --- Enhanced Smart API Manager with Improved Model Selection ---
# =================================================================================

class EnhancedSmartAPIManager:
    def __init__(self, usage_file: str, model_discoverer: EnhancedDynamicModelDiscoverer):
        self.usage_file = usage_file
        self.model_discoverer = model_discoverer
        self.usage_data = self.load_usage_data()
        self.available_models = {}
        self.failed_models = set()
        self.models_initialized = False
        self.gemini_disabled = False
        self.diversity_manager = ModelDiversityManager()
        self.performance_monitor = PerformanceMonitor()

    async def initialize_models(self):
        """Initialize available models dynamically"""
        if not self.models_initialized:
            self.available_models = await self.model_discoverer.discover_models()
            self.models_initialized = True
            logging.info("🎯 AI Models initialized dynamically with enhanced diversity")

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
        """Enhanced model selection ensuring diversity across providers and architectures"""
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

        # NEW: Get best performing models from performance monitor
        best_models = self.performance_monitor.get_best_performing_models(10)
        best_model_names = [model[0] for model in best_models]
        
        # Step 1: Select from best performing models first (if we have performance data)
        if best_models:
            for model_full_name, score in best_models:
                if len(selected_models) >= target_total:
                    break
                    
                # Parse provider and model name
                if '/' in model_full_name:
                    provider, model_name = model_full_name.split('/', 1)
                else:
                    continue
                    
                if (provider in provider_capacity and 
                    provider_capacity[provider] > 0 and 
                    not self.is_model_failed(provider, model_name)):
                    
                    selected_models.append((provider, model_name))
                    provider_capacity[provider] -= 1
                    logging.info(f"🎯 Selected best-performing model: {provider}/{model_name} (score: {score:.2f})")

        # Step 2: Ensure we have models from each provider category
        provider_categories = [
            ("google_gemini", "Gemini"),
            ("cloudflare", "Cloudflare"), 
            ("groq", "Groq")
        ]
        
        for provider, provider_name in provider_categories:
            if len(selected_models) >= target_total:
                break
                
            if provider_capacity[provider] > 0:
                available_models = self.available_models.get(provider, [])
                for model_name in available_models:
                    if ((provider, model_name) not in selected_models and 
                        not self.is_model_failed(provider, model_name)):
                        
                        selected_models.append((provider, model_name))
                        provider_capacity[provider] -= 1
                        logging.info(f"🎯 Added {provider_name} model for diversity: {model_name}")
                        break

        # Step 3: Ensure architectural diversity
        selected_models = self._ensure_architectural_diversity(selected_models, target_total)

        # Step 4: Fill remaining slots with any available models
        remaining_target = target_total - len(selected_models)
        if remaining_target > 0:
            logging.info(f"🔄 Filling {remaining_target} remaining model slots")
            
            # Try providers in round-robin fashion
            providers_order = ["groq", "cloudflare", "google_gemini"]
            round_robin_index = 0
            
            while remaining_target > 0 and any(provider_capacity[p] > 0 for p in providers_order):
                current_provider = providers_order[round_robin_index % len(providers_order)]
                
                if provider_capacity[current_provider] > 0:
                    for model_name in self.available_models.get(current_provider, []):
                        # For Gemini, only use free tier models
                        if current_provider == "google_gemini" and not FreeTierModelFilter.is_free_tier_model(model_name):
                            continue
                            
                        if ((current_provider, model_name) not in selected_models and 
                            not self.is_model_failed(current_provider, model_name)):
                            selected_models.append((current_provider, model_name))
                            provider_capacity[current_provider] -= 1
                            remaining_target -= 1
                            logging.info(f"🎯 Added {current_provider}/{model_name} to fill quota")
                            break
                        
                round_robin_index += 1
                
                # Safety break
                if round_robin_index > len(providers_order) * 3:
                    break

        # Step 5: Enhanced fallback system - ensure we always get minimum models
        if len(selected_models) < min_required:
            logging.warning(f"⚠️ Only {len(selected_models)} models selected. Activating enhanced fallback...")
            
            # Try to use any available model regardless of previous failures
            for provider in ["groq", "cloudflare", "google_gemini"]:
                if self.can_use_provider(provider):
                    for model_name in self.available_models.get(provider, []):
                        # For Gemini, only use free tier models
                        if provider == "google_gemini" and not FreeTierModelFilter.is_free_tier_model(model_name):
                            continue
                            
                        if (provider, model_name) not in selected_models:
                            selected_models.append((provider, model_name))
                            logging.info(f"🚨 Enhanced fallback: {provider}/{model_name}")
                            if len(selected_models) >= target_total:
                                break
                    if len(selected_models) >= target_total:
                        break

        # FINAL FALLBACK: If still not enough, use synthetic decision maker
        if len(selected_models) == 0:
            logging.error("❌ No AI models available. Using synthetic decision maker.")
            # This ensures we always have at least one "model"
            selected_models.append(("synthetic", "technical_analyzer"))

        # Log final selection diversity
        self._log_selection_diversity(selected_models)
            
        logging.info(f"🎯 {len(selected_models)} models selected: {selected_models}")
        return selected_models

    def _ensure_architectural_diversity(self, selected_models: List[Tuple[str, str]], target_total: int) -> List[Tuple[str, str]]:
        """Ensure selected models represent different architectures"""
        if len(selected_models) >= target_total:
            return selected_models
            
        current_categories = set()
        for provider, model in selected_models:
            category = self.diversity_manager.get_model_category(model)
            current_categories.add(category)
        
        # If we already have good diversity, return
        if len(current_categories) >= 3:
            return selected_models
            
        # Try to add missing architectures
        missing_architectures = self._get_missing_architectures(current_categories)
        
        for architecture in missing_architectures:
            if len(selected_models) >= target_total:
                break
                
            # Find a model with this architecture
            for provider in ["groq", "cloudflare", "google_gemini"]:
                if len(selected_models) >= target_total:
                    break
                    
                if not self.can_use_provider(provider):
                    continue
                    
                for model_name in self.available_models.get(provider, []):
                    category = self.diversity_manager.get_model_category(model_name)
                    if (category == architecture and 
                        (provider, model_name) not in selected_models and
                        not self.is_model_failed(provider, model_name)):
                        
                        selected_models.append((provider, model_name))
                        logging.info(f"🏗️  Added {architecture} model for diversity: {provider}/{model_name}")
                        break
        
        return selected_models

    def _get_missing_architectures(self, current_architectures: set) -> List[str]:
        """Get important missing architectures"""
        important_architectures = [
            "large_llama", "qwen_models", "gemini_models", 
            "mistral_models", "deepseek_models", "gemma_models"
        ]
        
        missing = []
        for arch in important_architectures:
            if arch not in current_architectures:
                missing.append(arch)
                
        return missing[:2]  # Return up to 2 missing architectures to target

    def _log_selection_diversity(self, selected_models: List[Tuple[str, str]]):
        """Log the diversity of selected models"""
        categories = {}
        providers = {}
        
        for provider, model in selected_models:
            category = self.diversity_manager.get_model_category(model)
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
            
            if provider not in providers:
                providers[provider] = 0
            providers[provider] += 1
        
        logging.info("📊 Selected Models Diversity:")
        logging.info(f"  Providers: {providers}")
        logging.info(f"  Architectures: {categories}")

    def record_api_usage(self, provider: str, count: int = 1):
        """Record API usage"""
        if provider in self.usage_data["providers"]:
            self.usage_data["providers"][provider]["used_today"] += count
            self.save_usage_data()

    def record_model_performance(self, provider: str, model: str, success: bool, response_time: float):
        """Record model performance for future selection"""
        self.performance_monitor.record_model_performance(provider, model, success, response_time)

    def get_usage_summary(self) -> str:
        """Get usage summary"""
        summary = "📊 API Usage Summary:\n"
        for provider, data in self.usage_data["providers"].items():
            remaining = data["limit"] - data["used_today"]
            summary += f"  {provider}: {data['used_today']}/{data['limit']} ({remaining} remaining)\n"
        
        # Add performance summary
        best_models = self.performance_monitor.get_best_performing_models(3)
        if best_models:
            summary += "🏆 Best Performing Models:\n"
            for model, score in best_models:
                summary += f"  {model}: {score:.2f}\n"
                
        return summary

# =================================================================================
# --- Enhanced AI Manager with Improved Model Diversity ---
# =================================================================================

class EnhancedAIManager:
    def __init__(self, gemini_api_key: str, cloudflare_api_key: str, groq_api_key: str, api_manager):
        self.gemini_api_key = gemini_api_key
        self.cloudflare_api_key = cloudflare_api_key
        self.groq_api_key = groq_api_key
        self.api_manager = api_manager
        
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)

    def _create_optimized_prompt(self, symbol: str, technical_analysis: Dict) -> str:
        """Create prompt that encourages decisive signals"""
        current_price = technical_analysis.get('current_price', 1.0850)
        trend = technical_analysis.get('htf_trend', {})
        momentum = technical_analysis.get('momentum', {})
        key_levels = technical_analysis.get('key_levels', {})
        risk = technical_analysis.get('risk_assessment', {})
        ml_signal = technical_analysis.get('ml_signal', {})
        
        return f"""As a professional forex trader, analyze {symbol} and provide a TRADING DECISION.

CRITICAL INSTRUCTIONS:
- Be DECISIVE: Prefer BUY/SELL over HOLD when there's reasonable evidence
- Only use HOLD if market conditions are completely unclear
- Consider risk-reward ratios above 1.5 as acceptable

MARKET DATA:
Price: {current_price:.5f}
Trend: {trend.get('direction', 'NEUTRAL')} (Strength: {trend.get('strength', 'UNKNOWN')})
RSI: {momentum.get('rsi', {}).get('value', 50):.1f} ({momentum.get('rsi', {}).get('signal', 'NEUTRAL')})
MACD: {momentum.get('macd', {}).get('trend', 'NEUTRAL')}
Support: {key_levels.get('support_1', current_price * 0.99):.5f}
Resistance: {key_levels.get('resistance_1', current_price * 1.01):.5f}
Volatility: {risk.get('volatility_percent', 0):.2f}%
Risk Level: {risk.get('risk_level', 'MEDIUM')}
ML Signal Strength: {ml_signal.get('signal_strength', 0):.2f}/1.0

TRADING GUIDELINES:
- BUY if trend is bullish and RSI is not overbought
- SELL if trend is bearish and RSI is not oversold  
- HOLD only if trend is completely neutral and no clear direction

Return ONLY this JSON format (NO other text):
{{
  "SYMBOL": "{symbol}",
  "ACTION": "BUY|SELL|HOLD",
  "CONFIDENCE": 1-10,
  "ENTRY": "{current_price:.5f}",
  "STOP_LOSS": "calculated_price",
  "TAKE_PROFIT": "calculated_price",
  "RISK_REWARD_RATIO": "1.5-3.0",
  "ANALYSIS": "brief_technical_reasoning"
}}"""

    async def get_enhanced_ai_analysis(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        """Get enhanced AI analysis with robust fallback system and performance tracking"""
        await self.api_manager.initialize_models()
        selected_models = self.api_manager.select_diverse_models(target_total=5, min_required=3)
        
        if len(selected_models) < 3:
            logging.error(f"❌ Cannot find minimum 3 AI models for {symbol}")
            return None
            
        logging.info(f"🎯 Using {len(selected_models)} diverse AI models for {symbol}")

        tasks = []
        for provider, model_name in selected_models:
            # Handle synthetic fallback
            if provider == "synthetic":
                task = self._get_synthetic_analysis(symbol, technical_analysis)
            else:
                task = self._get_single_analysis(symbol, technical_analysis, provider, model_name)
            tasks.append(task)

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            valid_results = []
            failed_count = 0
            
            for i, (provider, model_name) in enumerate(selected_models):
                start_time = time.time()
                result = results[i]
                response_time = time.time() - start_time
                
                if isinstance(result, Exception):
                    logging.error(f"❌ Error in {provider}/{model_name} for {symbol}: {str(result)}")
                    if provider != "synthetic":  # Don't mark synthetic as failed
                        self.api_manager.mark_model_failed(provider, model_name)
                        self.api_manager.record_model_performance(provider, model_name, False, response_time)
                    failed_count += 1
                    if provider != "synthetic":
                        self.api_manager.record_api_usage(provider)
                elif result is not None:
                    valid_results.append(result)
                    if provider != "synthetic":
                        self.api_manager.record_api_usage(provider)
                        self.api_manager.record_model_performance(provider, model_name, True, response_time)
                else:
                    if provider != "synthetic":
                        self.api_manager.record_api_usage(provider)
                        self.api_manager.record_model_performance(provider, model_name, False, response_time)

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
            logging.error(f"❌ Error in AI analysis for {symbol}: {str(e)}")
            return await self._get_technical_fallback(symbol, technical_analysis)

    async def _get_single_analysis(self, symbol: str, technical_analysis: Dict, provider: str, model_name: str) -> Optional[Dict]:
        """Get analysis from single AI model with enhanced error handling and performance tracking"""
        start_time = time.time()
        try:
            prompt = self._create_optimized_prompt(symbol, technical_analysis)
            
            if provider == "google_gemini":
                result = await self._get_gemini_analysis_optimized(symbol, prompt, model_name)
            elif provider == "cloudflare":
                result = await self._get_cloudflare_analysis(symbol, prompt, model_name)
            elif provider == "groq":
                result = await self._get_groq_analysis(symbol, prompt, model_name)
            else:
                result = None
                
            response_time = time.time() - start_time
            if result and provider != "synthetic":
                self.api_manager.record_model_performance(provider, model_name, True, response_time)
                
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            if provider != "synthetic":
                self.api_manager.record_model_performance(provider, model_name, False, response_time)
            logging.error(f"❌ Error in {provider}/{model_name} for {symbol}: {str(e)}")
            return None

    async def _get_synthetic_analysis(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        """Synthetic analysis based on technical indicators only"""
        try:
            trend = technical_analysis.get('htf_trend', {})
            momentum = technical_analysis.get('momentum', {})
            current_price = technical_analysis.get('current_price', 1.0)
            ml_signal = technical_analysis.get('ml_signal', {})
            
            # Enhanced decision logic with ML signal consideration
            trend_direction = trend.get('direction', 'NEUTRAL')
            rsi = momentum.get('rsi', {}).get('value', 50)
            rsi_signal = momentum.get('rsi', {}).get('signal', 'NEUTRAL')
            ml_strength = ml_signal.get('signal_strength', 0)
            
            action = "HOLD"
            confidence = 5
            
            # Consider ML signal strength
            if ml_strength > 0.6:
                if trend_direction in ["BULLISH", "STRONG_BULLISH"]:
                    action = "BUY"
                    confidence = min(10, 7 + int(ml_strength * 3))
                elif trend_direction in ["BEARISH", "STRONG_BEARISH"]:
                    action = "SELL"
                    confidence = min(10, 7 + int(ml_strength * 3))
            elif trend_direction == "BULLISH" and rsi_signal != "OVERBOUGHT" and rsi < 70:
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
                "ANALYSIS": f"Synthetic signal based on {trend_direction} trend, RSI {rsi:.1f}, ML strength {ml_strength:.2f}",
                "ai_model": "SYNTHETIC_TECHNICAL"
            }
            
        except Exception as e:
            logging.error(f"❌ Synthetic analysis error for {symbol}: {e}")
            return None

    async def _get_technical_fallback(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        """Ultimate fallback using pure technical analysis"""
        try:
            return await self._get_synthetic_analysis(symbol, technical_analysis)
        except Exception as e:
            logging.error(f"❌ Technical fallback also failed for {symbol}: {e}")
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
            logging.error(f"❌ Gemini analysis error for {symbol}: {str(e)}")
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
                        "content": "Return ONLY valid JSON format. No additional text. Be decisive in trading decisions."
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
                            content = str(data)
                            
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
                        "content": "Return ONLY valid JSON format. No additional text. Be decisive in trading decisions."
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
                            logging.warning(f"❌ No choices in Groq response for {symbol}: {data}")
                            return None
                    else:
                        error_text = await response.text()
                        logging.error(f"❌ Groq API error for {symbol}: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            logging.error(f"❌ Groq/{model_name} analysis error for {symbol}: {str(e)}")
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

            # If JSON parsing fails, try text extraction
            logging.warning(f"❌ {ai_name} response for {symbol} lacks valid JSON format, trying text extraction...")
            return self._extract_signal_from_text(cleaned_response, symbol, ai_name)

        except json.JSONDecodeError as e:
            logging.warning(f"❌ JSON error in {ai_name} response for {symbol}, trying text extraction: {e}")
            return self._extract_signal_from_text(cleaned_response, symbol, ai_name)
        except Exception as e:
            logging.error(f"❌ Error parsing {ai_name} response for {symbol}: {str(e)}")
            return None

    def _extract_signal_from_text(self, text: str, symbol: str, ai_name: str) -> Optional[Dict]:
        """Extract trading signal from unstructured text response"""
        try:
            text_upper = text.upper()
            action = "HOLD"
            
            # Enhanced action extraction
            if "BUY" in text_upper and "SELL" not in text_upper and "NOT BUY" not in text_upper:
                action = "BUY"
            elif "SELL" in text_upper and "BUY" not in text_upper and "NOT SELL" not in text_upper:
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
            logging.warning(f"❌ Text extraction failed for {symbol}: {e}")
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
        model_categories = {}
        
        for result in valid_results:
            action = result['ACTION'].upper()
            action_counts[action] = action_counts.get(action, 0) + 1
            
            # Track model categories for diversity analysis
            ai_model = result.get('ai_model', '')
            if 'Gemini' in ai_model:
                category = 'gemini'
            elif 'Cloudflare' in ai_model:
                category = 'cloudflare' 
            elif 'Groq' in ai_model:
                category = 'groq'
            else:
                category = 'other'
                
            if action not in model_categories:
                model_categories[action] = set()
            model_categories[action].add(category)
            
            # FIXED: Better confidence handling
            try:
                confidence_val = float(result.get('CONFIDENCE', 5))
                confidence_sum[action] = confidence_sum.get(action, 0) + confidence_val
            except (ValueError, TypeError):
                confidence_sum[action] = confidence_sum.get(action, 0) + 5  # Default value
        
        logging.info(f"📊 Signal combination for {symbol}: {action_counts}")
        logging.info(f"📊 Model diversity per action: {model_categories}")
        
        total_valid = len(valid_results)
        max_agreement = max(action_counts.values()) if action_counts else 0
        
        # Enhanced agreement calculation considering model diversity
        diversity_bonus = 0
        if max_agreement > 0:
            majority_action = max(action_counts, key=action_counts.get)
            diversity_score = len(model_categories.get(majority_action, set())) / 3.0  # Normalize to 0-1
            diversity_bonus = diversity_score * 0.5  # Up to 0.5 bonus
        
        if max_agreement >= 4:
            agreement_type = 'STRONG_CONSENSUS'
        elif max_agreement == 3:
            agreement_type = 'MEDIUM_CONSENSUS' 
        elif max_agreement == 2:
            agreement_type = 'WEAK_CONSENSUS'
        else:
            agreement_type = 'NO_CONSENSUS'
            
        majority_action = max(action_counts, key=action_counts.get) if action_counts else 'HOLD'
        
        # Calculate average confidence for majority action
        avg_confidence = 5.0  # Default
        if majority_action in confidence_sum and max_agreement > 0:
            avg_confidence = confidence_sum[majority_action] / max_agreement
        
        # Apply diversity bonus to confidence
        avg_confidence = min(10, avg_confidence + diversity_bonus)
        
        combined = {
            'SYMBOL': symbol,
            'ACTION': majority_action,
            'CONFIDENCE': round(avg_confidence, 1),
            'AGREEMENT_LEVEL': max_agreement,
            'AGREEMENT_TYPE': agreement_type,
            'VALID_MODELS': total_valid,
            'TOTAL_MODELS': total_models,
            'MODEL_DIVERSITY_SCORE': round(diversity_score, 2) if max_agreement > 0 else 0,
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
            combined['ANALYSIS'] = f"{majority_action} signal based on agreement of {max_agreement} out of {total_models} AI models with diversity score {combined['MODEL_DIVERSITY_SCORE']}"
            
        return combined

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
# --- Enhanced Trade Filter with Smart Market Analysis ---
# =================================================================================

class EnhancedTradeFilter:
    """
    Advanced trade filtering with intelligent market condition analysis
    """
    def __init__(self):
        self.last_stopout_time = {}
        self.cooldown_minutes = 120
        self.market_regime_cache = {}
        
        # Enhanced volatility ranges for different market conditions
        self.volatility_ranges = {
            'high_volatility': {'min': 0.8, 'max': 10.0},   # High volatility markets
            'normal_volatility': {'min': 0.2, 'max': 3.0},  # Normal trading conditions  
            'low_volatility': {'min': 0.05, 'max': 1.5},    # Low volatility/range-bound
        }
        
        # Time-based market session analysis
        self.trading_sessions = {
            'asian': (22, 8),      # 10 PM - 8 AM UTC
            'london': (8, 16),     # 8 AM - 4 PM UTC  
            'new_york': (12, 20),  # 12 PM - 8 PM UTC
            'overlap': (12, 16)    # London-New York overlap
        }

    def can_trade(self, symbol: str, technical_analysis: Dict, now: Optional[datetime] = None) -> bool:
        """Enhanced trading decision with market regime analysis"""
        now = now or datetime.now(UTC)
        
        # Cooldown check
        if not self._check_cooldown(symbol, now):
            logging.info(f"⏸️ {symbol} in cooldown period")
            return False
            
        # Enhanced volatility check with regime detection
        volatility_ok, volatility_value, regime = self._check_enhanced_volatility(technical_analysis)
        if not volatility_ok:
            logging.info(f"📊 {symbol} volatility {volatility_value:.2f}% outside {regime} range")
            return False
            
        # Market session analysis
        session_analysis = self._analyze_trading_session(now)
        if not self._check_session_suitability(session_analysis, symbol, technical_analysis):
            logging.info(f"⏰ {symbol} not optimal for current session: {session_analysis['current_session']}")
            return False
            
        # Enhanced trend and momentum filter
        if not self._check_enhanced_trend_momentum(technical_analysis):
            logging.info(f"📈 {symbol} trend/momentum conditions not favorable")
            return False
            
        # Market regime consistency check
        if not self._check_market_regime_consistency(technical_analysis, regime):
            logging.info(f"🔄 {symbol} market regime inconsistency detected")
            return False
            
        logging.info(f"✅ Trade filter PASSED for {symbol} (volatility: {volatility_value:.2f}%, regime: {regime}, session: {session_analysis['current_session']})")
        return True

    def _check_cooldown(self, symbol: str, now: datetime) -> bool:
        """Check if symbol is in cooldown period"""
        last_stopout = self.last_stopout_time.get(symbol)
        if last_stopout and (now - last_stopout).total_seconds() < self.cooldown_minutes * 60:
            return False
        return True

    def _check_enhanced_volatility(self, technical_analysis: Dict) -> Tuple[bool, float, str]:
        """Enhanced volatility check with market regime detection"""
        volatility = technical_analysis.get('risk_assessment', {}).get('volatility_percent', 0)
        
        # Determine market regime based on volatility
        if volatility >= self.volatility_ranges['high_volatility']['min']:
            regime = 'high_volatility'
            min_vol = self.volatility_ranges['high_volatility']['min']
            max_vol = self.volatility_ranges['high_volatility']['max']
        elif volatility >= self.volatility_ranges['normal_volatility']['min']:
            regime = 'normal_volatility'
            min_vol = self.volatility_ranges['normal_volatility']['min']
            max_vol = self.volatility_ranges['normal_volatility']['max']
        else:
            regime = 'low_volatility'
            min_vol = self.volatility_ranges['low_volatility']['min']
            max_vol = self.volatility_ranges['low_volatility']['max']
        
        return min_vol <= volatility <= max_vol, volatility, regime

    def _analyze_trading_session(self, now: datetime) -> Dict:
        """Analyze current trading session and characteristics"""
        current_hour = now.hour
        current_session = "unknown"
        session_quality = "low"
        
        # Determine current session
        if self.trading_sessions['asian'][0] <= current_hour < self.trading_sessions['asian'][1]:
            current_session = "asian"
            session_quality = "medium"
        elif self.trading_sessions['london'][0] <= current_hour < self.trading_sessions['london'][1]:
            current_session = "london" 
            session_quality = "high"
        elif self.trading_sessions['new_york'][0] <= current_hour < self.trading_sessions['new_york'][1]:
            current_session = "new_york"
            session_quality = "high"
        elif self.trading_sessions['overlap'][0] <= current_hour < self.trading_sessions['overlap'][1]:
            current_session = "overlap"
            session_quality = "very_high"
            
        return {
            'current_session': current_session,
            'session_quality': session_quality,
            'current_hour': current_hour
        }

    def _check_session_suitability(self, session_analysis: Dict, symbol: str, technical_analysis: Dict) -> bool:
        """Check if current session is suitable for trading the symbol"""
        session_quality = session_analysis['session_quality']
        
        # Session-specific filters
        if session_quality == "very_high":
            return True  # Always trade during overlap
        
        volatility = technical_analysis.get('risk_assessment', {}).get('volatility_percent', 0)
        trend_strength = technical_analysis.get('htf_trend', {}).get('strength', 'WEAK')
        
        if session_quality == "high":
            # For high quality sessions, be more selective with high volatility
            return volatility <= 2.5
        
        elif session_quality == "medium":
            # For medium sessions, require stronger trends
            return trend_strength in ['STRONG', 'VERY_STRONG'] and volatility <= 2.0
        
        else:  # low quality sessions
            # Only trade if conditions are very favorable
            return (trend_strength == 'VERY_STRONG' and 
                   volatility <= 1.5 and
                   technical_analysis.get('ml_signal', {}).get('signal_strength', 0) > 0.7)

    def _check_enhanced_trend_momentum(self, technical_analysis: Dict) -> bool:
        """Enhanced trend and momentum analysis"""
        trend = technical_analysis.get('htf_trend', {})
        momentum = technical_analysis.get('momentum', {})
        
        trend_direction = trend.get('direction', 'NEUTRAL')
        trend_strength = trend.get('strength', 'WEAK')
        rsi = momentum.get('rsi', {}).get('value', 50)
        macd_trend = momentum.get('macd', {}).get('trend', 'NEUTRAL')
        
        # Check for trend-momentum alignment
        trend_momentum_aligned = True
        if trend_direction in ['BULLISH', 'STRONG_BULLISH'] and (rsi > 80 or macd_trend == 'BEARISH'):
            trend_momentum_aligned = False
        elif trend_direction in ['BEARISH', 'STRONG_BEARISH'] and (rsi < 20 or macd_trend == 'BULLISH'):
            trend_momentum_aligned = False
            
        # Require minimum trend strength for trading
        min_trend_strength = trend_strength in ['MODERATE', 'STRONG', 'VERY_STRONG']
        
        return trend_momentum_aligned and min_trend_strength

    def _check_market_regime_consistency(self, technical_analysis: Dict, regime: str) -> bool:
        """Check if technical analysis is consistent with market regime"""
        volatility = technical_analysis.get('risk_assessment', {}).get('volatility_percent', 0)
        trend_strength = technical_analysis.get('htf_trend', {}).get('strength', 'WEAK')
        
        if regime == 'high_volatility':
            # In high volatility, expect strong trends or reversals
            return trend_strength in ['STRONG', 'VERY_STRONG'] or volatility > 2.0
        
        elif regime == 'normal_volatility':
            # Normal conditions - most strategies work
            return True
        
        else:  # low_volatility
            # In low volatility, range-bound or breakout strategies
            market_structure = technical_analysis.get('market_structure', {}).get('higher_timeframe_structure', 'UNKNOWN')
            return market_structure in ['RANGING', 'UPTREND', 'DOWNTREND']

    def mark_stopout(self, symbol: str):
        """Record stopout for cooldown period"""
        self.last_stopout_time[symbol] = datetime.now(UTC)
        logging.info(f"🛑 Stopout recorded for {symbol}, cooldown activated")

# =================================================================================
# --- Advanced Risk Manager with Dynamic Position Sizing ---
# =================================================================================

class AdvancedRiskManager:
    """
    Advanced position sizing and risk management with market regime adaptation
    """
    
    def __init__(self,
                 equity: float = 10000.0,
                 risk_per_trade_pct: float = 1.0,
                 max_leverage: float = 30.0,
                 kelly_cap: float = 0.5,
                 volatility_scaling: bool = True):
        self.equity = equity
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_leverage = max_leverage
        self.kelly_cap = kelly_cap
        self.volatility_scaling = volatility_scaling
        self.trade_history = deque(maxlen=100)

    def calculate_dynamic_stop_loss(self, action: str, current_price: float, 
                                  technical_analysis: Dict, atr: float) -> float:
        """Calculate dynamic stop loss using multiple methods with regime awareness"""
        key_levels = technical_analysis.get('key_levels', {})
        volatility = technical_analysis.get('risk_assessment', {}).get('volatility_percent', 0)
        trend_strength = technical_analysis.get('htf_trend', {}).get('strength', 'WEAK')
        
        # Adjust ATR multiplier based on volatility regime
        if volatility > 2.0:
            atr_multiplier = 1.5  # Tighter stops in high volatility
        elif volatility < 0.5:
            atr_multiplier = 2.5  # Wider stops in low volatility
        else:
            atr_multiplier = 2.0  # Normal conditions
            
        # Adjust for trend strength
        if trend_strength in ['STRONG', 'VERY_STRONG']:
            atr_multiplier *= 0.8  # Tighter stops in strong trends

        if action == "BUY":
            # Method 1: Below nearest support
            sl_support = key_levels.get('support_1', current_price * 0.99)
            
            # Method 2: ATR-based with dynamic multiplier
            sl_atr = current_price - (atr_multiplier * atr)
            
            # Method 3: Percentage-based with volatility adjustment
            volatility_adjustment = min(volatility / 2.0, 3.0)  # Cap at 3%
            sl_percentage = current_price * (1 - (volatility_adjustment / 100))
            
            # Choose the most conservative (highest) stop loss for BUY
            stop_loss = max(sl_support, sl_atr, sl_percentage)
            
        else:  # SELL
            # Method 1: Above nearest resistance
            sl_resistance = key_levels.get('resistance_1', current_price * 1.01)
            
            # Method 2: ATR-based with dynamic multiplier
            sl_atr = current_price + (atr_multiplier * atr)
            
            # Method 3: Percentage-based with volatility adjustment
            volatility_adjustment = min(volatility / 2.0, 3.0)  # Cap at 3%
            sl_percentage = current_price * (1 + (volatility_adjustment / 100))
            
            # Choose the most conservative (lowest) stop loss for SELL
            stop_loss = min(sl_resistance, sl_atr, sl_percentage)
            
        return round(stop_loss, 5)

    def calculate_intelligent_take_profit(self, action: str, entry_price: float, stop_loss: float,
                                        technical_analysis: Dict, atr: float) -> float:
        """Calculate intelligent take profit with dynamic risk-reward ratios"""
        key_levels = technical_analysis.get('key_levels', {})
        volatility = technical_analysis.get('risk_assessment', {}).get('volatility_percent', 0)
        trend_strength = technical_analysis.get('htf_trend', {}).get('strength', 'WEAK')
        risk_amount = abs(entry_price - stop_loss)
        
        # Dynamic risk-reward ratio based on market conditions
        if volatility > 2.0:
            min_rr_ratio = 1.8  # Higher RR in high volatility
        elif trend_strength in ['STRONG', 'VERY_STRONG']:
            min_rr_ratio = 2.0  # Higher RR in strong trends
        else:
            min_rr_ratio = 1.5  # Standard RR
            
        if action == "BUY":
            # Method 1: Risk-Reward Ratio with dynamic minimum
            tp_rr = entry_price + (risk_amount * min_rr_ratio)
            
            # Method 2: Key resistance level
            tp_resistance = key_levels.get('resistance_1', entry_price * (1 + (volatility * 0.02)))
            
            # Method 3: ATR-based (dynamic multiplier)
            atr_multiplier = min_rr_ratio * 1.2  # Scale with RR ratio
            tp_atr = entry_price + (atr_multiplier * atr)
            
            # Method 4: Fibonacci extension (if available)
            fib_extension = key_levels.get('resistance_2', tp_rr)
            
            # Choose the most conservative that meets minimum RR
            candidate_tps = [tp for tp in [tp_rr, tp_resistance, tp_atr, fib_extension] 
                            if (tp - entry_price) >= (risk_amount * min_rr_ratio)]
            
            take_profit = min(candidate_tps) if candidate_tps else tp_rr
            
        else:  # SELL
            # Method 1: Risk-Reward Ratio with dynamic minimum
            tp_rr = entry_price - (risk_amount * min_rr_ratio)
            
            # Method 2: Key support level
            tp_support = key_levels.get('support_1', entry_price * (1 - (volatility * 0.02)))
            
            # Method 3: ATR-based (dynamic multiplier)
            atr_multiplier = min_rr_ratio * 1.2  # Scale with RR ratio
            tp_atr = entry_price - (atr_multiplier * atr)
            
            # Method 4: Fibonacci extension (if available)
            fib_extension = key_levels.get('support_2', tp_rr)
            
            # Choose the most conservative that meets minimum RR
            candidate_tps = [tp for tp in [tp_rr, tp_support, tp_atr, fib_extension] 
                            if (entry_price - tp) >= (risk_amount * min_rr_ratio)]
            
            take_profit = max(candidate_tps) if candidate_tps else tp_rr
        
        return round(take_profit, 5)

    def calculate_dynamic_position_size(self, entry_price: float, stop_loss: float, 
                                      technical_analysis: Dict) -> Dict:
        """Calculate dynamic position size based on multiple factors"""
        risk_amount = abs(entry_price - stop_loss)
        if risk_amount == 0:
            risk_amount = entry_price * 0.01  # Default 1% risk
            
        # Base position size from risk percentage
        risk_amount_per_unit = risk_amount / entry_price if entry_price > 0 else 0.01
        base_units = (self.equity * self.risk_per_trade_pct / 100) / risk_amount_per_unit
        
        # Apply volatility scaling if enabled
        if self.volatility_scaling:
            volatility = technical_analysis.get('risk_assessment', {}).get('volatility_percent', 1.0)
            volatility_factor = max(0.5, min(2.0, 1.5 / (volatility + 0.5)))  # Scale inversely with volatility
            base_units *= volatility_factor
        
        # Apply trend strength factor
        trend_strength = technical_analysis.get('htf_trend', {}).get('strength', 'WEAK')
        trend_factor = {
            'VERY_STRONG': 1.2,
            'STRONG': 1.1,
            'MODERATE': 1.0,
            'WEAK': 0.8,
            'UNKNOWN': 0.7
        }.get(trend_strength, 1.0)
        base_units *= trend_factor
        
        # Apply ML signal confidence factor
        ml_confidence = technical_analysis.get('ml_signal', {}).get('confidence', 0.5)
        ml_factor = 0.5 + ml_confidence  # 0.5 to 1.5 range
        base_units *= ml_factor
        
        # Ensure position size is within leverage limits
        max_units_from_leverage = (self.equity * self.max_leverage) / entry_price
        final_units = min(base_units, max_units_from_leverage)
        
        # Calculate notional value and margin
        notional_value = final_units * entry_price
        margin_required = notional_value / self.max_leverage
        
        return {
            'position_size_units': round(final_units, 2),
            'notional_value': round(notional_value, 2),
            'margin_required': round(margin_required, 2),
            'risk_amount': round(risk_amount * final_units, 2),
            'risk_percent': round((risk_amount * final_units) / self.equity * 100, 2)
        }

    def validate_signal_risk(self, signal: Dict, technical_analysis: Dict) -> Dict:
        """Enhanced signal validation with dynamic risk management"""
        if not signal or signal.get('ACTION') == 'HOLD':
            return signal
            
        current_price = technical_analysis.get('current_price', 0)
        atr = technical_analysis.get('volatility', 0.001) or 0.001
        action = signal.get('ACTION')
        
        # Calculate intelligent stop loss and take profit
        stop_loss = self.calculate_dynamic_stop_loss(action, current_price, technical_analysis, atr)
        take_profit = self.calculate_intelligent_take_profit(action, current_price, stop_loss, technical_analysis, atr)
        
        # Calculate dynamic position size
        position_info = self.calculate_dynamic_position_size(current_price, stop_loss, technical_analysis)
        
        # Update signal with calculated levels and position info
        signal['STOP_LOSS'] = f"{stop_loss:.5f}"
        signal['TAKE_PROFIT'] = f"{take_profit:.5f}"
        signal['POSITION_SIZE'] = position_info
        
        # Calculate actual risk-reward ratio
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        actual_rr = reward / risk if risk > 0 else 1.8
        signal['ACTUAL_RR_RATIO'] = round(actual_rr, 2)
        
        # Add risk assessment
        signal['RISK_ASSESSMENT'] = {
            'volatility_regime': 'HIGH' if technical_analysis.get('risk_assessment', {}).get('volatility_percent', 0) > 2.0 else 'NORMAL',
            'trend_alignment': self._assess_trend_alignment(signal, technical_analysis),
            'position_size_rating': self._rate_position_size(position_info)
        }
        
        return signal

    def _assess_trend_alignment(self, signal: Dict, technical_analysis: Dict) -> str:
        """Assess how well the signal aligns with the trend"""
        action = signal.get('ACTION')
        trend_direction = technical_analysis.get('htf_trend', {}).get('direction', 'NEUTRAL')
        
        if (action == 'BUY' and 'BULLISH' in trend_direction) or (action == 'SELL' and 'BEARISH' in trend_direction):
            return 'ALIGNED'
        elif trend_direction == 'NEUTRAL':
            return 'NEUTRAL'
        else:
            return 'COUNTER_TREND'

    def _rate_position_size(self, position_info: Dict) -> str:
        """Rate the position size appropriateness"""
        risk_percent = position_info.get('risk_percent', 0)
        
        if risk_percent <= 0.5:
            return 'CONSERVATIVE'
        elif risk_percent <= 1.0:
            return 'MODERATE'
        elif risk_percent <= 2.0:
            return 'AGGRESSIVE'
        else:
            return 'HIGHLY_AGGRESSIVE'

    def record_trade(self, symbol: str, action: str, outcome: str, pnl: float):
        """Record trade for performance analysis and Kelly criterion"""
        trade_record = {
            'symbol': symbol,
            'action': action,
            'outcome': outcome,  # 'win', 'loss', 'breakeven'
            'pnl': pnl,
            'timestamp': datetime.now(UTC).isoformat()
        }
        self.trade_history.append(trade_record)

    def calculate_kelly_position_size(self, symbol: str, action: str) -> float:
        """Calculate position size using Kelly criterion based on trade history"""
        relevant_trades = [t for t in self.trade_history 
                          if t['symbol'] == symbol and t['action'] == action]
        
        if len(relevant_trades) < 10:
            return 1.0  # Default to full position if insufficient data
            
        wins = [t for t in relevant_trades if t['outcome'] == 'win']
        losses = [t for t in relevant_trades if t['outcome'] == 'loss']
        
        if not wins and not losses:
            return 1.0
            
        win_rate = len(wins) / len(relevant_trades)
        
        # Calculate average win/loss ratio
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t['pnl']) for t in losses]) if losses else 0
        
        if avg_loss == 0:
            return 1.0
            
        win_loss_ratio = avg_win / avg_loss
        
        # Kelly formula: f = (bp - q) / b
        # where b = win/loss ratio, p = win rate, q = loss rate
        kelly_fraction = (win_loss_ratio * win_rate - (1 - win_rate)) / win_loss_ratio
        
        # Apply Kelly cap and ensure positive
        kelly_fraction = max(0, min(kelly_fraction, self.kelly_cap))
        
        return kelly_fraction

# =================================================================================
# --- Enhanced Signal Quality Scorer with ML Integration ---
# =================================================================================

class EnhancedSignalQualityScorer:
    """Evaluate signal quality based on multiple factors with ML enhancement"""
    
    def __init__(self):
        self.weights = {
            'ai_agreement': 0.20,
            'technical_alignment': 0.18,
            'risk_reward': 0.15,
            'trend_strength': 0.12,
            'volatility_appropriateness': 0.10,
            'momentum_confirmation': 0.10,
            'market_structure': 0.08,
            'model_diversity': 0.07
        }
        
        # ML-based pattern recognition (simplified)
        self.pattern_weights = {
            'trend_continuation': 1.2,
            'breakout_confirmation': 1.15,
            'reversal_pattern': 1.1,
            'range_bound': 0.9,
            'counter_trend': 0.7
        }
    
    def calculate_signal_score(self, signal: Dict, technical_analysis: Dict) -> float:
        """Calculate comprehensive signal quality score (0-100) with ML enhancement"""
        if signal.get('ACTION') == 'HOLD':
            return 0.0
            
        scores = {}
        
        # 1. AI Agreement Score
        agreement_level = signal.get('AGREEMENT_LEVEL', 0)
        total_models = signal.get('TOTAL_MODELS', 1)
        scores['ai_agreement'] = (agreement_level / total_models) * 100
        
        # 2. Technical Alignment Score
        scores['technical_alignment'] = self._calculate_enhanced_technical_alignment(signal, technical_analysis)
        
        # 3. Risk-Reward Score
        rr_ratio = float(signal.get('ACTUAL_RR_RATIO', 1.0))
        scores['risk_reward'] = self._calculate_rr_score(rr_ratio)
        
        # 4. Trend Strength Score
        scores['trend_strength'] = self._calculate_trend_strength_score(technical_analysis)
        
        # 5. Volatility Appropriateness Score
        scores['volatility_appropriateness'] = self._calculate_enhanced_volatility_score(technical_analysis)
        
        # 6. Momentum Confirmation Score
        scores['momentum_confirmation'] = self._calculate_momentum_score(signal, technical_analysis)
        
        # 7. Market Structure Score
        scores['market_structure'] = self._calculate_market_structure_score(technical_analysis)
        
        # 8. Model Diversity Score
        diversity_score = signal.get('MODEL_DIVERSITY_SCORE', 0)
        scores['model_diversity'] = diversity_score * 100
        
        # Apply pattern recognition multiplier
        pattern_multiplier = self._identify_market_pattern(technical_analysis)
        
        # Calculate weighted total score
        total_score = 0
        for factor, weight in self.weights.items():
            total_score += scores.get(factor, 0) * weight
            
        # Apply pattern multiplier
        total_score *= pattern_multiplier
        
        return min(100, round(total_score, 1))
    
    def _calculate_enhanced_technical_alignment(self, signal: Dict, technical_analysis: Dict) -> float:
        """Enhanced technical alignment with multiple timeframe analysis"""
        action = signal.get('ACTION')
        htf_trend = technical_analysis.get('htf_trend', {})
        ltf_trend = technical_analysis.get('ltf_trend', {})
        momentum = technical_analysis.get('momentum', {})
        market_structure = technical_analysis.get('market_structure', {})
        
        alignment_score = 0
        
        # Multi-timeframe trend alignment (40 points max)
        htf_direction = htf_trend.get('direction', 'NEUTRAL')
        ltf_direction = ltf_trend.get('direction', 'NEUTRAL')
        
        if (action == 'BUY' and 'BULLISH' in htf_direction and 'BULLISH' in ltf_direction):
            alignment_score += 40
        elif (action == 'SELL' and 'BEARISH' in htf_direction and 'BEARISH' in ltf_direction):
            alignment_score += 40
        elif (action == 'BUY' and 'BULLISH' in htf_direction) or (action == 'SELL' and 'BEARISH' in htf_direction):
            alignment_score += 25
        elif htf_direction == 'NEUTRAL' and ltf_direction == 'NEUTRAL':
            alignment_score += 15
        else:  # Counter-trend or conflicting timeframes
            alignment_score += 5
            
        # Momentum alignment (30 points max)
        momentum_bias = momentum.get('overall_bias', 'NEUTRAL')
        rsi_signal = momentum.get('rsi', {}).get('signal', 'NEUTRAL')
        macd_trend = momentum.get('macd', {}).get('trend', 'NEUTRAL')
        
        momentum_signals = 0
        if (action == 'BUY' and momentum_bias == 'BULLISH') or (action == 'SELL' and momentum_bias == 'BEARISH'):
            momentum_signals += 1
        if (action == 'BUY' and rsi_signal == 'OVERSOLD') or (action == 'SELL' and rsi_signal == 'OVERBOUGHT'):
            momentum_signals += 1
        if (action == 'BUY' and macd_trend == 'BULLISH') or (action == 'SELL' and macd_trend == 'BEARISH'):
            momentum_signals += 1
            
        alignment_score += (momentum_signals / 3) * 30
        
        # Market structure alignment (30 points max)
        structure = market_structure.get('higher_timeframe_structure', 'UNKNOWN')
        is_breaking = market_structure.get('is_breaking_structure', False)
        
        if (action == 'BUY' and structure == 'UPTREND' and not is_breaking) or \
           (action == 'SELL' and structure == 'DOWNTREND' and not is_breaking):
            alignment_score += 30
        elif (action == 'BUY' and is_breaking and structure == 'DOWNTREND') or \
             (action == 'SELL' and is_breaking and structure == 'UPTREND'):
            alignment_score += 25  # Breakout signals
        elif structure == 'RANGING':
            alignment_score += 20
        else:
            alignment_score += 10
            
        return min(alignment_score, 100)
    
    def _calculate_rr_score(self, rr_ratio: float) -> float:
        """Calculate risk-reward score with progressive scaling"""
        if rr_ratio >= 3.0:
            return 100
        elif rr_ratio >= 2.0:
            return 80 + (rr_ratio - 2.0) * 20
        elif rr_ratio >= 1.5:
            return 60 + (rr_ratio - 1.5) * 40
        elif rr_ratio >= 1.0:
            return 30 + (rr_ratio - 1.0) * 60
        else:
            return max(0, rr_ratio * 30)
    
    def _calculate_trend_strength_score(self, technical_analysis: Dict) -> float:
        """Calculate trend strength score with multiple confirmations"""
        htf_trend = technical_analysis.get('htf_trend', {})
        ltf_trend = technical_analysis.get('ltf_trend', {})
        
        htf_strength = htf_trend.get('strength', 'WEAK')
        ltf_strength = ltf_trend.get('strength', 'WEAK')
        htf_adx = htf_trend.get('adx', 0)
        ema_alignment = htf_trend.get('ema_alignment', 0)
        
        strength_scores = {
            'VERY_STRONG': 100, 'STRONG': 80, 'MODERATE': 60, 'WEAK': 30, 'UNKNOWN': 0
        }
        
        # Average strength from both timeframes
        base_score = (strength_scores.get(htf_strength, 0) + strength_scores.get(ltf_strength, 0)) / 2
        
        # ADX confirmation (up to +20 points)
        adx_bonus = min(20, htf_adx / 2)
        
        # EMA alignment bonus (up to +15 points)
        ema_bonus = min(15, ema_alignment * 3)
        
        return min(100, base_score + adx_bonus + ema_bonus)
    
    def _calculate_enhanced_volatility_score(self, technical_analysis: Dict) -> float:
        """Enhanced volatility score with regime awareness"""
        volatility = technical_analysis.get('risk_assessment', {}).get('volatility_percent', 0)
        trend_strength = technical_analysis.get('htf_trend', {}).get('strength', 'WEAK')
        market_phase = technical_analysis.get('market_structure', {}).get('market_phase', 'UNKNOWN')
        
        # Different optimal volatility ranges based on market conditions
        if trend_strength in ['STRONG', 'VERY_STRONG']:
            # Strong trends work well with moderate volatility
            if 0.5 <= volatility <= 2.5:
                return 100
            elif 0.2 <= volatility < 0.5 or 2.5 < volatility <= 4.0:
                return 70
            else:
                return 30
        elif market_phase == 'RANGING':
            # Range markets prefer lower volatility
            if 0.1 <= volatility <= 1.5:
                return 100
            elif 0.05 <= volatility < 0.1 or 1.5 < volatility <= 2.5:
                return 60
            else:
                return 20
        else:
            # General conditions
            if 0.3 <= volatility <= 2.0:
                return 100
            elif 0.1 <= volatility < 0.3 or 2.0 < volatility <= 3.0:
                return 70
            elif 0.05 <= volatility < 0.1 or 3.0 < volatility <= 5.0:
                return 40
            else:
                return 20
    
    def _calculate_momentum_score(self, signal: Dict, technical_analysis: Dict) -> float:
        """Calculate momentum confirmation score"""
        momentum = technical_analysis.get('momentum', {})
        action = signal.get('ACTION')
        
        score = 50  # Base score
        
        # RSI alignment
        rsi_signal = momentum.get('rsi', {}).get('signal', 'NEUTRAL')
        if (action == 'BUY' and rsi_signal == 'OVERSOLD') or (action == 'SELL' and rsi_signal == 'OVERBOUGHT'):
            score += 25
        elif rsi_signal == 'NEUTRAL':
            score += 10
        else:  # Wrong signal
            score -= 15
            
        # MACD alignment
        macd_trend = momentum.get('macd', {}).get('trend', 'NEUTRAL')
        macd_cross = momentum.get('macd', {}).get('cross', 'NO_CROSS')
        if (action == 'BUY' and macd_trend == 'BULLISH') or (action == 'SELL' and macd_trend == 'BEARISH'):
            score += 15
        if macd_cross in ['BULLISH_CROSS', 'BEARISH_CROSS']:
            score += 10
            
        # Stochastic alignment
        stoch_signal = momentum.get('stochastic', {}).get('signal', 'NEUTRAL')
        if (action == 'BUY' and stoch_signal == 'OVERSOLD') or (action == 'SELL' and stoch_signal == 'OVERBOUGHT'):
            score += 10
            
        return max(0, min(score, 100))
    
    def _calculate_market_structure_score(self, technical_analysis: Dict) -> float:
        """Calculate market structure score"""
        market_structure = technical_analysis.get('market_structure', {})
        structure = market_structure.get('higher_timeframe_structure', 'UNKNOWN')
        is_breaking = market_structure.get('is_breaking_structure', False)
        market_phase = market_structure.get('market_phase', 'UNKNOWN')
        
        if structure in ['UPTREND', 'DOWNTREND'] and not is_breaking:
            return 100
        elif structure in ['UPTREND', 'DOWNTREND'] and is_breaking:
            return 70
        elif structure == 'RANGING' and market_phase in ['ACCUMULATION', 'DISTRIBUTION']:
            return 60
        elif structure == 'RANGING':
            return 40
        else:
            return 20
    
    def _identify_market_pattern(self, technical_analysis: Dict) -> float:
        """Identify market patterns and return score multiplier"""
        trend = technical_analysis.get('htf_trend', {})
        market_structure = technical_analysis.get('market_structure', {})
        momentum = technical_analysis.get('momentum', {})
        
        trend_direction = trend.get('direction', 'NEUTRAL')
        trend_strength = trend.get('strength', 'WEAK')
        is_breaking = market_structure.get('is_breaking_structure', False)
        rsi = momentum.get('rsi', {}).get('value', 50)
        
        # Pattern recognition logic
        if trend_strength in ['STRONG', 'VERY_STRONG'] and not is_breaking:
            return self.pattern_weights['trend_continuation']
        elif is_breaking and trend_strength in ['MODERATE', 'STRONG']:
            return self.pattern_weights['breakout_confirmation']
        elif (trend_direction in ['BULLISH', 'STRONG_BULLISH'] and rsi > 70) or \
             (trend_direction in ['BEARISH', 'STRONG_BEARISH'] and rsi < 30):
            return self.pattern_weights['reversal_pattern']
        elif trend_strength == 'WEAK' and not is_breaking:
            return self.pattern_weights['range_bound']
        else:
            return 1.0  # Default multiplier

    def get_detailed_breakdown(self, signal: Dict, technical_analysis: Dict) -> Dict:
        """Get detailed breakdown of signal scoring"""
        scores = {}
        
        scores['ai_agreement'] = (signal.get('AGREEMENT_LEVEL', 0) / signal.get('TOTAL_MODELS', 1)) * 100
        scores['technical_alignment'] = self._calculate_enhanced_technical_alignment(signal, technical_analysis)
        scores['risk_reward'] = self._calculate_rr_score(float(signal.get('ACTUAL_RR_RATIO', 1.0)))
        scores['trend_strength'] = self._calculate_trend_strength_score(technical_analysis)
        scores['volatility_appropriateness'] = self._calculate_enhanced_volatility_score(technical_analysis)
        scores['momentum_confirmation'] = self._calculate_momentum_score(signal, technical_analysis)
        scores['market_structure'] = self._calculate_market_structure_score(technical_analysis)
        scores['model_diversity'] = signal.get('MODEL_DIVERSITY_SCORE', 0) * 100
        
        total_score = 0
        for factor, weight in self.weights.items():
            total_score += scores.get(factor, 0) * weight
            
        pattern_multiplier = self._identify_market_pattern(technical_analysis)
        total_score *= pattern_multiplier
        
        return {
            'final_score': min(100, round(total_score, 1)),
            'factor_scores': scores,
            'weights': self.weights,
            'pattern_multiplier': pattern_multiplier,
            'pattern_identified': self._get_pattern_description(technical_analysis)
        }
    
    def _get_pattern_description(self, technical_analysis: Dict) -> str:
        """Get description of identified market pattern"""
        trend = technical_analysis.get('htf_trend', {})
        market_structure = technical_analysis.get('market_structure', {})
        momentum = technical_analysis.get('momentum', {})
        
        trend_direction = trend.get('direction', 'NEUTRAL')
        trend_strength = trend.get('strength', 'WEAK')
        is_breaking = market_structure.get('is_breaking_structure', False)
        rsi = momentum.get('rsi', {}).get('value', 50)
        
        if trend_strength in ['STRONG', 'VERY_STRONG'] and not is_breaking:
            return f"Strong {trend_direction.lower()} trend continuation"
        elif is_breaking and trend_strength in ['MODERATE', 'STRONG']:
            return "Breakout confirmation pattern"
        elif (trend_direction in ['BULLISH', 'STRONG_BULLISH'] and rsi > 70) or \
             (trend_direction in ['BEARISH', 'STRONG_BEARISH'] and rsi < 30):
            return "Potential reversal pattern"
        elif trend_strength == 'WEAK' and not is_breaking:
            return "Range-bound market conditions"
        else:
            return "Standard market conditions"

# =================================================================================
# --- Enhanced Gemini Direct Signal Agent ---
# =================================================================================

class EnhancedGeminiDirectSignalAgent:
    """
    Enhanced direct Gemini signal agent with improved error handling and market context
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = 'gemini-1.5-flash'):
        self.model_name = model_name
        self.fallback_model = 'gemini-1.5-pro'
        self.available = False
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'avg_response_time': 0
        }
        try:
            k = api_key or os.getenv("GOOGLE_API_KEY")
            if k:
                genai.configure(api_key=k)
                self.available = True
                logging.info(f"✅ EnhancedGeminiDirectSignalAgent initialized with {model_name}")
            else:
                logging.warning("⚠️ EnhancedGeminiDirectSignalAgent: GOOGLE_API_KEY not set.")
        except Exception as e:
            logging.error(f"❌ EnhancedGeminiDirectSignalAgent init error: {e}")

    def _create_enhanced_prompt(self, symbol: str, ta: Dict) -> str:
        """Create enhanced prompt with comprehensive market context"""
        price = ta.get("current_price", 1.0)
        htf_trend = ta.get("htf_trend", {})
        ltf_trend = ta.get("ltf_trend", {})
        risk = ta.get("risk_assessment", {})
        ml = ta.get("ml_signal", {})
        key_levels = ta.get("key_levels", {})
        momentum = ta.get("momentum", {})
        market_structure = ta.get("market_structure", {})
        
        return f"""As an expert forex trading analyst, analyze {symbol} and provide ONLY JSON output.

CRITICAL: Be DECISIVE - prefer BUY/SELL over HOLD unless market is completely unclear.

COMPREHENSIVE MARKET ANALYSIS:

PRICE & TREND:
- Current Price: {price:.5f}
- HTF Trend: {htf_trend.get('direction', 'NEUTRAL')} ({htf_trend.get('strength', 'UNKNOWN')})
- LTF Trend: {ltf_trend.get('direction', 'NEUTRAL')} ({ltf_trend.get('strength', 'UNKNOWN')})
- ADX Strength: {htf_trend.get('adx', 0):.1f}
- EMA Alignment: {htf_trend.get('ema_alignment', 0)}/4

MOMENTUM & OSCILLATORS:
- RSI: {momentum.get('rsi', {}).get('value', 50):.1f} ({momentum.get('rsi', {}).get('signal', 'NEUTRAL')})
- MACD: {momentum.get('macd', {}).get('trend', 'NEUTRAL')} ({momentum.get('macd', {}).get('cross', 'NO_CROSS')})
- Stochastic: K={momentum.get('stochastic', {}).get('k', 50):.1f}, Signal: {momentum.get('stochastic', {}).get('signal', 'NEUTRAL')}
- Momentum Bias: {momentum.get('overall_bias', 'NEUTRAL')}

KEY LEVELS:
- Support 1: {key_levels.get('support_1', price*0.99):.5f}
- Resistance 1: {key_levels.get('resistance_1', price*1.01):.5f}
- Pivot: {key_levels.get('pivot', price):.5f}
- Bollinger Bands: {key_levels.get('bb_lower', price*0.98):.5f} - {key_levels.get('bb_upper', price*1.02):.5f}

MARKET STRUCTURE:
- HTF Structure: {market_structure.get('higher_timeframe_structure', 'UNKNOWN')}
- Breaking Structure: {market_structure.get('is_breaking_structure', False)}
- Market Phase: {market_structure.get('market_phase', 'UNKNOWN')}

RISK & VOLATILITY:
- Volatility: {risk.get('volatility_percent', 0):.2f}%
- Risk Level: {risk.get('risk_level', 'MEDIUM')}
- ATR: {risk.get('atr_value', 0.001):.5f}
- ML Signal Strength: {ml.get('signal_strength', 0):.2f}/1.0

TRADING DECISION MATRIX:
1. STRONG BUY: HTF & LTF bullish + RSI < 70 + Strong momentum
2. BUY: HTF bullish + RSI < 70 + Good risk-reward
3. HOLD: Mixed signals, unclear direction, or poor risk-reward
4. SELL: HTF bearish + RSI > 30 + Good risk-reward  
5. STRONG SELL: HTF & LTF bearish + RSI > 30 + Strong momentum

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
  "TRADE_RATIONALE": "detailed_explanation_based_on_above_data",
  "EXPECTED_MOVE_PERCENT": "estimated_price_move"
}}"""

    async def fetch_signal(self, symbol: str, technical_analysis: Dict) -> Optional[Dict]:
        """Fetch direct signal from Gemini with enhanced reliability and performance tracking"""
        if not self.available:
            return None
            
        start_time = time.time()
        
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
                        max_output_tokens=800,
                        top_p=0.8,
                        top_k=40
                    )
                )
                
                text = self._extract_text(response)
                data = self._clean_parse(text)
                if data and self._validate_direct_signal(data, symbol):
                    response_time = time.time() - start_time
                    self._update_performance_stats(True, response_time)
                    logging.info(f"✅ EnhancedGeminiDirect signal for {symbol}: {data.get('ACTION', 'HOLD')} (Confidence: {data.get('CONFIDENCE', 0)})")
                    return data
                    
            except Exception as e:
                logging.warning(f"⚠️ EnhancedGeminiDirect primary model failed: {e}")
                # Try fallback model
                if self.model_name != self.fallback_model:
                    logging.info(f"🔄 Trying fallback model: {self.fallback_model}")
                    model = genai.GenerativeModel(self.fallback_model)
                    response = await asyncio.to_thread(
                        model.generate_content,
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.1,
                            max_output_tokens=800,
                        )
                    )
                    text = self._extract_text(response)
                    data = self._clean_parse(text)
                    if data and self._validate_direct_signal(data, symbol):
                        response_time = time.time() - start_time
                        self._update_performance_stats(True, response_time)
                        logging.info(f"✅ EnhancedGeminiDirect (fallback) signal for {symbol}: {data.get('ACTION', 'HOLD')}")
                        return data

            response_time = time.time() - start_time
            self._update_performance_stats(False, response_time)
            return None
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_performance_stats(False, response_time)
            logging.error(f"❌ EnhancedGeminiDirectSignalAgent error for {symbol}: {e}")
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
            logging.warning(f"EnhancedGeminiDirect JSON parsing error: {e}")
            return None

    def _validate_direct_signal(self, signal_data: Dict, symbol: str) -> bool:
        """Validate direct signal data with enhanced checks"""
        required = ['SYMBOL', 'ACTION', 'CONFIDENCE']
        for field in required:
            if field not in signal_data:
                logging.warning(f"❌ EnhancedGeminiDirect missing field {field} for {symbol}")
                return False
                
        action = signal_data['ACTION'].upper()
        if action not in ['BUY', 'SELL', 'HOLD']:
            logging.warning(f"❌ EnhancedGeminiDirect invalid ACTION for {symbol}: {action}")
            return False
            
        try:
            confidence = float(signal_data['CONFIDENCE'])
            if not (1 <= confidence <= 10):
                logging.warning(f"❌ EnhancedGeminiDirect CONFIDENCE out of range for {symbol}: {confidence}")
                return False
        except (ValueError, TypeError):
            logging.warning(f"❌ EnhancedGeminiDirect invalid CONFIDENCE for {symbol}: {signal_data['CONFIDENCE']}")
            return False
            
        # Enhanced validation for trading levels
        if action != 'HOLD':
            for field in ['ENTRY', 'STOP_LOSS', 'TAKE_PROFIT']:
                if field not in signal_data or not signal_data[field]:
                    logging.warning(f"❌ EnhancedGeminiDirect missing {field} for {symbol} {action} signal")
                    return False
                    
        return True

    def _update_performance_stats(self, success: bool, response_time: float):
        """Update performance statistics"""
        self.performance_stats['total_requests'] += 1
        if success:
            self.performance_stats['successful_requests'] += 1
        
        # Update average response time
        current_avg = self.performance_stats['avg_response_time']
        total_requests = self.performance_stats['total_requests']
        self.performance_stats['avg_response_time'] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        success_rate = (
            (self.performance_stats['successful_requests'] / self.performance_stats['total_requests'] * 100)
            if self.performance_stats['total_requests'] > 0 else 0
        )
        
        return {
            'total_requests': self.performance_stats['total_requests'],
            'successful_requests': self.performance_stats['successful_requests'],
            'success_rate': round(success_rate, 1),
            'avg_response_time': round(self.performance_stats['avg_response_time'], 2)
        }

# =================================================================================
# --- Enhanced Main Forex Analyzer Class ---
# =================================================================================

class EnhancedForexAnalyzer:
    def __init__(self, strict_filters: bool = False):
        self.model_discoverer = EnhancedDynamicModelDiscoverer()
        self.api_manager = EnhancedSmartAPIManager(USAGE_TRACKER_FILE, self.model_discoverer)
        self.technical_analyzer = AdvancedTechnicalAnalyzer()
        self.ai_manager = EnhancedAIManager(google_api_key, CLOUDFLARE_AI_API_KEY, GROQ_API_KEY, self.api_manager)
        self.data_fetcher = EnhancedDataFetcher()
        self.performance_monitor = PerformanceMonitor()

        # Enhanced components
        self.gemini_direct = EnhancedGeminiDirectSignalAgent(google_api_key, 'gemini-1.5-flash')
        self.risk_manager = AdvancedRiskManager()
        self.signal_scorer = EnhancedSignalQualityScorer()
        self.trade_filter = EnhancedTradeFilter()
        self.strict_filters = strict_filters
        
        # Analysis cache for performance
        self.analysis_cache = {}
        self.cache_ttl = 300  # 5 minutes

    async def analyze_pair(self, pair: str) -> Optional[Dict]:
        """Complete enhanced analysis with intelligent caching and robust fallbacks"""
        logging.info(f"🔍 Starting enhanced analysis for {pair}")
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{pair}_{datetime.now(UTC).strftime('%Y%m%d_%H%M')}"
        if cache_key in self.analysis_cache:
            cached_result, timestamp = self.analysis_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                logging.info(f"📦 Using cached analysis for {pair}")
                return cached_result
        
        try:
            logging.info(self.api_manager.get_usage_summary())
            
            # Get market data with enhanced error handling
            htf_df = await self.data_fetcher.get_market_data(pair, HIGH_TIMEFRAME)
            ltf_df = await self.data_fetcher.get_market_data(pair, LOW_TIMEFRAME)
            
            if htf_df is None or ltf_df is None:
                logging.warning(f"⚠️ Market data retrieval failed for {pair}")
                self.performance_monitor.record_failure()
                return None
                
            logging.info(f"✅ Retrieved data: HTF={len(htf_df)} rows, LTF={len(ltf_df)} rows")
            
            # Enhanced technical analysis with progressive fallbacks
            htf_df_processed = self.technical_analyzer.calculate_enhanced_indicators(htf_df)
            ltf_df_processed = self.technical_analyzer.calculate_enhanced_indicators(ltf_df)
            
            if htf_df_processed is None or ltf_df_processed is None:
                logging.warning(f"⚠️ Technical analysis failed for {pair}, using basic indicators")
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

            # Enhanced trade filter with market regime awareness
            filter_result = self._apply_enhanced_filters(pair, technical_analysis)
            if not filter_result['can_trade']:
                logging.info(f"⏸️ Trade filter blocked {pair}: {filter_result['reason']}")
                self.performance_monitor.record_success()
                
                # Return analysis even if filtered for monitoring purposes
                analysis_duration = time.time() - start_time
                self.performance_monitor.record_analysis_time(pair, analysis_duration)
                
                return {
                    'SYMBOL': pair,
                    'ACTION': 'HOLD',
                    'CONFIDENCE': 0,
                    'FILTER_REASON': filter_result['reason'],
                    'TECHNICAL_ANALYSIS': technical_analysis,
                    'QUALITY_SCORE': 0,
                    'timestamp': datetime.now(UTC).isoformat()
                }
                
            # Enhanced AI analysis (ensemble with robust fallbacks)
            ai_analysis = await self.ai_manager.get_enhanced_ai_analysis(pair, technical_analysis)

            # Enhanced Gemini direct as intelligent backup
            if (not ai_analysis or ai_analysis.get('ACTION') == 'HOLD') and self.gemini_direct.available:
                logging.info(f"🔄 Trying EnhancedGeminiDirect as intelligent backup for {pair}")
                direct_signal = await self.gemini_direct.fetch_signal(pair, technical_analysis)
                if direct_signal and direct_signal.get('ACTION') != 'HOLD':
                    logging.info(f"🎯 EnhancedGeminiDirect provided decisive signal for {pair}")
                    ai_analysis = direct_signal
                    ai_analysis['AGREEMENT_LEVEL'] = 1
                    ai_analysis['AGREEMENT_TYPE'] = 'DIRECT_GEMINI_SIGNAL'
                    ai_analysis['VALID_MODELS'] = 1
                    ai_analysis['TOTAL_MODELS'] = 1

            if ai_analysis:
                # Enhanced risk management and position sizing
                ai_analysis = self.risk_manager.validate_signal_risk(ai_analysis, technical_analysis)
                
                # Calculate comprehensive signal quality score
                quality_score = self.signal_scorer.calculate_signal_score(ai_analysis, technical_analysis)
                ai_analysis['QUALITY_SCORE'] = quality_score
                
                # Add detailed technical context
                ai_analysis['TECHNICAL_CONTEXT'] = self._create_technical_context(technical_analysis)
                
                # Add filter information
                ai_analysis['FILTER_STATUS'] = filter_result
                
                analysis_duration = time.time() - start_time
                self.performance_monitor.record_analysis_time(pair, analysis_duration)
                self.performance_monitor.record_success()
                
                # Cache successful analysis
                self.analysis_cache[cache_key] = (ai_analysis, time.time())
                
                logging.info(f"✅ Enhanced signal for {pair}: {ai_analysis['ACTION']} "
                           f"(Quality: {quality_score}/100, Agreement: {ai_analysis.get('AGREEMENT_LEVEL', 0)}/{ai_analysis.get('TOTAL_MODELS', 0)})")
                return ai_analysis
                
            self.performance_monitor.record_failure()
            logging.info(f"🔍 No trading signal for {pair}")
            
            # Return technical analysis even without signal
            analysis_duration = time.time() - start_time
            self.performance_monitor.record_analysis_time(pair, analysis_duration)
            
            return {
                'SYMBOL': pair,
                'ACTION': 'HOLD',
                'CONFIDENCE': 0,
                'TECHNICAL_ANALYSIS': technical_analysis,
                'QUALITY_SCORE': 0,
                'timestamp': datetime.now(UTC).isoformat()
            }
            
        except Exception as e:
            self.performance_monitor.record_failure()
            logging.error(f"❌ Error analyzing {pair}: {str(e)}")
            logging.error(f"❌ Traceback: {traceback.format_exc()}")
            return None

    def _apply_enhanced_filters(self, symbol: str, technical_analysis: Dict) -> Dict:
        """Apply enhanced trading filters with detailed reporting"""
        result = {
            'can_trade': True,
            'reason': 'All filters passed',
            'detailed_checks': {}
        }
        
        # Market hours check
        session_analysis = self.trade_filter._analyze_trading_session(datetime.now(UTC))
        result['detailed_checks']['market_session'] = {
            'session': session_analysis['current_session'],
            'quality': session_analysis['session_quality'],
            'passed': True  # We're more permissive with sessions now
        }
        
        # Volatility check
        volatility_ok, volatility_value, regime = self.trade_filter._check_enhanced_volatility(technical_analysis)
        result['detailed_checks']['volatility'] = {
            'value': volatility_value,
            'regime': regime,
            'passed': volatility_ok
        }
        
        if not volatility_ok:
            result['can_trade'] = False
            result['reason'] = f'Volatility {volatility_value:.2f}% outside {regime} range'
            
        # Trend and momentum check
        trend_ok = self.trade_filter._check_enhanced_trend_momentum(technical_analysis)
        result['detailed_checks']['trend_momentum'] = {
            'passed': trend_ok
        }
        
        if not trend_ok:
            result['can_trade'] = False
            result['reason'] = 'Trend and momentum conditions not favorable'
            
        # Market regime consistency
        regime_ok = self.trade_filter._check_market_regime_consistency(technical_analysis, regime)
        result['detailed_checks']['market_regime'] = {
            'passed': regime_ok
        }
        
        if not regime_ok:
            result['can_trade'] = False
            result['reason'] = 'Market regime inconsistency detected'
            
        return result

    def _create_technical_context(self, technical_analysis: Dict) -> Dict:
        """Create comprehensive technical context for signals"""
        return {
            'trend_analysis': {
                'htf_direction': technical_analysis.get('htf_trend', {}).get('direction', 'NEUTRAL'),
                'htf_strength': technical_analysis.get('htf_trend', {}).get('strength', 'UNKNOWN'),
                'ltf_direction': technical_analysis.get('ltf_trend', {}).get('direction', 'NEUTRAL'),
                'adx': technical_analysis.get('htf_trend', {}).get('adx', 0),
                'ema_alignment': technical_analysis.get('htf_trend', {}).get('ema_alignment', 0)
            },
            'momentum_analysis': {
                'rsi': technical_analysis.get('momentum', {}).get('rsi', {}).get('value', 50),
                'rsi_signal': technical_analysis.get('momentum', {}).get('rsi', {}).get('signal', 'NEUTRAL'),
                'macd_trend': technical_analysis.get('momentum', {}).get('macd', {}).get('trend', 'NEUTRAL'),
                'overall_bias': technical_analysis.get('momentum', {}).get('overall_bias', 'NEUTRAL'),
                'convergence_score': technical_analysis.get('momentum', {}).get('convergence_score', 0)
            },
            'key_levels': {
                'support_1': technical_analysis.get('key_levels', {}).get('support_1', 0),
                'resistance_1': technical_analysis.get('key_levels', {}).get('resistance_1', 0),
                'pivot': technical_analysis.get('key_levels', {}).get('pivot', 0)
            },
            'risk_assessment': {
                'volatility': technical_analysis.get('risk_assessment', {}).get('volatility_percent', 0),
                'risk_level': technical_analysis.get('risk_assessment', {}).get('risk_level', 'MEDIUM'),
                'current_range': technical_analysis.get('risk_assessment', {}).get('current_range_percent', 0)
            },
            'market_structure': {
                'htf_structure': technical_analysis.get('market_structure', {}).get('higher_timeframe_structure', 'UNKNOWN'),
                'is_breaking': technical_analysis.get('market_structure', {}).get('is_breaking_structure', False),
                'market_phase': technical_analysis.get('market_structure', {}).get('market_phase', 'UNKNOWN')
            },
            'ml_signal': technical_analysis.get('ml_signal', {})
        }

    async def analyze_all_pairs(self, pairs: List[str]) -> List[Dict]:
        """Analyze all currency pairs with enhanced parallel processing"""
        logging.info(f"🚀 Starting enhanced analysis for {len(pairs)} currency pairs")
        
        # Initialize models first
        await self.api_manager.initialize_models()
        
        # Log initial model diversity
        available_models = self.api_manager.available_models
        logging.info("🎯 Available Model Diversity:")
        for provider, models in available_models.items():
            categories = {}
            for model in models:
                category = self.model_discoverer.diversity_manager.get_model_category(model)
                categories[category] = categories.get(category, 0) + 1
            logging.info(f"  {provider}: {categories}")
        
        # Enhanced parallel processing with concurrency control
        semaphore = asyncio.Semaphore(3)  # Limit concurrent analyses
        
        async def analyze_with_semaphore(pair):
            async with semaphore:
                return await self.analyze_pair(pair)
        
        tasks = [analyze_with_semaphore(pair) for pair in pairs]
        results = await asyncio.gather(*tasks)
        
        # Filter out None results and separate signals from analysis
        valid_results = [r for r in results if r is not None]
        trading_signals = [r for r in valid_results if r.get('ACTION') != 'HOLD']
        analysis_only = [r for r in valid_results if r.get('ACTION') == 'HOLD']
        
        # Sort trading signals by quality score
        trading_signals.sort(key=lambda x: x.get('QUALITY_SCORE', 0), reverse=True)
        
        logging.info(f"📊 Enhanced analysis complete. {len(trading_signals)} trading signals, {len(analysis_only)} analysis-only results")
        
        # Enhanced performance logging
        perf_stats = self.performance_monitor.get_performance_stats()
        logging.info(f"📈 Enhanced Performance Statistics:")
        logging.info(f"  Total Analyses: {perf_stats['total_analyses']}")
        logging.info(f"  Success Rate: {perf_stats['success_rate']}%")
        logging.info(f"  Avg Analysis Time: {perf_stats['avg_analysis_time_sec']}s")
        
        # Model performance summary
        if perf_stats['model_performance']:
            logging.info("🏆 Model Performance Summary:")
            best_models = sorted(
                perf_stats['model_performance'].items(),
                key=lambda x: x[1]['success_rate'],
                reverse=True
            )[:5]
            for model, stats in best_models:
                logging.info(f"  {model}: {stats['success_rate']}% success, {stats['avg_response_time']}s avg")
        
        # Gemini Direct performance
        gemini_stats = self.gemini_direct.get_performance_stats()
        logging.info(f"🤖 Gemini Direct: {gemini_stats['success_rate']}% success rate")
        
        return trading_signals + analysis_only

    def save_enhanced_signals(self, signals: List[Dict]):
        """Save signals with enhanced categorization and metadata"""
        import os
        
        current_dir = os.getcwd()
        logging.info(f"📁 Current directory for file saving: {current_dir}")
        
        if not signals:
            logging.info("📝 No signals to save")
            self._create_empty_signal_files()
            return

        # Enhanced categorization with quality scoring and market context
        strong_signals = []
        medium_signals = []
        weak_signals = []
        analysis_only = []
        
        for signal in signals:
            action = signal.get('ACTION', 'HOLD')
            quality_score = signal.get('QUALITY_SCORE', 0)
            agreement_type = signal.get('AGREEMENT_TYPE', '')
            
            if action == 'HOLD':
                analysis_only.append(signal)
                continue
                
            # Enhanced categorization logic
            if (quality_score >= 80 and agreement_type in ['STRONG_CONSENSUS', 'DIRECT_GEMINI_SIGNAL']):
                strong_signals.append(signal)
            elif (quality_score >= 60 or 
                  (agreement_type == 'MEDIUM_CONSENSUS' and quality_score >= 50)):
                medium_signals.append(signal)
            else:
                weak_signals.append(signal)
                
        # Save to files with enhanced metadata
        try:
            # Strong signals with detailed metadata
            strong_data = {
                'metadata': {
                    'generated_at': datetime.now(UTC).isoformat(),
                    'total_signals': len(strong_signals),
                    'average_quality': sum(s.get('QUALITY_SCORE', 0) for s in strong_signals) / len(strong_signals) if strong_signals else 0,
                    'filter_strictness': 'STRICT' if self.strict_filters else 'FLEXIBLE'
                },
                'signals': strong_signals
            }
            with open("strong_consensus_signals.json", 'w', encoding='utf-8') as f:
                json.dump(strong_data, f, indent=2, ensure_ascii=False)
            logging.info(f"💾 {len(strong_signals)} strong signals saved")
            
            # Medium signals
            medium_data = {
                'metadata': {
                    'generated_at': datetime.now(UTC).isoformat(),
                    'total_signals': len(medium_signals),
                    'average_quality': sum(s.get('QUALITY_SCORE', 0) for s in medium_signals) / len(medium_signals) if medium_signals else 0
                },
                'signals': medium_signals
            }
            with open("medium_consensus_signals.json", 'w', encoding='utf-8') as f:
                json.dump(medium_data, f, indent=2, ensure_ascii=False)
            logging.info(f"💾 {len(medium_signals)} medium signals saved")
            
            # Weak signals
            weak_data = {
                'metadata': {
                    'generated_at': datetime.now(UTC).isoformat(),
                    'total_signals': len(weak_signals),
                    'average_quality': sum(s.get('QUALITY_SCORE', 0) for s in weak_signals) / len(weak_signals) if weak_signals else 0
                },
                'signals': weak_signals
            }
            with open("weak_consensus_signals.json", 'w', encoding='utf-8') as f:
                json.dump(weak_data, f, indent=2, ensure_ascii=False)
            logging.info(f"💾 {len(weak_signals)} weak signals saved")
            
            # Analysis only (market analysis without signals)
            analysis_data = {
                'metadata': {
                    'generated_at': datetime.now(UTC).isoformat(),
                    'total_analysis': len(analysis_only)
                },
                'analysis': analysis_only
            }
            with open("market_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)
            logging.info(f"💾 {len(analysis_only)} market analysis results saved")
            
            # Enhanced summary file
            summary = self._create_enhanced_summary(signals, strong_signals, medium_signals, weak_signals, analysis_only)
            
            with open("enhanced_analysis_summary.json", 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logging.info("💾 Enhanced analysis summary saved")
            
            # Verify file creation
            for filename in ["strong_consensus_signals.json", "medium_consensus_signals.json", 
                           "weak_consensus_signals.json", "market_analysis.json", "enhanced_analysis_summary.json"]:
                if os.path.exists(filename):
                    logging.info(f"✅ File {filename} successfully created")
                else:
                    logging.error(f"❌ File {filename} not created!")
                    
        except Exception as e:
            logging.error(f"❌ Error saving enhanced signals: {e}")

    def _create_empty_signal_files(self):
        """Create empty signal files with metadata"""
        empty_data = {
            'metadata': {
                'generated_at': datetime.now(UTC).isoformat(),
                'total_signals': 0,
                'message': 'No trading signals generated'
            },
            'signals': []
        }
        
        try:
            files_to_create = [
                "strong_consensus_signals.json",
                "medium_consensus_signals.json", 
                "weak_consensus_signals.json",
                "market_analysis.json"
            ]
            for filename in files_to_create:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(empty_data, f, indent=2, ensure_ascii=False)
                logging.info(f"💾 Empty file created: {filename}")
        except Exception as e:
            logging.error(f"❌ Error creating empty files: {e}")

    def _create_enhanced_summary(self, all_signals: List[Dict], strong: List[Dict], 
                               medium: List[Dict], weak: List[Dict], analysis: List[Dict]) -> Dict:
        """Create enhanced analysis summary with comprehensive metrics"""
        trading_signals = [s for s in all_signals if s.get('ACTION') != 'HOLD']
        
        # Calculate various metrics
        avg_quality = sum(s.get('QUALITY_SCORE', 0) for s in trading_signals) / len(trading_signals) if trading_signals else 0
        avg_confidence = sum(float(s.get('CONFIDENCE', 0)) for s in trading_signals) / len(trading_signals) if trading_signals else 0
        avg_rr_ratio = sum(float(s.get('ACTUAL_RR_RATIO', 1.0)) for s in trading_signals) / len(trading_signals) if trading_signals else 0
        
        # Action distribution
        buy_signals = [s for s in trading_signals if s.get('ACTION') == 'BUY']
        sell_signals = [s for s in trading_signals if s.get('ACTION') == 'SELL']
        
        # Agreement distribution
        agreement_types = {}
        for signal in trading_signals:
            agreement = signal.get('AGREEMENT_TYPE', 'UNKNOWN')
            agreement_types[agreement] = agreement_types.get(agreement, 0) + 1
        
        # Performance stats
        perf_stats = self.performance_monitor.get_performance_stats()
        gemini_stats = self.gemini_direct.get_performance_stats()
        data_source_stats = self.data_fetcher.get_data_source_stats()
        
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "summary_metrics": {
                "total_analysis": len(all_signals),
                "trading_signals": len(trading_signals),
                "analysis_only": len(analysis),
                "strong_signals": len(strong),
                "medium_signals": len(medium),
                "weak_signals": len(weak),
                "buy_signals": len(buy_signals),
                "sell_signals": len(sell_signals),
                "average_quality_score": round(avg_quality, 1),
                "average_confidence": round(avg_confidence, 1),
                "average_rr_ratio": round(avg_rr_ratio, 2)
            },
            "agreement_distribution": agreement_types,
            "performance_metrics": {
                "analysis_success_rate": perf_stats['success_rate'],
                "avg_analysis_time": perf_stats['avg_analysis_time_sec'],
                "gemini_direct_success_rate": gemini_stats['success_rate'],
                "data_sources": data_source_stats
            },
            "model_performance": perf_stats.get('model_performance', {}),
            "filter_settings": {
                "strict_mode": self.strict_filters,
                "volatility_ranges": self.trade_filter.volatility_ranges
            },
            "risk_settings": {
                "equity": self.risk_manager.equity,
                "risk_per_trade": self.risk_manager.risk_per_trade_pct,
                "max_leverage": self.risk_manager.max_leverage
            }
        }

# =================================================================================
# --- Installation Helper & Main Execution ---
# =================================================================================

def install_required_packages():
    """Install required packages if missing"""
    required_packages = ['yfinance', 'pandas-ta', 'aiohttp', 'scipy', 'google-generativeai']
    
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
            elif package == 'google-generativeai':
                import google.generativeai
            print(f"✅ {package} is already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully!")

# =================================================================================
# --- Enhanced Main Function ---
# =================================================================================

async def enhanced_main():
    """Enhanced main program execution function"""
    logging.info("🎯 Starting Enhanced Forex Analysis System (Advanced AI Engine)")
    
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
    parser.add_argument("--volatility_scaling", action="store_true", help="Enable volatility-based position scaling")
    
    args = parser.parse_args()
    
    if args.pair:
        pairs_to_analyze = [args.pair]
    elif args.pairs:
        pairs_to_analyze = [p.strip() for p in args.pairs.split(',')]
    elif args.all:
        pairs_to_analyze = CURRENCY_PAIRS_TO_ANALYZE
    else:
        pairs_to_analyze = CURRENCY_PAIRS_TO_ANALYZE[:4]  # Default to first 4 pairs
        logging.info(f"🔍 Using default currency pairs: {', '.join(pairs_to_analyze)}")
    
    logging.info(f"🎯 Currency pairs to analyze: {', '.join(pairs_to_analyze)}")
    
    # Use strict filters only if explicitly requested
    analyzer = EnhancedForexAnalyzer(strict_filters=args.strict)

    # Configure risk manager with CLI arguments
    analyzer.risk_manager.equity = args.equity
    analyzer.risk_manager.risk_per_trade_pct = args.risk_pct
    analyzer.risk_manager.kelly_cap = args.kelly_cap
    analyzer.risk_manager.max_leverage = args.max_lev
    analyzer.risk_manager.volatility_scaling = args.volatility_scaling

    logging.info(f"⚙️ Enhanced Risk Configuration: Equity=${args.equity:.0f}, Risk={args.risk_pct}%, "
               f"Kelly Cap={args.kelly_cap}, Max Leverage={args.max_lev}x, Volatility Scaling={args.volatility_scaling}")
    logging.info(f"⚙️ Filter Mode: {'STRICT' if args.strict else 'FLEXIBLE'}")

    # Analyze all pairs
    signals = await analyzer.analyze_all_pairs(pairs_to_analyze)
    
    # Save enhanced signals
    analyzer.save_enhanced_signals(signals)
    
    # Display enhanced results
    logging.info("📈 Enhanced Results Summary:")
    trading_signals = [s for s in signals if s.get('ACTION') != 'HOLD']
    
    if trading_signals:
        strong_count = len([s for s in trading_signals if s.get('QUALITY_SCORE', 0) >= 80])
        medium_count = len([s for s in trading_signals if 60 <= s.get('QUALITY_SCORE', 0) < 80])
        weak_count = len([s for s in trading_signals if s.get('QUALITY_SCORE', 0) < 60])
        
        avg_quality = sum(s.get('QUALITY_SCORE', 0) for s in trading_signals) / len(trading_signals)
        avg_confidence = sum(float(s.get('CONFIDENCE', 0)) for s in trading_signals) / len(trading_signals)
        avg_rr = sum(float(s.get('ACTUAL_RR_RATIO', 1.0)) for s in trading_signals) / len(trading_signals)
        
        logging.info(f"🎯 Strong signals (80+): {strong_count}")
        logging.info(f"📊 Medium signals (60-79): {medium_count}") 
        logging.info(f"📈 Weak signals (<60): {weak_count}")
        logging.info(f"📋 Average Quality Score: {avg_quality:.1f}/100")
        logging.info(f"🎯 Average Confidence: {avg_confidence:.1f}/10")
        logging.info(f"⚖️ Average R:R Ratio: {avg_rr:.2f}:1")
        
        for signal in trading_signals:
            action_icon = "🟢" if signal['ACTION'] == 'BUY' else "🔴" if signal['ACTION'] == 'SELL' else "⚪"
            quality_score = signal.get('QUALITY_SCORE', 0)
            quality_icon = "🔥" if quality_score >= 80 else "✅" if quality_score >= 60 else "⚠️"
            agreement = signal.get('AGREEMENT_TYPE', '')
            
            logging.info(f"  {action_icon} {quality_icon} {signal['SYMBOL']}: {signal['ACTION']} "
                       f"(Quality: {quality_score}/100, Conf: {signal.get('CONFIDENCE', 0)}/10)"
                       f" | RR: {signal.get('ACTUAL_RR_RATIO', 'N/A')}:1"
                       f" | Agreement: {agreement}")
            
            # Log position sizing for strong signals
            if quality_score >= 70 and 'POSITION_SIZE' in signal:
                pos_info = signal['POSITION_SIZE']
                logging.info(f"     📊 Position: {pos_info['position_size_units']:.2f} units, "
                           f"Risk: ${pos_info['risk_amount']:.2f} ({pos_info['risk_percent']:.1f}%)")
    else:
        logging.info("🔍 No trading signals generated. Check market_analysis.json for detailed analysis.")
    
    # Display enhanced statistics
    data_source_stats = analyzer.data_fetcher.get_data_source_stats()
    logging.info("📊 Data Source Statistics:")
    for source, count in data_source_stats.items():
        logging.info(f"  {source}: {count} pairs")
    
    # Final API status
    analyzer.api_manager.save_usage_data()
    logging.info(analyzer.api_manager.get_usage_summary())
    
    # Gemini Direct performance
    gemini_stats = analyzer.gemini_direct.get_performance_stats()
    logging.info(f"🤖 Gemini Direct Performance: {gemini_stats['success_rate']}% success rate "
               f"({gemini_stats['successful_requests']}/{gemini_stats['total_requests']})")
    
    if trading_signals:
        logging.info("🏁 Enhanced system execution completed successfully with signals!")
    else:
        logging.info("🏁 Enhanced system executed successfully. Check market_analysis.json for market insights.")

if __name__ == "__main__":
    asyncio.run(enhanced_main())            
