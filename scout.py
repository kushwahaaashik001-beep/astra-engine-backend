#!/usr/bin/env python3
"""
Astra Scout - The Omniscient Hunter (Level ∞)
Author: Your Name
Purpose: Self-evolving keyword hunter that finds real, high-profit search gaps.
         Uses AI validation, recursive expansion, SERP authority analysis,
         and continuous learning to ensure only gold enters the pipeline.
"""

import os
import sys
import time
import random
import logging
import re
import json
from typing import List, Dict, Optional, Tuple, Set
from urllib.parse import quote_plus, urlparse
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================ CONFIGURATION ============================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Google Suggest API
GOOGLE_SUGGEST_URL = "http://suggestqueries.google.com/complete/search"

# SERP API (free tier from serpapi.com or similar)
SERP_API_KEY = os.getenv("SERP_API_KEY")  # optional
SERP_API_URL = "https://serpapi.com/search"

# Groq AI API (for validation)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # optional
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-70b-8192"  # or any other

# Proxies (comma-separated in env, or file)
PROXIES = os.getenv("PROXY_LIST", "").split(",") if os.getenv("PROXY_LIST") else []
if not PROXIES or PROXIES == [""]:
    PROXIES = [None]  # fallback direct

# User agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
]

# Thresholds
MIN_SEARCH_VOLUME = 10
MIN_CPC = 1.0
MAX_COMPETITION_SCORE = 0.3  # lower is better (0-1 scale)
MIN_GAP_SCORE = 0.6  # SERP gap must be at least this to consider

# Recursive expansion settings
ALPHABET = 'abcdefghijklmnopqrstuvwxyz0123456789'
MAX_RECURSION_DEPTH = 2  # how deep to go (1 means just append one letter)
MAX_SUGGESTIONS_PER_SEED = 20

# Patterns for initial seed generation (will be expanded dynamically)
PATTERNS = [
    "how to fix {brand} {model} error {code}",
    "{brand} {device} alarm {code} reset",
    "troubleshoot {brand} {part} fault {code}",
    "{brand} plc error code {code} solution",
    "resolve {brand} {device} error {code}",
    "{brand} {model} {code} manual",
]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AstraScout")

# ============================ SUPABASE CLIENT ============================
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    logger.error("Supabase credentials missing. Set SUPABASE_URL and SUPABASE_SERVICE_KEY.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ============================ SEED TABLE MANAGEMENT ============================
def ensure_seed_tables():
    """Create seed tables if they don't exist (call this once manually)."""
    # We assume tables exist; if not, create them via Supabase SQL.
    pass

def load_seed_list(table_name: str) -> List[str]:
    """Load seed list from Supabase table."""
    try:
        response = supabase.table(table_name).select("name").execute()
        return [row['name'] for row in response.data]
    except Exception as e:
        logger.error(f"Failed to load {table_name}: {e}")
        return []

def add_seed_item(table_name: str, item: str):
    """Insert new seed item if not exists."""
    if not item or len(item) < 2:
        return
    try:
        supabase.table(table_name).upsert({"name": item.strip()}, on_conflict="name").execute()
        logger.debug(f"Added new seed to {table_name}: {item}")
    except Exception as e:
        logger.error(f"Failed to add {item} to {table_name}: {e}")

# Default seeds (used only if tables are empty)
DEFAULT_BRANDS = ["Siemens", "Fanuc", "Allen Bradley", "ABB", "Mitsubishi", "Yaskawa", "Omron", "Schneider Electric"]
DEFAULT_DEVICES = ["CNC", "drive", "servo", "HMI", "PLC", "controller", "robot", "inverter"]
DEFAULT_MODELS = ["V-Series", "S7-1200", "S7-1500", "R30iA", "C1000", "ACS880", "G5", "NexGen"]
DEFAULT_PARTS = ["motor", "encoder", "power supply", "I/O module", "communication card", "driver"]
DEFAULT_CODES = ["16#0001", "SV0401", "F001", "ALM-402", "ERR-99", "E-101", "F002", "AL-001", "ER-12"]

def initialize_seeds_if_empty():
    """Populate seed tables with defaults if empty."""
    for table, defaults in [
        ("brands", DEFAULT_BRANDS),
        ("devices", DEFAULT_DEVICES),
        ("models", DEFAULT_MODELS),
        ("parts", DEFAULT_PARTS),
        ("codes", DEFAULT_CODES)
    ]:
        existing = load_seed_list(table)
        if not existing:
            logger.info(f"Populating {table} with defaults.")
            for item in defaults:
                add_seed_item(table, item)

# ============================ HELPER FUNCTIONS ============================
def get_random_proxy():
    return random.choice(PROXIES) if PROXIES else None

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def create_session_with_retries():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def fetch_google_suggestions(query: str) -> List[str]:
    """
    Fetch Google Autocomplete suggestions for a query.
    Returns list of suggestion strings.
    """
    params = {
        "client": "firefox",
        "q": query,
        "hl": "en",
        "gl": "us"
    }
    headers = {
        "User-Agent": get_random_user_agent(),
        "Accept-Language": "en-US,en;q=0.9"
    }
    proxy = get_random_proxy()
    proxies = {"http": proxy, "https": proxy} if proxy else None

    session = create_session_with_retries()
    try:
        resp = session.get(GOOGLE_SUGGEST_URL, params=params, headers=headers, proxies=proxies, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
                return data[1]
        else:
            logger.warning(f"Suggest API returned {resp.status_code} for '{query}'")
    except Exception as e:
        logger.error(f"Error fetching suggestions: {e}")
    return []

def recursive_suggestion_expansion(seed: str, depth: int = 0) -> Set[str]:
    """
    Recursively expand a seed query by appending letters/numbers
    to get deep long-tail suggestions.
    Returns a set of unique keywords.
    """
    if depth >= MAX_RECURSION_DEPTH:
        return set()
    suggestions = set()
    base_suggestions = fetch_google_suggestions(seed)
    for sugg in base_suggestions:
        sugg = sugg.strip()
        if len(sugg) > 5:
            suggestions.add(sugg)
            # For each suggestion, try appending each letter to get more
            for ch in ALPHABET:
                extended = f"{sugg} {ch}"
                deeper = recursive_suggestion_expansion(extended, depth+1)
                suggestions.update(deeper)
            if len(suggestions) > MAX_SUGGESTIONS_PER_SEED:
                break
    return suggestions

# ============================ AI VALIDATION (Groq) ============================
def validate_with_ai(keyword: str) -> Tuple[bool, float]:
    """
    Use Groq AI to determine if keyword represents a real technical problem
    worth solving. Returns (is_valid, confidence_score 0-1).
    If Groq is not configured, falls back to rule-based.
    """
    if not GROQ_API_KEY:
        # fallback to rule-based
        return rule_based_validation(keyword), 0.7

    prompt = f"""You are an expert in industrial automation and technical troubleshooting. 
Analyze the following search query and determine if it represents a real technical problem that would require a detailed solution. 
Answer only with a JSON object: {{"valid": true/false, "confidence": 0.0-1.0, "reason": "short reason"}}.

Query: "{keyword}"
"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 100
    }
    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            content = data['choices'][0]['message']['content']
            # extract JSON
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                result = json.loads(match.group())
                return result.get('valid', False), result.get('confidence', 0.5)
        else:
            logger.warning(f"Groq API error: {resp.status_code}")
    except Exception as e:
        logger.error(f"Groq validation exception: {e}")
    # fallback
    return rule_based_validation(keyword), 0.5

def rule_based_validation(keyword: str) -> bool:
    """Rule-based validation as fallback."""
    kw = keyword.lower()
    brands = [b.lower() for b in load_seed_list("brands")]
    codes = load_seed_list("codes")
    has_brand = any(b in kw for b in brands)
    has_code = any(c.lower() in kw for c in codes) or bool(re.search(r'[0-9a-f#x]{3,}', kw))
    intent_words = ['fix', 'repair', 'reset', 'troubleshoot', 'error', 'fault', 'alarm', 'code', 'solution', 'manual', 'problem']
    has_intent = any(word in kw for word in intent_words)
    return has_brand and has_code and has_intent

# ============================ SERP GAP ANALYSIS ============================
def analyze_serp_gap(keyword: str) -> float:
    """
    Analyze top Google results and return a gap score (0-1):
    1 = perfect gap (only forums, low-authority)
    0 = saturated (official docs, high-authority dominate)
    Uses SerpAPI if available, else heuristics.
    """
    if SERP_API_KEY:
        return analyze_serp_api(keyword)
    else:
        return analyze_serp_fallback(keyword)

def analyze_serp_api(keyword: str) -> float:
    """Use SerpAPI to get organic results and compute gap score."""
    try:
        params = {
            "q": keyword,
            "api_key": SERP_API_KEY,
            "num": 10,
            "gl": "us",
            "hl": "en"
        }
        resp = requests.get(SERP_API_URL, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            organic = data.get("organic_results", [])
            if not organic:
                return 0.5  # unknown

            # Define authority indicators
            low_authority_domains = ['forum', 'reddit', 'quora', 'stackoverflow', 'community', 'answers', 'wordpress', 'blogspot']
            high_authority_domains = ['siemens', 'fanuc', 'abb', 'wikipedia', 'youtube', 'manualslib', 'pdf', 'manual']
            pdf_indicator = '.pdf'

            low_count = 0
            high_count = 0
            total = 0
            for result in organic[:5]:  # top 5
                link = result.get("link", "").lower()
                if pdf_indicator in link:
                    high_count += 1  # PDF is usually official manual
                elif any(domain in link for domain in high_authority_domains):
                    high_count += 1
                elif any(domain in link for domain in low_authority_domains):
                    low_count += 1
                total += 1

            if total == 0:
                return 0.5
            # Gap score: proportion of low authority among considered
            gap = low_count / total if (low_count + high_count) > 0 else 0.5
            # Also penalize if any PDF is present
            if high_count > 0:
                gap *= 0.8  # reduce gap if high authority present
            return min(gap, 1.0)
        else:
            logger.warning(f"SERP API error: {resp.status_code}")
    except Exception as e:
        logger.error(f"SERP API exception: {e}")
    return 0.5

def analyze_serp_fallback(keyword: str) -> float:
    """Heuristic fallback when no SERP API."""
    # If keyword contains long hex code, likely low competition
    if re.search(r'[0-9a-f]{4,}', keyword.lower()):
        return 0.8
    # If contains very specific model number, also good
    if re.search(r'[A-Z0-9\-]{5,}', keyword):
        return 0.7
    return 0.5

# ============================ CPC/VOLUME ESTIMATION ============================
def estimate_cpc(keyword: str) -> float:
    """Placeholder for real CPC data. Replace with Google Ads API or similar."""
    # For demo, return random but could be improved
    return round(random.uniform(0.5, 50.0), 2)

def estimate_search_volume(keyword: str) -> int:
    """Placeholder for real volume data."""
    return random.randint(10, 1000)

# ============================ SELF-LEARNING EXTRACTION ============================
def extract_new_seeds(keyword: str):
    """
    From keyword like "Siemens S7-1200 error 16#0001", extract brand, model, code, etc.
    Add to respective seed tables.
    """
    kw_lower = keyword.lower()
    # Brand extraction (if not already in list, add first word if capitalized)
    brands = [b.lower() for b in load_seed_list("brands")]
    found_brand = None
    for b in brands:
        if b in kw_lower:
            found_brand = b
            break
    if not found_brand:
        # Try to extract first word if it looks like a brand
        words = keyword.split()
        if words and words[0][0].isupper() and len(words[0]) > 2:
            add_seed_item("brands", words[0])

    # Code extraction (hex patterns)
    code_pattern = r'([0-9a-fA-F#x]{3,})'
    codes = re.findall(code_pattern, keyword)
    for code in codes:
        if code.upper() not in [c.upper() for c in load_seed_list("codes")]:
            add_seed_item("codes", code.upper())

    # Model extraction (e.g., S7-1200)
    model_pattern = r'([A-Z][A-Z0-9\-]+[0-9])'
    models = re.findall(model_pattern, keyword)
    for model in models:
        if model not in load_seed_list("models"):
            add_seed_item("models", model)

    # Device extraction (common device keywords)
    device_keywords = ['cnc', 'drive', 'servo', 'hmi', 'plc', 'robot', 'inverter', 'controller']
    for dev in device_keywords:
        if dev in kw_lower and dev.capitalize() not in [d.lower() for d in load_seed_list("devices")]:
            add_seed_item("devices", dev.capitalize())

# ============================ SCORING ============================
def compute_score(volume: int, cpc: float, gap_score: float) -> float:
    """Higher volume, higher CPC, higher gap = better."""
    comp_factor = 1 / (1 - gap_score + 0.01)  # avoid div by zero
    return volume * cpc * comp_factor

# ============================ STORAGE ============================
def store_keyword(keyword: str, volume: int, cpc: float, gap_score: float, score: float):
    """Upsert keyword into astra_data."""
    try:
        supabase.table("astra_data").upsert({
            "keyword": keyword,
            "category": "technical error",  # could be refined
            "cpc_estimate": cpc,
            "search_volume": volume,
            "competition": 1 - gap_score,  # store as competition (0-1, higher means more comp)
            "score": score,
            "is_indexed": False,
            "created_at": datetime.utcnow().isoformat()
        }, on_conflict="keyword").execute()
        logger.info(f"Stored: {keyword} | Score: {score:.2f}")
    except Exception as e:
        logger.error(f"Failed to store {keyword}: {e}")

# ============================ MAIN HUNT CYCLE ============================
def hunt_cycle():
    logger.info("=== Starting Omniscient Hunt Cycle ===")

    # Load latest seeds
    brands = load_seed_list("brands")
    devices = load_seed_list("devices")
    models = load_seed_list("models")
    parts = load_seed_list("parts")
    codes = load_seed_list("codes")

    if not brands or not codes:
        logger.warning("Seed lists empty, can't generate keywords.")
        return

    # Generate seed queries by combining patterns with seeds (sampling to avoid explosion)
    seed_queries = []
    for pattern in PATTERNS:
        for brand in random.sample(brands, min(5, len(brands))):
            for device in random.sample(devices, min(3, len(devices))):
                for model in random.sample(models, min(3, len(models))):
                    for part in random.sample(parts, min(3, len(parts))):
                        for code in random.sample(codes, min(3, len(codes))):
                            q = pattern.format(brand=brand, device=device, model=model, part=part, code=code)
                            seed_queries.append(q)
    # Limit seeds per cycle
    random.shuffle(seed_queries)
    seed_queries = seed_queries[:30]  # 30 seeds per cycle to avoid overload

    logger.info(f"Generated {len(seed_queries)} seed queries.")

    for seed in seed_queries:
        logger.debug(f"Expanding seed: {seed}")
        suggestions = recursive_suggestion_expansion(seed)
        logger.info(f"Got {len(suggestions)} unique suggestions from seed.")

        for kw in suggestions:
            if len(kw) < 10:
                continue

            # 1. AI Validation
            is_valid, confidence = validate_with_ai(kw)
            if not is_valid or confidence < 0.6:
                logger.debug(f"AI rejected: {kw} (conf={confidence:.2f})")
                continue

            # 2. SERP gap analysis
            gap = analyze_serp_gap(kw)
            if gap < MIN_GAP_SCORE:
                logger.debug(f"Gap too low ({gap:.2f}): {kw}")
                continue

            # 3. Estimate metrics
            volume = estimate_search_volume(kw)
            cpc = estimate_cpc(kw)

            # 4. Score
            score = compute_score(volume, cpc, gap)

            # 5. Store if meets thresholds
            if volume >= MIN_SEARCH_VOLUME and cpc >= MIN_CPC:
                store_keyword(kw, volume, cpc, gap, score)

                # 6. Self-learn: extract new seeds
                extract_new_seeds(kw)

            # Random delay to be polite
            time.sleep(random.uniform(0.5, 1.2))

        # Delay between seeds
        time.sleep(random.uniform(3, 6))

    logger.info("=== Hunt cycle completed ===")

def continuous_hunt(interval_minutes: int = 60):
    while True:
        hunt_cycle()
        logger.info(f"Sleeping for {interval_minutes} minutes...")
        time.sleep(interval_minutes * 60)

# ============================ MAIN ============================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Astra Scout - Ultimate Keyword Hunter")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--interval", type=int, default=60, help="Minutes between cycles")
    args = parser.parse_args()

    # Initialize seed tables with defaults if empty
    initialize_seeds_if_empty()

    if args.once:
        hunt_cycle()
    else:
        continuous_hunt(args.interval)
