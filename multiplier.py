#!/usr/bin/env python3
"""
Astra Multiplier - The Infinity Ultra Pro Max Engine (Level ∞+)
Author: Your Name
Purpose: Generate infinite high-intent, high-CPC keyword combinations from seed tables.
         Features: Dynamic placeholder handling, memory-safe sampling, smart scoring,
                   batch upsert, failure-zero defaults, and detailed logging.
         Designed to feed the Synthesizer with gold-tier keywords.
"""

import os
import sys
import itertools
import logging
import random
import math
from datetime import datetime
from typing import List, Set, Dict, Any, Optional, Tuple
from collections import defaultdict

from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment
load_dotenv()

# ============================ CONFIGURATION ============================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Batch size for upsert (Supabase recommended < 1000)
BATCH_SIZE = 500

# Maximum combinations to generate per pattern (to avoid memory explosion)
# Set to None for unlimited, but be careful with huge seed lists
MAX_COMBOS_PER_PATTERN = 5000  # 50k per pattern is plenty

# Default seeds if tables are empty (ensures system never crashes)
DEFAULT_BRANDS = ["Siemens", "Fanuc", "Allen Bradley", "ABB", "Mitsubishi", "Yaskawa", "Omron", "Schneider Electric"]
DEFAULT_DEVICES = ["CNC", "drive", "servo", "HMI", "PLC", "controller", "robot", "inverter", "encoder", "motor"]
DEFAULT_MODELS = ["S7-1200", "S7-1500", "R30iA", "C1000", "ACS880", "G5", "NexGen", "RX3i", "Micro850", "KRC4"]
DEFAULT_PARTS = ["motor", "encoder", "power supply", "I/O module", "communication card", "driver", "capacitor", "fuse", "brake", "sensor"]
DEFAULT_CODES = ["16#0001", "SV0401", "F001", "ALM-402", "ERR-99", "E-101", "F002", "AL-001", "ER-12", "0x80070002", "E-Stop"]

# Score weights for different pattern types (higher = more valuable)
PATTERN_SCORES = {
    "error_fix": 100,      # e.g., "How to fix Siemens S7-1200 error 16#0001"
    "troubleshoot": 90,     # e.g., "Siemens drive fault F001 troubleshooting"
    "replacement": 80,      # e.g., "Fanuc CNC motor replacement guide"
    "installation": 70,     # e.g., "ABB drive installation manual"
    "symptoms": 60,         # e.g., "Siemens servo motor failure symptoms"
    "manual": 50,           # e.g., "Siemens S7-1200 manual PDF"
    "reset": 85,            # e.g., "Siemens error 16#0001 reset procedure"
    "diagnostic": 75,       # e.g., "Fanuc servo drive diagnostic codes"
    "wiring": 65,           # e.g., "Siemens S7-1200 wiring diagram"
    "parameter": 70,        # e.g., "Siemens drive parameter settings"
}

# Patterns organized by type (using placeholders: {brand}, {device}, {model}, {part}, {code})
PATTERNS = [
    # Error fix patterns (highest intent)
    ("error_fix", "How to fix {brand} {model} error {code}"),
    ("error_fix", "{brand} {device} fault {code} troubleshooting"),
    ("error_fix", "Resolve {brand} {part} error {code}"),
    ("error_fix", "{brand} {model} {code} fix guide"),
    ("error_fix", "{brand} {device} alarm {code} reset"),
    ("error_fix", "{brand} {model} {code} error code meaning"),

    # Troubleshooting patterns
    ("troubleshoot", "{brand} {model} {code} troubleshooting steps"),
    ("troubleshoot", "{brand} {device} {part} failure diagnosis"),
    ("troubleshoot", "How to diagnose {brand} {device} error {code}"),

    # Replacement & repair patterns
    ("replacement", "{brand} {device} {part} replacement procedure"),
    ("replacement", "{brand} {model} {part} repair manual"),
    ("replacement", "{brand} {device} {part} rebuild guide"),

    # Installation & setup
    ("installation", "{brand} {device} {part} installation guide"),
    ("installation", "{brand} {model} setup instructions"),
    ("installation", "{brand} {device} wiring diagram"),

    # Symptoms
    ("symptoms", "{brand} {device} {part} failure symptoms"),
    ("symptoms", "{brand} {model} error {code} symptoms"),

    # Manuals & documentation
    ("manual", "{brand} {model} {code} manual PDF"),
    ("manual", "{brand} {device} {part} technical data sheet"),
    ("manual", "{brand} {model} user manual"),

    # Diagnostic
    ("diagnostic", "{brand} {device} diagnostic code {code} meaning"),
    ("diagnostic", "{brand} {model} error {code} diagnostic guide"),

    # Wiring & parameter
    ("wiring", "{brand} {model} wiring diagram"),
    ("wiring", "{brand} {device} {part} connection diagram"),
    ("parameter", "{brand} {device} parameter settings for {code}"),
    ("parameter", "{brand} {model} {code} parameter list"),
]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AstraMultiplier")

# ============================ SUPABASE CLIENT ============================
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    logger.error("Supabase credentials missing. Exiting.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ============================ DATA FETCHING WITH FALLBACK ============================
def fetch_table_data(table_name: str, default_list: List[str]) -> List[str]:
    """Fetch names from Supabase table; if fails or empty, return default list."""
    try:
        response = supabase.table(table_name).select("name").execute()
        if response.data:
            # Clean and deduplicate
            names = list(set(item['name'].strip() for item in response.data if item['name']))
            logger.info(f"Fetched {len(names)} from {table_name}")
            return names
        else:
            logger.warning(f"Table '{table_name}' is empty. Using defaults.")
            return default_list
    except Exception as e:
        logger.error(f"Error fetching {table_name}: {e}. Using defaults.")
        return default_list

# ============================ KEYWORD GENERATION ============================
def generate_keywords_smart(
    brands: List[str],
    devices: List[str],
    models: List[str],
    parts: List[str],
    codes: List[str],
    max_combos: Optional[int] = MAX_COMBOS_PER_PATTERN
) -> Set[Tuple[str, int, str]]:
    """
    Generate keyword combinations with memory safety.
    Uses random sampling if total combos exceed max_combos.
    Returns set of (keyword, score, pattern_type).
    """
    keywords = set()
    total_combinations = 0

    # Prepare placeholder mapping
    placeholder_pools = {
        "{brand}": brands,
        "{device}": devices,
        "{model}": models,
        "{part}": parts,
        "{code}": codes
    }

    for pattern_type, pattern in PATTERNS:
        base_score = PATTERN_SCORES.get(pattern_type, 50)

        # Find which placeholders are active in this pattern
        active_placeholders = [p for p in placeholder_pools.keys() if p in pattern]
        active_lists = [placeholder_pools[p] for p in active_placeholders]

        if not active_placeholders:
            continue  # pattern with no placeholders? skip

        # Calculate total possible combos for this pattern
        total_possible = math.prod(len(lst) for lst in active_lists)
        logger.debug(f"Pattern '{pattern}': {total_possible} possible combinations")

        # If total possible is huge, we sample randomly to avoid memory blow
        if max_combos and total_possible > max_combos:
            logger.info(f"Pattern '{pattern}' has {total_possible} combos, sampling {max_combos} randomly.")
            # Generate random indices
            # We'll generate random combinations by random choice from each list
            sampled = set()
            attempts = 0
            max_attempts = max_combos * 3  # to avoid infinite loop if lists are small
            while len(sampled) < max_combos and attempts < max_attempts:
                # Randomly pick one from each active list
                combo = tuple(random.choice(lst) for lst in active_lists)
                sampled.add(combo)
                attempts += 1
            # Now generate keywords from sampled combos
            for combo in sampled:
                kw = pattern
                for i, placeholder in enumerate(active_placeholders):
                    kw = kw.replace(placeholder, combo[i])
                kw = ' '.join(kw.split())  # clean extra spaces
                score = base_score
                # Additional scoring tweaks
                if "error" in kw.lower() or "fault" in kw.lower():
                    score += 10
                if "replacement" in kw.lower() or "repair" in kw.lower():
                    score += 5
                if "manual" in kw.lower() or "pdf" in kw.lower():
                    score += 2
                # Add randomness to avoid identical scores
                score += random.randint(-3, 3)
                keywords.add((kw, score, pattern_type))
                total_combinations += 1
        else:
            # Generate all combinations using itertools.product
            for combo in itertools.product(*active_lists):
                kw = pattern
                for i, placeholder in enumerate(active_placeholders):
                    kw = kw.replace(placeholder, combo[i])
                kw = ' '.join(kw.split())
                score = base_score
                if "error" in kw.lower() or "fault" in kw.lower():
                    score += 10
                if "replacement" in kw.lower() or "repair" in kw.lower():
                    score += 5
                if "manual" in kw.lower() or "pdf" in kw.lower():
                    score += 2
                score += random.randint(-3, 3)
                keywords.add((kw, score, pattern_type))
                total_combinations += 1

    logger.info(f"Total unique keywords generated: {len(keywords)} (from {total_combinations} iterations)")
    return keywords

def prepare_batch(keywords_set: Set[tuple]) -> List[Dict]:
    """Convert set of (keyword, score, pattern_type) into list of dicts for upsert."""
    batch = []
    for kw, score, ptype in keywords_set:
        batch.append({
            "keyword": kw,
            "category": ptype.replace("_", " ").title(),
            "cpc_estimate": None,
            "search_volume": None,
            "competition": None,
            "score": score,
            "is_indexed": False,
            "created_at": datetime.utcnow().isoformat()
        })
    return batch

def upsert_batches(batch_data: List[Dict], batch_size: int = BATCH_SIZE):
    """Upsert data in batches to Supabase."""
    total = len(batch_data)
    logger.info(f"Starting upsert of {total} keywords in batches of {batch_size}...")
    for i in range(0, total, batch_size):
        batch = batch_data[i:i+batch_size]
        try:
            supabase.table("astra_data").upsert(batch, on_conflict="keyword").execute()
            logger.info(f"✅ Upserted batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ({len(batch)} keywords)")
        except Exception as e:
            logger.error(f"❌ Batch upsert failed at index {i}: {e}")
            # Optionally retry individually? For now log and continue.
    logger.info("✅ All batches processed.")

# ============================ MAIN ============================
def main(dry_run: bool = False, max_combos: Optional[int] = MAX_COMBOS_PER_PATTERN):
    logger.info("🚀 Astra Multiplier - Infinity Ultra Pro Max Engine Started")

    # 1. Fetch seeds with fallback defaults
    brands = fetch_table_data("brands", DEFAULT_BRANDS)
    devices = fetch_table_data("devices", DEFAULT_DEVICES)
    models = fetch_table_data("models", DEFAULT_MODELS)
    parts = fetch_table_data("parts", DEFAULT_PARTS)
    codes = fetch_table_data("codes", DEFAULT_CODES)

    logger.info(f"Loaded seeds: Brands={len(brands)}, Devices={len(devices)}, Models={len(models)}, Parts={len(parts)}, Codes={len(codes)}")

    # 2. Generate keywords
    keywords_set = generate_keywords_smart(brands, devices, models, parts, codes, max_combos)

    if not keywords_set:
        logger.warning("No keywords generated. Exiting.")
        return

    # 3. Prepare batch data
    batch_data = prepare_batch(keywords_set)

    if dry_run:
        logger.info(f"Dry run: would upsert {len(batch_data)} keywords. Sample:")
        for sample in sorted(batch_data, key=lambda x: -x['score'])[:10]:
            logger.info(f"  - {sample['keyword']} (score={sample['score']}, cat={sample['category']})")
        return

    # 4. Upsert to Supabase
    upsert_batches(batch_data)

    logger.info(f"🏆 Mission Accomplished: {len(batch_data)} keywords added/updated in astra_data.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Astra Multiplier - Generate infinite keywords")
    parser.add_argument("--dry-run", action="store_true", help="Generate but don't upsert")
    parser.add_argument("--max-combos", type=int, default=MAX_COMBOS_PER_PATTERN,
                        help=f"Max combinations per pattern (default {MAX_COMBOS_PER_PATTERN})")
    args = parser.parse_args()

    main(dry_run=args.dry_run, max_combos=args.max_combos)
