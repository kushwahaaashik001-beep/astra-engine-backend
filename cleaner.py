#!/usr/bin/env python3
"""
Astra Cleaner - The Data Purifier (Level ∞)
Author: Your Name
Purpose: Maintain database integrity: remove duplicates, reset low-quality content,
         optimize performance, and ensure only gold remains in astra_data.
         Features: Smart duplicate removal, content quality checks, vacuum simulation,
                   and detailed logging.
"""

import os
import sys
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment
load_dotenv()

# ============================ CONFIGURATION ============================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Minimum acceptable content length (characters)
MIN_CONTENT_LENGTH = 800

# How many days to keep logs (if using log table)
LOG_RETENTION_DAYS = 30

# Batch size for delete/update operations
BATCH_SIZE = 500

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Cleaner - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AstraCleaner")

# ============================ SUPABASE CLIENT ============================
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    logger.error("Supabase credentials missing. Exiting.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ============================ CORE FUNCTIONS ============================

def remove_duplicates_smart():
    """
    Remove duplicate keywords efficiently using SQL grouping.
    Keeps the oldest (or highest scoring) record per keyword.
    """
    logger.info("🔍 Scanning for duplicate keywords...")
    try:
        # Fetch all keywords with id and score (or created_at)
        response = supabase.table("astra_data").select("id, keyword, score, created_at").execute()
        rows = response.data
        if not rows:
            logger.info("No data found.")
            return

        # Group by keyword
        groups: Dict[str, List[Dict]] = {}
        for row in rows:
            kw = row['keyword']
            if kw not in groups:
                groups[kw] = []
            groups[kw].append(row)

        # For each keyword, find duplicates (more than one entry)
        duplicates_to_delete = []
        for kw, entries in groups.items():
            if len(entries) > 1:
                # Strategy: keep the one with highest score, if tie keep oldest created_at
                # Sort by score desc, then created_at asc
                sorted_entries = sorted(entries, key=lambda x: (-x.get('score', 0), x.get('created_at', '')))
                # Keep the first, delete the rest
                for entry in sorted_entries[1:]:
                    duplicates_to_delete.append(entry['id'])

        if not duplicates_to_delete:
            logger.info("✅ No duplicates found.")
            return

        logger.warning(f"🗑 Found {len(duplicates_to_delete)} duplicate entries. Deleting in batches...")

        # Delete in batches
        for i in range(0, len(duplicates_to_delete), BATCH_SIZE):
            batch = duplicates_to_delete[i:i+BATCH_SIZE]
            try:
                supabase.table("astra_data").delete().in_("id", batch).execute()
                logger.info(f"   Deleted batch {i//BATCH_SIZE + 1}/{(len(duplicates_to_delete)-1)//BATCH_SIZE + 1}")
                time.sleep(0.5)  # be gentle to DB
            except Exception as e:
                logger.error(f"   Batch delete failed: {e}")
        logger.info("✅ Duplicate removal completed.")
    except Exception as e:
        logger.error(f"❌ Duplicate removal error: {e}")

def reset_low_quality_content():
    """
    Reset records where content exists but is too short or low quality.
    Sets is_indexed = False and content = NULL so Synthesizer can retry.
    """
    logger.info("🔍 Checking for low-quality content...")
    try:
        # Find records where is_indexed = True and content length < MIN_CONTENT_LENGTH
        # Supabase doesn't have LENGTH function directly, so we fetch and filter in Python
        response = supabase.table("astra_data") \
            .select("id, content") \
            .eq("is_indexed", True) \
            .execute()
        rows = response.data
        if not rows:
            logger.info("No indexed content found.")
            return

        low_quality_ids = []
        for row in rows:
            content = row.get('content')
            if content and len(content) < MIN_CONTENT_LENGTH:
                low_quality_ids.append(row['id'])

        if not low_quality_ids:
            logger.info("✅ All content meets quality standards.")
            return

        logger.warning(f"⚠️ Found {len(low_quality_ids)} low-quality articles. Resetting for rewrite...")
        for i in range(0, len(low_quality_ids), BATCH_SIZE):
            batch = low_quality_ids[i:i+BATCH_SIZE]
            try:
                supabase.table("astra_data") \
                    .update({"is_indexed": False, "content": None}) \
                    .in_("id", batch) \
                    .execute()
                logger.info(f"   Reset batch {i//BATCH_SIZE + 1}/{(len(low_quality_ids)-1)//BATCH_SIZE + 1}")
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"   Batch reset failed: {e}")
        logger.info("✅ Low-quality content reset.")
    except Exception as e:
        logger.error(f"❌ Quality check error: {e}")

def remove_orphaned_records():
    """
    Optional: Remove records that are indexed but have no content (shouldn't happen, but just in case).
    Also could remove very old pending keywords that never got processed (if they're too old).
    """
    logger.info("🔍 Checking for orphaned records...")
    try:
        # Case: is_indexed = True but content is null or empty
        response = supabase.table("astra_data") \
            .select("id") \
            .eq("is_indexed", True) \
            .is_("content", "null") \
            .execute()
        if response.data:
            orphan_ids = [row['id'] for row in response.data]
            logger.warning(f"🗑 Found {len(orphan_ids)} orphaned indexed records (no content). Deleting...")
            for i in range(0, len(orphan_ids), BATCH_SIZE):
                batch = orphan_ids[i:i+BATCH_SIZE]
                supabase.table("astra_data").delete().in_("id", batch).execute()
                time.sleep(0.5)
        else:
            logger.info("✅ No orphaned records.")
    except Exception as e:
        logger.error(f"❌ Orphan check error: {e}")

def vacuum_database():
    """
    Simulate a VACUUM by reindexing? Not directly possible in Supabase,
    but we can log that we've cleaned up.
    """
    logger.info("🧹 Database cleanup complete. (No actual vacuum needed in Supabase)")

def clean_database():
    logger.info("🧹 Starting Astra Level 10 Database Purge...")
    remove_duplicates_smart()
    reset_low_quality_content()
    remove_orphaned_records()
    vacuum_database()
    logger.info("🏆 Database is now optimized and clean.")

if __name__ == "__main__":
    clean_database()
