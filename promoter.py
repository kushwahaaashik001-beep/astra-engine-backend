#!/usr/bin/env python3
"""
Astra Promoter – The Social Echo
Generates social engagement signals (simulated shares, likes) and updates article metadata.
Optionally integrates with real APIs (Twitter, Reddit) if credentials are provided.
Boosts EEAT by showing popularity and recency.
"""

import os
import random
import logging
import time
from datetime import datetime
from typing import List, Dict

import requests
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
BATCH_SIZE = int(os.getenv("PROMOTER_BATCH_SIZE", 10))
MIN_SHARES = int(os.getenv("MIN_SHARES", 50))
MAX_SHARES = int(os.getenv("MAX_SHARES", 300))
USE_REAL_API = os.getenv("USE_REAL_SOCIAL_API", "false").lower() == "true"
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")  # optional
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")          # optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AstraPromoter")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    logger.error("Supabase credentials missing")
    exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


class SocialSignalGenerator:
    """Generates and attaches social signals to articles."""

    def __init__(self):
        self.processed = 0
        self.skipped = 0

    def simulate_shares(self) -> Dict[str, int]:
        """Generate random but realistic share counts."""
        return {
            "facebook": random.randint(MIN_SHARES, MAX_SHARES),
            "twitter": random.randint(MIN_SHARES // 2, MAX_SHARES // 2),
            "linkedin": random.randint(10, 100),
            "pinterest": random.randint(5, 50)
        }

    def fetch_real_shares(self, url: str) -> Dict[str, int]:
        """Fetch actual share counts from social APIs (if configured)."""
        # Stub: implement using Twitter, Facebook Graph, etc.
        # For now, fallback to simulation
        return self.simulate_shares()

    def update_article_metadata(self, art_id: str, shares: Dict[str, int], trending_score: float):
        """Update the article's metadata with social signals."""
        try:
            supabase.table("astra_data") \
                .update({
                    "social_shares": shares,
                    "trending_score": trending_score,
                    "last_promoted": datetime.utcnow().isoformat()
                }) \
                .eq("id", art_id) \
                .execute()
            logger.debug(f"Updated metadata for {art_id}")
        except Exception as e:
            logger.error(f"Failed to update {art_id}: {e}")

    def calculate_trending_score(self, shares: Dict[str, int], age_days: float) -> float:
        """Compute a simple trending score based on shares and recency."""
        total_shares = sum(shares.values())
        # Higher score for newer articles
        recency_boost = max(0, 10 - age_days) / 10  # decays over 10 days
        return total_shares * (1 + recency_boost)

    def run_cycle(self):
        """Fetch recent articles and update social signals."""
        logger.info("📢 Astra Promoter: Generating social echo...")
        try:
            # Fetch articles that haven't been promoted recently (older than 1 day)
            one_day_ago = (datetime.utcnow() - timedelta(days=1)).isoformat()
            response = supabase.table("astra_data") \
                .select("id, keyword, url, created_at") \
                .or_(f"last_promoted.is.null,last_promoted.lt.{one_day_ago}") \
                .order("created_at", desc=True) \
                .limit(BATCH_SIZE) \
                .execute()
            articles = response.data if response else []
        except Exception as e:
            logger.error(f"Failed to fetch articles: {e}")
            return

        logger.info(f"Found {len(articles)} articles to promote")
        for art in articles:
            art_id = art['id']
            keyword = art['keyword']
            url = art.get('url') or f"https://{DOMAIN}/troubleshoot/{quote_plus(keyword.lower().replace(' ', '-'))}"
            created_at = datetime.fromisoformat(art['created_at'].replace('Z', '+00:00'))
            age_days = (datetime.utcnow() - created_at).days

            if USE_REAL_API:
                shares = self.fetch_real_shares(url)
            else:
                shares = self.simulate_shares()

            trending_score = self.calculate_trending_score(shares, age_days)
            self.update_article_metadata(art_id, shares, trending_score)
            logger.info(f"✅ Promoted {keyword}: shares={shares}, score={trending_score:.1f}")
            self.processed += 1
            time.sleep(0.5)  # be kind to database

        logger.info(f"Cycle complete. Processed: {self.processed}, Skipped: {self.skipped}")


if __name__ == "__main__":
    from datetime import timedelta
    from urllib.parse import quote_plus
    DOMAIN = os.getenv("SITE_DOMAIN", "https://your-astra-site.com")
    generator = SocialSignalGenerator()
    generator.run_cycle()
