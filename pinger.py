#!/usr/bin/env python3
"""
Astra Pinger v4 – The Omniscient Zero‑API Indexing Engine
Generates RSS feed, pings IndexNow in bulk, checks content hashes, uses jittered retries,
and ensures perfect RFC 822 dates and robust slug generation.
Monitors ping results and adapts to failures. All free, no API limits, ultra fast.
"""

import os
import re
import time
import json
import hashlib
import logging
import random
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
from datetime import datetime, timedelta
from xml.dom import minidom
from xml.etree import ElementTree as ET
from email.utils import formatdate

import requests
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DOMAIN = os.getenv("SITE_DOMAIN", "https://your-astra-site.com")
SITEMAP_URL = urljoin(DOMAIN, "sitemap.xml")
RSS_URL = urljoin(DOMAIN, "rss.xml")
INDEXNOW_KEY = os.getenv("INDEXNOW_KEY")           # required for IndexNow
INDEXNOW_KEY_LOCATION = os.getenv("INDEXNOW_KEY_LOCATION", f"{DOMAIN}/indexnow.txt")

# Supabase for tracking state
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Paths to save RSS (and optionally trigger sitemap generation)
PUBLIC_DIR = os.getenv("PUBLIC_DIR", "./public")   # where frontend files live
RSS_FILE = os.path.join(PUBLIC_DIR, "rss.xml")

# RSS feed settings
RSS_MAX_ITEMS = int(os.getenv("RSS_MAX_ITEMS", 20))
RSS_TITLE = os.getenv("RSS_TITLE", "Astra Troubleshooting Updates")
RSS_DESCRIPTION = os.getenv("RSS_DESCRIPTION", "Latest industrial troubleshooting guides")

# Ping settings
MAX_URLS_PER_INDEXNOW = 10000                      # IndexNow limit
PING_RETRIES = int(os.getenv("PING_RETRIES", 3))
PING_BACKOFF_FACTOR = float(os.getenv("PING_BACKOFF_FACTOR", 2.0))
PING_JITTER = float(os.getenv("PING_JITTER", 0.5))

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AstraPinger")

# ------------------------------------------------------------------
# PingerEngine class
# ------------------------------------------------------------------
class PingerEngine:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "AstraPinger/4.0 (+https://astra.com)"})
        self.results: Dict[str, bool] = {}
        self.ping_history: List[Dict] = []          # store recent ping attempts

        # Ensure public directory exists
        os.makedirs(PUBLIC_DIR, exist_ok=True)

        # Check IndexNow key presence
        if not INDEXNOW_KEY:
            logger.warning("INDEXNOW_KEY not set – IndexNow pings will be skipped.")

    # ------------------------------------------------------------------
    # 1. Content hash tracking (avoid duplicate pings)
    # ------------------------------------------------------------------
    def get_content_hash(self, article: Dict) -> str:
        """Generate a hash of the article's content and metadata."""
        content = article.get('content', '')
        key = article.get('keyword', '') + content + article.get('updated_at', '')
        return hashlib.sha256(key.encode()).hexdigest()

    def needs_ping(self, article: Dict) -> bool:
        """Check if the article's content has changed since last ping."""
        if not supabase:
            return True
        try:
            last_hash = article.get('content_hash')
            current_hash = self.get_content_hash(article)
            return last_hash != current_hash
        except Exception as e:
            logger.error(f"Hash check failed: {e}")
            return True

    def update_ping_state(self, article: Dict, success: bool):
        """Store the new content hash, ping time, and result in the database."""
        if not supabase:
            return
        try:
            supabase.table("astra_data").update({
                "content_hash": self.get_content_hash(article),
                "last_pinged": datetime.utcnow().isoformat(),
                "last_ping_success": success
            }).eq("id", article['id']).execute()
        except Exception as e:
            logger.error(f"Failed to update ping state: {e}")

    # ------------------------------------------------------------------
    # 2. Fetch recently updated articles
    # ------------------------------------------------------------------
    def fetch_updated_articles(self, limit: int = 50) -> List[Dict]:
        """Retrieve articles that have changed since last ping."""
        if not supabase:
            return []
        try:
            # Articles where content_hash is null OR last_pinged is older than 7 days
            seven_days_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
            response = supabase.table("astra_data") \
                .select("id, keyword, content, updated_at, content_hash, last_pinged") \
                .or_(f"content_hash.is.null,last_pinged.lt.{seven_days_ago}") \
                .order("updated_at", desc=True) \
                .limit(limit) \
                .execute()
            return response.data if response else []
        except Exception as e:
            logger.error(f"Failed to fetch updated articles: {e}")
            return []

    # ------------------------------------------------------------------
    # 3. Generate RSS feed (RFC 822 dates, robust slug)
    # ------------------------------------------------------------------
    def generate_rss(self, articles: List[Dict]) -> str:
        """
        Create an RSS 2.0 feed from the latest articles.
        Uses email.utils.formatdate for correct RFC 822 timestamps.
        """
        rss = ET.Element("rss", version="2.0")
        channel = ET.SubElement(rss, "channel")
        ET.SubElement(channel, "title").text = RSS_TITLE
        ET.SubElement(channel, "link").text = DOMAIN
        ET.SubElement(channel, "description").text = RSS_DESCRIPTION
        ET.SubElement(channel, "language").text = "en-us"
        ET.SubElement(channel, "lastBuildDate").text = formatdate(localtime=False)

        for art in articles[:RSS_MAX_ITEMS]:
            if not art.get('content') or len(art['content']) < 200:
                continue   # skip low‑quality or empty articles

            slug = self._make_slug(art['keyword'])
            url = urljoin(DOMAIN, f"/troubleshoot/{slug}")
            title = art['keyword']
            pub_date = art.get('updated_at', datetime.utcnow().isoformat())

            # Parse ISO date to timestamp for formatdate
            try:
                dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                rss_date = formatdate(dt.timestamp(), localtime=False)
            except:
                rss_date = formatdate(localtime=False)

            item = ET.SubElement(channel, "item")
            ET.SubElement(item, "title").text = title
            ET.SubElement(item, "link").text = url
            ET.SubElement(item, "guid", isPermaLink="true").text = url
            ET.SubElement(item, "pubDate").text = rss_date
            ET.SubElement(item, "description").text = f"Complete guide to {title}"

        # Pretty print XML
        rough_string = ET.tostring(rss, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        # Save to file
        with open(RSS_FILE, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
        logger.info(f"✅ RSS feed updated at {RSS_FILE} with {len(articles[:RSS_MAX_ITEMS])} items")

        return pretty_xml

    def _make_slug(self, keyword: str) -> str:
        """
        Generate a safe, consistent slug.
        Removes all non‑alphanumeric characters (except spaces) and collapses hyphens.
        """
        # Replace special characters with nothing, keep letters, numbers, spaces
        cleaned = re.sub(r'[^a-zA-Z0-9\s-]', '', keyword).strip().lower()
        # Replace spaces and multiple hyphens with a single hyphen
        slug = re.sub(r'[\s-]+', '-', cleaned)
        return slug

    # ------------------------------------------------------------------
    # 4. IndexNow bulk POST (fast, free, unlimited)
    # ------------------------------------------------------------------
    def ping_indexnow_post(self, url_list: List[str]) -> bool:
        """
        IndexNow POST ping (bulk up to 10,000 URLs per request).
        Uses JSON payload. This is the fastest free method.
        """
        if not INDEXNOW_KEY or not url_list:
            return False

        # Split into batches if necessary
        success_all = True
        for i in range(0, len(url_list), MAX_URLS_PER_INDEXNOW):
            batch = url_list[i:i+MAX_URLS_PER_INDEXNOW]
            payload = {
                "host": urlparse(DOMAIN).netloc,
                "key": INDEXNOW_KEY,
                "keyLocation": INDEXNOW_KEY_LOCATION,
                "urlList": batch
            }
            try:
                resp = self.session.post("https://api.indexnow.org/indexnow",
                                         json=payload, timeout=15)
                if resp.status_code in (200, 202):
                    logger.info(f"✅ IndexNow batch {i//MAX_URLS_PER_INDEXNOW+1} successful for {len(batch)} URLs")
                    self.results[f"indexnow_batch_{i}"] = True
                else:
                    logger.warning(f"IndexNow batch returned {resp.status_code}: {resp.text}")
                    self.results[f"indexnow_batch_{i}"] = False
                    success_all = False
                time.sleep(1)   # be gentle
            except Exception as e:
                logger.error(f"IndexNow POST exception: {e}")
                self.results[f"indexnow_batch_{i}"] = False
                success_all = False
        return success_all

    # ------------------------------------------------------------------
    # 5. Legacy sitemap ping (still useful)
    # ------------------------------------------------------------------
    def ping_legacy_sitemap(self) -> bool:
        """Ping Google and Bing with sitemap URL (legacy)."""
        google_url = f"https://www.google.com/ping?sitemap={SITEMAP_URL}"
        bing_url = f"https://www.bing.com/ping?sitemap={SITEMAP_URL}"
        success = True
        if not self._ping_with_retry(google_url, "google_sitemap"):
            success = False
        time.sleep(2)
        if not self._ping_with_retry(bing_url, "bing_sitemap"):
            success = False
        return success

    def _ping_with_retry(self, url: str, name: str) -> bool:
        """Internal: GET request with exponential backoff + jitter and result tracking."""
        for attempt in range(1, PING_RETRIES + 1):
            try:
                resp = self.session.get(url, timeout=15)
                if resp.status_code == 200:
                    logger.debug(f"Ping successful: {name}")
                    self.results[name] = True
                    return True
                else:
                    logger.warning(f"Ping {name} returned {resp.status_code}")
                    if attempt < PING_RETRIES:
                        sleep_time = (PING_BACKOFF_FACTOR ** attempt) + (random.random() * PING_JITTER)
                        logger.info(f"Retrying {name} in {sleep_time:.1f}s...")
                        time.sleep(sleep_time)
                    else:
                        self.results[name] = False
                        return False
            except requests.exceptions.RequestException as e:
                logger.error(f"Ping {name} failed (attempt {attempt}): {e}")
                if attempt < PING_RETRIES:
                    sleep_time = (PING_BACKOFF_FACTOR ** attempt) + (random.random() * PING_JITTER)
                    time.sleep(sleep_time)
                else:
                    self.results[name] = False
                    return False
        return False

    # ------------------------------------------------------------------
    # 6. Trigger sitemap regeneration (optional)
    # ------------------------------------------------------------------
    def trigger_sitemap_update(self):
        """
        Notify the sitemap generator (if separate) to rebuild sitemap.
        Could be a local HTTP call or simply touching a file.
        """
        # Example: touch a file that a sitemap generator watches
        sitemap_trigger = os.path.join(PUBLIC_DIR, ".sitemap_trigger")
        with open(sitemap_trigger, 'w') as f:
            f.write(datetime.utcnow().isoformat())
        logger.info("Sitemap update triggered (touch file).")

    # ------------------------------------------------------------------
    # 7. Log ping results to a separate table (optional)
    # ------------------------------------------------------------------
    def log_ping_results(self):
        """Store ping outcomes in Supabase for monitoring."""
        if not supabase:
            return
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "results": json.dumps(self.results)
            }
            supabase.table("ping_log").insert(log_entry).execute()
        except Exception as e:
            logger.error(f"Failed to log ping results: {e}")

    # ------------------------------------------------------------------
    # 8. Main orchestration
    # ------------------------------------------------------------------
    def run_cycle(self):
        """Main ping cycle: fetch changed articles, generate RSS, ping IndexNow, and update state."""
        logger.info("=== Astra Pinger v4 Cycle Started ===")

        # Reset results for this cycle
        self.results = {}

        # Fetch articles that need pinging
        articles = self.fetch_updated_articles()
        if not articles:
            logger.info("No articles need pinging.")
        else:
            logger.info(f"Found {len(articles)} articles to ping.")

            # Prepare URL list for bulk IndexNow
            urls_to_ping = []
            articles_to_update = []
            for art in articles:
                if not self.needs_ping(art):
                    continue
                slug = self._make_slug(art['keyword'])
                url = urljoin(DOMAIN, f"/troubleshoot/{slug}")
                urls_to_ping.append(url)
                articles_to_update.append(art)

            if urls_to_ping:
                success = self.ping_indexnow_post(urls_to_ping)
                for art in articles_to_update:
                    self.update_ping_state(art, success)

            # Generate RSS feed from the latest articles (even unchanged ones)
            # to keep RSS fresh.
            self.generate_rss(articles)

        # Legacy sitemap ping (still useful for Google)
        self.ping_legacy_sitemap()

        # Trigger sitemap rebuild (optional)
        self.trigger_sitemap_update()

        # Log results for monitoring
        self.log_ping_results()

        logger.info("=== Cycle Complete ===")
        logger.info(f"Ping results: {self.results}")


if __name__ == "__main__":
    engine = PingerEngine()
    engine.run_cycle()
