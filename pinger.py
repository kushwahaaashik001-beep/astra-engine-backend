#!/usr/bin/env python3
"""
Astra Pinger – The Crawler Bait
Pings Google, Bing, and IndexNow to alert search engines of new/updated content.
Uses retries, respects rate limits, and logs all responses.
"""

import os
import time
import logging
from typing import List, Dict
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration
DOMAIN = os.getenv("SITE_DOMAIN", "https://your-astra-site.com")
SITEMAP_URL = urljoin(DOMAIN, "sitemap.xml")
PING_INTERVAL = int(os.getenv("PING_INTERVAL", 3600))  # seconds between pings
INDEXNOW_KEY = os.getenv("INDEXNOW_KEY")  # optional, for IndexNow
USER_AGENT = "AstraPinger/1.0 (+https://astra.com)"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AstraPinger")


class PingerEngine:
    """Pings search engines with retries and exponential backoff."""

    def __init__(self):
        self.ping_urls = [
            f"https://www.google.com/ping?sitemap={SITEMAP_URL}",
            f"https://www.bing.com/ping?sitemap={SITEMAP_URL}",
        ]
        if INDEXNOW_KEY:
            self.ping_urls.append(f"https://api.indexnow.org/indexnow?url={DOMAIN}&key={INDEXNOW_KEY}&keyLocation={DOMAIN}/indexnow.txt")

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.results: Dict[str, bool] = {}

    def ping_with_retry(self, url: str, max_retries: int = 3) -> bool:
        """Ping a URL with exponential backoff."""
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.session.get(url, timeout=15)
                if resp.status_code == 200:
                    logger.info(f"✅ Ping successful: {url}")
                    return True
                else:
                    logger.warning(f"⚠️ Ping to {url} returned {resp.status_code}")
                    if attempt < max_retries:
                        sleep_time = 2 ** attempt
                        logger.info(f"Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                    else:
                        return False
            except requests.exceptions.RequestException as e:
                logger.error(f"🚨 Ping failed (attempt {attempt}): {e}")
                if attempt < max_retries:
                    sleep_time = 2 ** attempt
                    time.sleep(sleep_time)
                else:
                    return False
        return False

    def run_ping(self):
        """Ping all configured engines."""
        logger.info("📣 Astra Pinger: Alerting search engines...")
        success_count = 0
        for url in self.ping_urls:
            ok = self.ping_with_retry(url)
            self.results[url] = ok
            if ok:
                success_count += 1
            time.sleep(2)  # polite delay between pings

        logger.info(f"Ping complete. {success_count}/{len(self.ping_urls)} successful.")
        return self.results

    def log_results(self):
        """Write results to a log file for monitoring."""
        log_entry = {
            "timestamp": time.time(),
            "results": self.results,
            "domain": DOMAIN
        }
        with open("pinger.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")


if __name__ == "__main__":
    import json
    engine = PingerEngine()
    engine.run_ping()
    engine.log_results()
