#!/usr/bin/env python3
"""
Astra Linker – The Authority Web Builder
Builds intelligent internal links between related articles based on brand, model, and error codes.
Enhances site structure for Google crawlers and improves EEAT signals.
"""

import os
import re
import logging
import time
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote_plus

import requests
from supabase import create_client, Client
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# Load environment
load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
BATCH_SIZE = int(os.getenv("LINKER_BATCH_SIZE", 20))
MAX_LINKS_PER_ARTICLE = int(os.getenv("MAX_LINKS_PER_ARTICLE", 5))
SLEEP_BETWEEN_UPDATES = float(os.getenv("LINKER_SLEEP", 1.0))  # seconds

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AstraLinker")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    logger.error("Supabase credentials missing")
    exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


class LinkerEngine:
    """Intelligent internal linker using entity extraction and semantic matching."""

    def __init__(self):
        self.processed_count = 0
        self.error_count = 0

    def extract_entities(self, keyword: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract brand, model, and error code from keyword."""
        brand_match = re.search(r'(Siemens|Fanuc|ABB|Allen Bradley|Mitsubishi|Omron|Schneider|Yaskawa)', keyword, re.IGNORECASE)
        model_match = re.search(r'([A-Z][A-Z0-9\-]+[0-9])', keyword)
        code_match = re.search(r'([0-9a-fA-F#x]{3,})', keyword)
        return (
            brand_match.group(0) if brand_match else None,
            model_match.group(0) if model_match else None,
            code_match.group(0) if code_match else None
        )

    def fetch_candidates(self, brand: Optional[str], model: Optional[str], code: Optional[str], current_id: str, limit: int = 5) -> List[Dict]:
        """Fetch related articles from Supabase based on entities."""
        candidates = []
        try:
            # Try exact entity matches first
            if brand:
                resp = supabase.table("astra_data") \
                    .select("id, keyword") \
                    .eq("entity_brand", brand) \
                    .neq("id", current_id) \
                    .limit(limit) \
                    .execute()
                if resp.data:
                    candidates.extend(resp.data)
            if model and len(candidates) < limit:
                resp = supabase.table("astra_data") \
                    .select("id, keyword") \
                    .eq("entity_model", model) \
                    .neq("id", current_id) \
                    .limit(limit - len(candidates)) \
                    .execute()
                if resp.data:
                    candidates.extend(resp.data)
            if code and len(candidates) < limit:
                resp = supabase.table("astra_data") \
                    .select("id, keyword") \
                    .eq("entity_code", code) \
                    .neq("id", current_id) \
                    .limit(limit - len(candidates)) \
                    .execute()
                if resp.data:
                    candidates.extend(resp.data)

            # Fallback to text search if still few
            if len(candidates) < 3:
                # Use full-text search on keyword
                search_term = brand or model or code or ""
                if search_term:
                    resp = supabase.table("astra_data") \
                        .select("id, keyword") \
                        .text_search("keyword", search_term) \
                        .neq("id", current_id) \
                        .limit(limit - len(candidates)) \
                        .execute()
                    if resp.data:
                        candidates.extend(resp.data)
        except Exception as e:
            logger.error(f"Error fetching candidates: {e}")
        # Remove duplicates by id
        seen = set()
        unique = []
        for c in candidates:
            if c['id'] not in seen:
                seen.add(c['id'])
                unique.append(c)
        return unique[:limit]

    def build_link_html(self, candidates: List[Dict]) -> str:
        """Generate HTML for internal links section."""
        if not candidates:
            return ""
        html = '<div class="astra-internal-links"><h4>🔍 Related Troubleshooting Manuals</h4><ul>'
        for art in candidates:
            slug = quote_plus(art['keyword'].lower().replace(' ', '-'))
            html += f'<li><a href="/troubleshoot/{slug}">{art["keyword"]}</a></li>'
        html += '</ul></div>'
        return html

    def inject_links(self, content: str, links_html: str) -> str:
        """Insert the links section before the closing tags."""
        if not links_html:
            return content
        # Place it before the final schemas or at the end of the article body
        soup = BeautifulSoup(content, 'html.parser')
        # Look for a good insertion point: before the first <script> after the body, or at body end
        body = soup.find('body')
        if body:
            # Insert as last element in body
            body.append(BeautifulSoup(links_html, 'html.parser'))
        else:
            # Fallback: append at the end
            content += "\n" + links_html
        return str(soup)

    def process_article(self, article: Dict) -> bool:
        """Process a single article: find related articles and inject links."""
        art_id = article['id']
        keyword = article['keyword']
        content = article.get('content')
        if not content:
            logger.debug(f"Article {art_id} has no content, skipping")
            return False

        brand, model, code = self.extract_entities(keyword)
        candidates = self.fetch_candidates(brand, model, code, art_id, MAX_LINKS_PER_ARTICLE)
        if not candidates:
            logger.debug(f"No candidates for {keyword}")
            return False

        links_html = self.build_link_html(candidates)
        new_content = self.inject_links(content, links_html)

        # Update Supabase
        try:
            supabase.table("astra_data") \
                .update({"content": new_content, "updated_at": "now()"}) \
                .eq("id", art_id) \
                .execute()
            logger.info(f"✅ Linked {keyword} with {len(candidates)} related articles")
            return True
        except Exception as e:
            logger.error(f"Failed to update {art_id}: {e}")
            return False

    def run_cycle(self):
        """Main loop: fetch articles needing linking (those with content but maybe without links)."""
        logger.info("=== Astra Linker Cycle Started ===")
        try:
            # Fetch articles that have content but no links (simple heuristic: we'll reprocess all with content)
            # In production, you'd maintain a flag `is_linked` or check for existence of the links div.
            # Here we simply fetch the latest N with content.
            response = supabase.table("astra_data") \
                .select("id, keyword, content") \
                .not_.is_("content", "null") \
                .order("created_at", desc=True) \
                .limit(BATCH_SIZE) \
                .execute()
            articles = response.data if response else []
        except Exception as e:
            logger.error(f"Failed to fetch articles: {e}")
            return

        logger.info(f"Found {len(articles)} articles to process")
        for art in articles:
            success = self.process_article(art)
            if success:
                self.processed_count += 1
            else:
                self.error_count += 1
            time.sleep(SLEEP_BETWEEN_UPDATES)

        logger.info(f"Cycle complete. Processed: {self.processed_count}, Errors: {self.error_count}")


if __name__ == "__main__":
    engine = LinkerEngine()
    engine.run_cycle()
