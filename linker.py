#!/usr/bin/env python3
"""
Astra Linker v4 – The Intelligent Internal Linking Engine
Builds hyper‑relevant internal links with:
  - Reverse freshness push (old pages link to new ones)
  - Semantic anchor diversity with action verbs
  - Contextual placement where brand/model/code appear
  - Link throttling based on content length
  - Dead‑link caching for speed
  - Silo‑strict linking (same brand only)
All free, no APIs, optimised for EEAT and crawl efficiency.
"""

import os
import re
import time
import random
import logging
from typing import List, Dict, Optional, Tuple, Set
from urllib.parse import quote_plus, urljoin
from datetime import datetime

import requests
from supabase import create_client, Client
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# Load environment
load_dotenv()

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SITE_DOMAIN = os.getenv("SITE_DOMAIN", "https://your-astra-site.com")

# Batch settings
BATCH_SIZE = int(os.getenv("LINKER_BATCH_SIZE", 20))
MAX_LINKS_PER_ARTICLE = int(os.getenv("MAX_LINKS_PER_ARTICLE", 5))
MIN_LINKS_PER_ARTICLE = int(os.getenv("MIN_LINKS_PER_ARTICLE", 2))
LINK_DENSITY_FACTOR = float(os.getenv("LINK_DENSITY_FACTOR", 0.01))  # 1 link per ~100 words

# Reverse linking: number of old articles to update per new article
REVERSE_LINKS_PER_ARTICLE = int(os.getenv("REVERSE_LINKS_PER_ARTICLE", 2))

# Performance
SLEEP_BETWEEN_UPDATES = float(os.getenv("LINKER_SLEEP", 0.5))
ENABLE_DEAD_LINK_CHECK = os.getenv("ENABLE_DEAD_LINK_CHECK", "true").lower() == "true"
DEAD_LINK_CACHE_TTL = int(os.getenv("DEAD_LINK_CACHE_TTL", 86400))  # 24 hours
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 5))

# Logging
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
    """
    Intelligent internal linker with advanced features.
    """

    def __init__(self):
        self.processed_count = 0
        self.error_count = 0
        self.reverse_processed = 0
        self.dead_link_cache: Dict[str, float] = {}   # url -> timestamp of last check
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "AstraLinker/4.0"})

    # ------------------------------------------------------------------
    # 1. Entity extraction (enhanced)
    # ------------------------------------------------------------------
    def extract_entities(self, keyword: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Extract brand, model, error code, and language hint from keyword.
        """
        brand_match = re.search(
            r'(Siemens|Fanuc|ABB|Allen Bradley|Mitsubishi|Omron|Schneider|Yaskawa|Rockwell)',
            keyword, re.IGNORECASE
        )
        model_match = re.search(r'([A-Z][A-Z0-9\-]+[0-9])', keyword)
        code_match = re.search(r'([0-9a-fA-F#x]{3,})', keyword)
        # Simple language detection (could be extended with DB column)
        lang = "en"
        return (
            brand_match.group(0) if brand_match else None,
            model_match.group(0) if model_match else None,
            code_match.group(0) if code_match else None,
            lang
        )

    # ------------------------------------------------------------------
    # 2. Fetch candidate articles (silo + traffic + freshness)
    # ------------------------------------------------------------------
    def fetch_candidates(self, brand: Optional[str], model: Optional[str],
                         code: Optional[str], lang: str, current_id: str,
                         limit: int = 5, older_than: Optional[str] = None) -> List[Dict]:
        """
        Retrieve related articles respecting silo (same brand) and sorted by:
          - trending_score (if exists)
          - view_count
          - created_at (freshness)
        Optionally filter to articles older than a given timestamp (for reverse linking).
        """
        candidates = []
        try:
            query = supabase.table("astra_data") \
                .select("id, keyword, view_count, trending_score, url, language, created_at, content") \
                .neq("id", current_id)

            if brand:
                query = query.eq("entity_brand", brand)
            if lang:
                query = query.eq("language", lang)

            if older_than:
                query = query.lt("created_at", older_than)

            # Order by popularity and freshness
            query = query.order("trending_score", desc=True, nullsfirst=False) \
                         .order("view_count", desc=True, nullsfirst=False) \
                         .order("created_at", desc=True) \
                         .limit(limit * 3)

            response = query.execute()
            if response.data:
                candidates = response.data
        except Exception as e:
            logger.error(f"Error fetching candidates: {e}")

        # Filter out dead links (cached check)
        if ENABLE_DEAD_LINK_CHECK:
            candidates = [c for c in candidates if self._is_url_alive(c.get("url"))]

        # Ensure meaningful content
        candidates = [c for c in candidates if c.get('content') and len(c['content']) > 500]

        return candidates[:limit]

    def _is_url_alive(self, url: Optional[str]) -> bool:
        """Check if a URL is accessible (200 OK) with caching."""
        if not url:
            return False
        now = time.time()
        # Use cache if recent
        if url in self.dead_link_cache:
            last_check = self.dead_link_cache[url]
            if now - last_check < DEAD_LINK_CACHE_TTL:
                return True   # we treat as alive (or you could store boolean; here we only cache dead ones)
            else:
                del self.dead_link_cache[url]  # expired

        try:
            resp = self.session.head(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            if resp.status_code == 200:
                return True
            else:
                self.dead_link_cache[url] = now
                return False
        except Exception:
            self.dead_link_cache[url] = now
            return False

    # ------------------------------------------------------------------
    # 3. Smart anchor generation (action verbs + context)
    # ------------------------------------------------------------------
    def _smart_anchor(self, target_keyword: str, brand: Optional[str] = None,
                      model: Optional[str] = None, position: int = 0) -> str:
        """
        Generate a diverse, action‑oriented anchor text.
        """
        templates = [
            "how to {action} {kw}",
            "guide to {action} {kw}",
            "{kw} troubleshooting",
            "fixing {kw}",
            "diagnose {kw}",
            "resolve {kw}",
            "{kw} step‑by‑step",
            "download {kw} manual",
            "understand {kw}",
            "clear {kw}",
            "reset {kw}",
        ]
        actions = ["fix", "repair", "troubleshoot", "resolve", "diagnose", "clear", "reset"]

        # If brand and model are present, use a more specific template
        if brand and model:
            specific_templates = [
                f"how to {random.choice(actions)} {{brand}} {{model}}",
                f"{{brand}} {{model}} troubleshooting guide",
                f"fixing {{brand}} {{model}} error",
            ]
            template = random.choice(specific_templates)
            return template.format(brand=brand, model=model)
        else:
            # General case
            base = re.sub(r'\s+(error|code|fault|alarm|problem|issue)$', '', target_keyword, flags=re.I)
            template = templates[position % len(templates)]
            action = random.choice(actions) if "{action}" in template else ""
            return template.format(action=action, kw=base).strip()

    # ------------------------------------------------------------------
    # 4. Determine max links based on content length (throttling)
    # ------------------------------------------------------------------
    def _max_links_for_content(self, content: str) -> int:
        """Calculate allowed links based on word count."""
        word_count = len(content.split())
        density_based = int(word_count * LINK_DENSITY_FACTOR)
        return min(max(density_based, MIN_LINKS_PER_ARTICLE), MAX_LINKS_PER_ARTICLE)

    # ------------------------------------------------------------------
    # 5. In‑content link placement (context‑aware)
    # ------------------------------------------------------------------
    def inject_links_in_content(self, content: str, candidates: List[Dict]) -> str:
        """
        Insert links naturally into the article body, preferring paragraphs
        where relevant terms (brand, model, code) appear.
        """
        soup = BeautifulSoup(content, 'html.parser')
        body = soup.find('body')
        if not body:
            body = soup

        paragraphs = body.find_all('p')
        if not paragraphs:
            return self._append_links_section(content, candidates)

        max_links = self._max_links_for_content(content)
        links_to_insert = candidates[:max_links]

        # Shuffle to avoid bias
        random.shuffle(links_to_insert)

        inserted = 0
        for idx, cand in enumerate(links_to_insert):
            target_keyword = cand['keyword']
            # Extract entities of the target for contextual placement
            t_brand, t_model, t_code, _ = self.extract_entities(target_keyword)
            anchor = self._smart_anchor(target_keyword, t_brand, t_model, idx)
            url = cand.get('url')
            if not url:
                slug = quote_plus(target_keyword.lower().replace(' ', '-'))
                url = urljoin(SITE_DOMAIN, f"/troubleshoot/{slug}")

            # Find a paragraph that contains any relevant term from the target
            relevant_terms = [term for term in (t_brand, t_model, t_code) if term]
            target_para = None
            if relevant_terms:
                for para in paragraphs:
                    text = para.get_text()
                    if any(term and term.lower() in text.lower() for term in relevant_terms):
                        target_para = para
                        break
            if not target_para:
                # fallback to a random paragraph
                target_para = random.choice(paragraphs)

            # Build the link as a proper Tag
            a_tag = soup.new_tag('a', href=url)
            a_tag.string = anchor

            # Append to the end of the paragraph (natural)
            target_para.append(" ")
            target_para.append(a_tag)
            target_para.append(".")
            inserted += 1

        # If we inserted fewer than MIN_LINKS_PER_ARTICLE, add a fallback section
        if inserted < MIN_LINKS_PER_ARTICLE:
            fallback_html = self._build_links_section(candidates[:max_links])
            body.append(BeautifulSoup(fallback_html, 'html.parser'))

        return str(soup)

    def _build_links_section(self, candidates: List[Dict]) -> str:
        """Build a standalone 'Related' section HTML."""
        if not candidates:
            return ""
        html = '<div class="astra-internal-links"><h4>🔍 Related Troubleshooting Manuals</h4><ul>'
        for i, cand in enumerate(candidates):
            t_brand, t_model, _, _ = self.extract_entities(cand['keyword'])
            anchor = self._smart_anchor(cand['keyword'], t_brand, t_model, i)
            slug = quote_plus(cand['keyword'].lower().replace(' ', '-'))
            url = urljoin(SITE_DOMAIN, f"/troubleshoot/{slug}")
            html += f'<li><a href="{url}">{anchor}</a></li>'
        html += '</ul></div>'
        return html

    def _append_links_section(self, content: str, candidates: List[Dict]) -> str:
        """Fallback: add a 'Related' section at the end."""
        if not candidates:
            return content
        html = self._build_links_section(candidates)
        soup = BeautifulSoup(content, 'html.parser')
        body = soup.find('body')
        if body:
            body.append(BeautifulSoup(html, 'html.parser'))
            return str(soup)
        else:
            return content + "\n" + html

    # ------------------------------------------------------------------
    # 6. Reverse linking: update older articles to link to newer ones
    # ------------------------------------------------------------------
    def reverse_link_new_article(self, new_article: Dict):
        """
        Find a few older, popular articles (same brand) and inject a link to the new article.
        This pushes fresh content through established pages.
        """
        art_id = new_article['id']
        keyword = new_article['keyword']
        brand, model, code, lang = self.extract_entities(keyword)

        # Fetch older articles (created before this one) that are popular
        older_candidates = self.fetch_candidates(
            brand, model, code, lang, art_id,
            limit=REVERSE_LINKS_PER_ARTICLE,
            older_than=new_article.get('created_at')
        )

        if not older_candidates:
            logger.debug(f"No older candidates for reverse linking of {keyword}")
            return

        # For each older article, inject a link to the new article
        for old in older_candidates:
            old_id = old['id']
            old_content = old.get('content')
            if not old_content:
                continue

            # Build link to new article
            new_slug = quote_plus(keyword.lower().replace(' ', '-'))
            new_url = urljoin(SITE_DOMAIN, f"/troubleshoot/{new_slug}")
            anchor = self._smart_anchor(keyword, brand, model)

            # Insert link into old content (preferably in a relevant paragraph)
            soup = BeautifulSoup(old_content, 'html.parser')
            body = soup.find('body') or soup
            paragraphs = body.find_all('p')
            if paragraphs:
                # Try to find a paragraph that mentions the same brand/model
                target_para = None
                relevant_terms = [term for term in (brand, model, code) if term]
                if relevant_terms:
                    for para in paragraphs:
                        text = para.get_text()
                        if any(term and term.lower() in text.lower() for term in relevant_terms):
                            target_para = para
                            break
                if not target_para:
                    target_para = random.choice(paragraphs)

                a_tag = soup.new_tag('a', href=new_url)
                a_tag.string = anchor
                target_para.append(" ")
                target_para.append(a_tag)
                target_para.append(".")

                # Update Supabase
                try:
                    supabase.table("astra_data") \
                        .update({"content": str(soup), "updated_at": "now()"}) \
                        .eq("id", old_id) \
                        .execute()
                    logger.info(f"✅ Reverse-linked old article {old['keyword']} to new {keyword}")
                    self.reverse_processed += 1
                except Exception as e:
                    logger.error(f"Failed to reverse-link old article {old_id}: {e}")
            time.sleep(SLEEP_BETWEEN_UPDATES)

    # ------------------------------------------------------------------
    # 7. Main processing for a single article (forward linking)
    # ------------------------------------------------------------------
    def process_article(self, article: Dict, is_new: bool = False) -> bool:
        """
        Process a single article: find related articles and inject links.
        If is_new is True, also trigger reverse linking.
        """
        art_id = article['id']
        keyword = article['keyword']
        content = article.get('content')
        if not content:
            logger.debug(f"Article {art_id} has no content, skipping")
            return False

        brand, model, code, lang = self.extract_entities(keyword)
        candidates = self.fetch_candidates(brand, model, code, lang, art_id, MAX_LINKS_PER_ARTICLE)

        if candidates:
            new_content = self.inject_links_in_content(content, candidates)

            # Update Supabase
            try:
                supabase.table("astra_data") \
                    .update({
                        "content": new_content,
                        "updated_at": "now()",
                        "internal_links": len(candidates)
                    }) \
                    .eq("id", art_id) \
                    .execute()
                logger.info(f"✅ Forward-linked {keyword} with {len(candidates)} related articles")
            except Exception as e:
                logger.error(f"Failed to update {art_id}: {e}")
                return False
        else:
            logger.debug(f"No suitable candidates for {keyword}")

        # If this is a new article, perform reverse linking
        if is_new:
            self.reverse_link_new_article(article)

        return True

    # ------------------------------------------------------------------
    # 8. Main cycle: fetch articles and process
    # ------------------------------------------------------------------
    def run_cycle(self):
        """Main loop: fetch articles needing linking (both new and existing)."""
        logger.info("=== Astra Linker v4 Cycle Started ===")
        try:
            # Fetch articles that have content, ordered by creation date (newest first)
            response = supabase.table("astra_data") \
                .select("id, keyword, content, url, created_at") \
                .not_.is_("content", "null") \
                .order("created_at", desc=True) \
                .limit(BATCH_SIZE) \
                .execute()
            articles = response.data if response else []
        except Exception as e:
            logger.error(f"Failed to fetch articles: {e}")
            return

        logger.info(f"Found {len(articles)} articles to process")

        # The first article in the batch is the newest – treat as new for reverse linking
        for idx, art in enumerate(articles):
            is_new = (idx == 0)  # newest article gets reverse linking
            success = self.process_article(art, is_new=is_new)
            if success:
                self.processed_count += 1
            else:
                self.error_count += 1
            time.sleep(SLEEP_BETWEEN_UPDATES)

        logger.info(f"Cycle complete. Processed: {self.processed_count}, Errors: {self.error_count}, Reverse links added: {self.reverse_processed}")


if __name__ == "__main__":
    engine = LinkerEngine()
    engine.run_cycle()
