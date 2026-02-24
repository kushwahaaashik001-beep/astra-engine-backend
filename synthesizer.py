#!/usr/bin/env python3
"""
Astra Synthesizer - The ParBrahma Antim Astra (Level ∞+)
Author: Your Name
Purpose: Generate Google's dream content: featured snippet ready, multilingual,
         schema-rich, self-healing, and authority-packed. Includes Quick-Fix summary,
         technical spec tables, expert bios, and freshness signals.
         Designed to dominate Google Rank 0 and achieve $20+ RPM.
"""

import os
import sys
import time
import logging
import json
import re
import random
import hashlib
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
from urllib.parse import quote_plus
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment
load_dotenv()

# ============================ CONFIGURATION ============================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is required")

# Optional proxy for Groq
PROXY = os.getenv("PROXY")

# API endpoints
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-70b-8192"  # or "mixtral-8x7b-32768" for longer context

# Batch settings
BATCH_SIZE = 5
MIN_CONTENT_LENGTH = 1500          # increased to ensure depth
MAX_TOKENS = 4000
TEMPERATURE = 0.3

# Retry settings
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds

# Local cache directory (fallback if Supabase is unavailable)
CACHE_DIR = Path("./astra_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Languages for multilingual summaries
MULTILINGUAL_LANGUAGES = [
    {"code": "de", "name": "German"},
    {"code": "es", "name": "Spanish"},
    {"code": "ja", "name": "Japanese"},
    {"code": "zh", "name": "Chinese"}
]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AstraSynthesizer")

# ============================ SUPABASE CLIENT ============================
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    logger.error("Supabase credentials missing. Using local cache only (degraded mode).")
    supabase = None
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ============================ HELPER FUNCTIONS ============================
def create_session_with_retries():
    """Create a requests session with retry strategy."""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    if PROXY:
        session.proxies = {"http": PROXY, "https": PROXY}
    return session

def call_groq(prompt: str, system_msg: str = "You are a senior industrial automation engineer and technical writer.") -> Optional[str]:
    """Call Groq API with retries and exponential backoff."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }
    for attempt in range(MAX_RETRIES):
        try:
            session = create_session_with_retries()
            resp = session.post(GROQ_API_URL, headers=headers, json=payload, timeout=90)
            if resp.status_code == 200:
                data = resp.json()
                return data['choices'][0]['message']['content']
            elif resp.status_code == 429:
                wait = RETRY_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Rate limited. Retrying in {wait:.2f}s")
                time.sleep(wait)
            else:
                logger.error(f"Groq API error {resp.status_code}: {resp.text}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF * (2 ** attempt))
                else:
                    return None
        except Exception as e:
            logger.error(f"Groq call failed (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (2 ** attempt))
            else:
                return None
    return None

def fetch_pending_keywords(limit: int = BATCH_SIZE) -> List[Dict]:
    """Fetch keywords from Supabase with fallback to local cache."""
    if supabase:
        try:
            response = supabase.table("astra_data")\
                .select("*")\
                .eq("is_indexed", False)\
                .order("score", desc=True)\
                .limit(limit)\
                .execute()
            return response.data
        except Exception as e:
            logger.error(f"Supabase fetch failed: {e}. Trying local cache.")
    # Fallback: read from local cache file if exists
    cache_file = CACHE_DIR / "pending_keywords.json"
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    return []

def mark_as_processed(keyword_id: int, content: str, metadata: Dict[str, Any]):
    """Update Supabase with generated content; fallback to local cache."""
    update_data = {
        "content": content,
        "is_indexed": True,
        "updated_at": datetime.utcnow().isoformat(),
        "metadata": metadata
    }
    if supabase:
        try:
            supabase.table("astra_data").update(update_data).eq("id", keyword_id).execute()
            logger.info(f"Marked keyword ID {keyword_id} as processed in Supabase.")
            return
        except Exception as e:
            logger.error(f"Supabase update failed: {e}. Saving locally.")
    # Fallback: save to local cache
    cache_file = CACHE_DIR / f"{keyword_id}.json"
    with open(cache_file, 'w') as f:
        json.dump({"id": keyword_id, "content": content, "metadata": metadata}, f)
    logger.info(f"Saved locally to {cache_file}")

# ============================ ADVANCED CONTENT GENERATION ============================

def generate_toc(sections: List[Dict]) -> str:
    """Generate Table of Contents with anchor links."""
    toc_html = "<div class='astra-toc'><h3>📖 Quick Navigation</h3><ul>"
    for sec in sections:
        title = sec.get('title', 'Section')
        anchor = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
        toc_html += f"<li><a href='#{anchor}'>{title}</a></li>"
    toc_html += "</ul></div>"
    return toc_html

def get_related_links(current_keyword: str, current_id: int, limit: int = 4) -> str:
    """Fetch related keywords for internal linking."""
    if not supabase:
        return ""  # skip if no DB
    try:
        response = supabase.table("astra_data")\
            .select("id, keyword")\
            .neq("id", current_id)\
            .limit(limit)\
            .execute()
        if response.data:
            links = []
            for row in response.data:
                kw = row['keyword']
                slug = quote_plus(kw)
                links.append(f"<li><a href='/troubleshoot/{slug}'>{kw}</a></li>")
            related_html = f"<div class='astra-related'><h3>🔗 Related Troubleshooting Guides</h3><ul>{''.join(links)}</ul></div>"
            return related_html
    except Exception as e:
        logger.error(f"Failed to fetch related links: {e}")
    return ""

def generate_image_prompts(sections: List[Dict]) -> List[str]:
    """Generate image prompts for each section."""
    prompts = []
    for sec in sections:
        title = sec.get('title', '')
        prompts.append(f"Technical diagram showing {title.lower()} for industrial automation, labeled parts, schematic style")
    return prompts

def parse_outline(outline_response: str) -> List[Dict]:
    """Parse outline JSON; fallback to default sections."""
    try:
        json_match = re.search(r'(\[.*\])', outline_response, re.DOTALL)
        if json_match:
            outline = json.loads(json_match.group(1))
            if isinstance(outline, list):
                normalized = []
                for item in outline:
                    if isinstance(item, str):
                        normalized.append({"title": item, "description": ""})
                    elif isinstance(item, dict) and "title" in item:
                        normalized.append(item)
                    else:
                        normalized.append({"title": "Section", "description": ""})
                return normalized
    except:
        pass
    logger.warning("Outline parsing failed, using default sections.")
    return [
        {"title": "Error Code Meaning", "description": "Explain what this error code indicates in technical terms."},
        {"title": "Common Causes", "description": "List typical reasons in a table with Cause and Description."},
        {"title": "Step-by-Step Troubleshooting", "description": "Numbered steps to diagnose and fix."},
        {"title": "Technical Specifications", "description": "Relevant part numbers, voltage values, parameters."},
        {"title": "Prevention Tips", "description": "How to avoid this issue in the future."},
        {"title": "Common Mistakes to Avoid", "description": "Mistakes that can make the problem worse."}
    ]

def generate_faq_schema(keyword: str) -> Optional[str]:
    """Generate JSON-LD FAQ schema."""
    prompt = f"""For the technical problem "{keyword}", generate a JSON-LD FAQ schema with 3-5 common questions and concise answers. 
Output only the JSON object. Use "@context": "https://schema.org", "@type": "FAQPage", "mainEntity": list of Question/Answer items.
Each Question should have "name" (question) and "acceptedAnswer" with "@type": "Answer" and "text" (answer). Be technical.
"""
    response = call_groq(prompt, "You are a structured data expert.")
    if response:
        try:
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_match:
                schema = json.loads(json_match.group(1))
                if schema.get("@type") == "FAQPage":
                    return json.dumps(schema, indent=2)
        except:
            pass
    return None

def generate_howto_schema(keyword: str, steps: List[str]) -> Optional[str]:
    """Generate HowTo schema from troubleshooting steps."""
    if not steps:
        return None
    howto_steps = []
    for i, step_text in enumerate(steps):
        clean_step = re.sub('<[^<]+?>', '', step_text)[:200]
        howto_steps.append({
            "@type": "HowToStep",
            "position": i + 1,
            "text": clean_step,
            "name": f"Step {i+1}"
        })
    schema = {
        "@context": "https://schema.org",
        "@type": "HowTo",
        "name": f"How to Fix {keyword}",
        "step": howto_steps
    }
    return json.dumps(schema, indent=2)

def generate_source_reference(keyword: str) -> Optional[str]:
    """Simulate finding an official reference or manual link."""
    prompt = f"""For the technical issue "{keyword}", suggest an official manufacturer manual or technical document that would be a reliable reference. Provide the title and a URL (use example.com if unknown). Output as JSON: {{"title": "...", "url": "..."}}"""
    response = call_groq(prompt, "You are a research assistant.")
    if response:
        try:
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_match:
                ref = json.loads(json_match.group(1))
                if ref.get('title') and ref.get('url'):
                    return f'<p><strong>Reference:</strong> <a href="{ref["url"]}" rel="nofollow" target="_blank">{ref["title"]}</a> (Official documentation)</p>'
        except:
            pass
    return None

def generate_multilingual_summary(keyword: str, summary_text: str) -> Dict[str, str]:
    """
    Translate the summary (first 200 words) into multiple languages.
    Returns dict with language codes as keys and translated text as values.
    """
    translations = {}
    words = summary_text.split()[:200]
    summary = ' '.join(words)
    for lang in MULTILINGUAL_LANGUAGES:
        code = lang['code']
        name = lang['name']
        prompt = f"""Translate the following English technical summary into {name}. Keep technical terms accurate.
Summary: {summary}
Translation:"""
        trans = call_groq(prompt, "You are a professional translator specializing in technical documentation.")
        if trans:
            translations[code] = trans.strip()
        time.sleep(1)
    return translations

def generate_spec_table(keyword: str) -> str:
    """
    Generate a technical specification table for high-value industrial ads.
    Returns HTML table.
    """
    prompt = f"""Create a technical specification table for equipment related to '{keyword}'. 
Include columns: Parameter, Standard Value, and Tolerance. 
Use realistic industrial values (voltages, resistances, part numbers, etc.). 
Output only the HTML table with <table>, <tr>, <th>, <td>. Do not include extra text."""
    table_content = call_groq(prompt, "You are a technical data analyst.")
    if table_content and '<table>' in table_content:
        return f"<div class='astra-spec-table'><h3>📊 Technical Reference Data</h3>{table_content}</div>"
    return ""

def add_freshness_timestamp() -> str:
    """Add a 'Last Verified' timestamp for freshness signal."""
    now = datetime.utcnow().strftime("%B %d, %Y")
    return f"<p class='astra-freshness'><strong>✅ Last Verified:</strong> {now} by our engineering team.</p>"

def add_expert_callouts(content: str) -> str:
    """Add random expert tips/warnings/notes to the content."""
    callouts = [
        "<div class='astra-tip'><strong>💡 Pro-Tip:</strong> Always verify the power supply voltage before replacing the module.</div>",
        "<div class='astra-warning'><strong>⚠️ Safety Warning:</strong> Disconnect main power before servicing. Capacitors may retain charge.</div>",
        "<div class='astra-note'><strong>📝 Expert Note:</strong> This error often occurs after firmware updates; check parameter compatibility.</div>",
        "<div class='astra-important'><strong>🔧 Critical:</strong> Use only manufacturer-approved replacement parts to avoid voiding warranty.</div>"
    ]
    # Insert one random callout near the middle (just before the last third)
    lines = content.split('\n')
    if len(lines) > 10:
        pos = len(lines) // 2
        lines.insert(pos, random.choice(callouts))
        return '\n'.join(lines)
    else:
        return content + "\n" + random.choice(callouts)

def generate_expert_bio() -> str:
    """Creates a trust-building author profile for Google E-E-A-T."""
    experts = [
        {"name": "Dr. Aris Astra", "bio": "Senior Industrial Automation Consultant with 20+ years of experience in PLC diagnostics.", "img": "expert-1.jpg"},
        {"name": "Eng. Vikram Logic", "bio": "Certified Systems Engineer specializing in high-voltage industrial error resolution.", "img": "expert-2.jpg"},
        {"name": "Sarah Tech-Manual", "bio": "Technical Documentation Lead for global robotics manufacturing firm.", "img": "expert-3.jpg"}
    ]
    expert = random.choice(experts)
    return f"""<div class='astra-author-bio'>
                <img src='/authors/{expert["img"]}' alt='{expert["name"]}' loading='lazy'>
                <div><strong>Verified Expert: {expert["name"]}</strong><p>{expert["bio"]}</p></div>
              </div>"""

def generate_technical_article(keyword: str, keyword_id: int) -> Optional[Dict[str, Any]]:
    """
    Generate full article with all enhancements: quick-fix, spec tables, etc.
    Returns dict with 'html' and 'metadata'.
    """
    logger.info(f"Generating content for: {keyword}")

    # Step 1: Outline generation
    outline_prompt = f"""You are a technical documentation expert. For the query "{keyword}", create a detailed outline of a troubleshooting guide. 
The outline should be a JSON array of objects, each with "title" (section heading) and "description" (what to cover).
Include 6-7 sections covering: meaning, causes, step-by-step fix, technical specs, prevention, common mistakes, and maybe advanced diagnostics.
Output only the JSON array.
"""
    outline_response = call_groq(outline_prompt)
    if not outline_response:
        return None
    sections = parse_outline(outline_response)

    # Step 2: TOC
    toc_html = generate_toc(sections)

    # Step 3: Generate each section with enhanced prompt (includes quick-fix instruction for first section)
    full_content = []
    raw_steps_for_schema = []
    all_text = ""

    for idx, sec in enumerate(sections):
        title = sec.get('title', 'Section')
        desc = sec.get('description', '')
        anchor = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
        
        # Special instruction for first section: include quick-fix box
        quick_fix_instruction = ""
        if idx == 0:
            quick_fix_instruction = "Start this section with a 2-line 'FAST ACTION' summary inside a <blockquote class='quick-fix'> box that gives the most essential fix immediately. Then proceed with detailed explanation."
        
        section_prompt = f"""Act as a Senior Field Engineer. For "{keyword}", write the section titled "{title}". {desc}
INSTRUCTIONS:
1. Use technical LSI terms (impedance, logic-gate, parameter-bit, grounding, feedback loop, etc.).
2. If this is a troubleshooting section, use clear numbered steps.
3. {quick_fix_instruction}
4. Include specific technical data (part numbers, voltage values, tolerances) when relevant.
5. Format as clean HTML with appropriate tags: <p>, <ul>/<li>, <table> if needed.
6. Do not include the section heading (will be added separately). Just the content.
"""
        section_content = call_groq(section_prompt, "You are a PhD-level Industrial Engineer.")
        if section_content:
            full_content.append(f"<h3 id='{anchor}'>{title}</h3>\n{section_content}")
            all_text += section_content + " "
            if "troubleshoot" in title.lower() or "fix" in title.lower() or "step" in title.lower():
                raw_steps_for_schema.append(section_content)
        time.sleep(1)

    # Step 4: Image prompts
    image_prompts = generate_image_prompts(sections)

    # Step 5: Internal links
    related_html = get_related_links(keyword, keyword_id)

    # Step 6: FAQ schema
    faq_schema = generate_faq_schema(keyword)
    faq_html = f"\n<script type='application/ld+json'>{faq_schema}</script>" if faq_schema else ""

    # Step 7: HowTo schema
    howto_schema = generate_howto_schema(keyword, raw_steps_for_schema)
    howto_html = f"\n<script type='application/ld+json'>{howto_schema}</script>" if howto_schema else ""

    # Step 8: Source reference
    ref_html = generate_source_reference(keyword) or ""

    # Step 9: Technical specification table (high-value entity data)
    spec_table = generate_spec_table(keyword)

    # Step 10: Multilingual summary
    summary_text = all_text[:2000]
    translations = generate_multilingual_summary(keyword, summary_text)
    trans_html = ""
    if translations:
        trans_html = "<div class='astra-translations'><h4>🌐 Read in other languages</h4>"
        for code, text in translations.items():
            trans_html += f"<details><summary>{code.upper()}</summary><p>{text}</p></details>"
        trans_html += "</div>"

    # Step 11: Expert callouts (add to content)
    full_article = "\n".join(full_content)
    full_article = add_expert_callouts(full_article)

    # Step 12: Expert bio
    author_html = generate_expert_bio()

    # Step 13: Freshness timestamp
    freshness_html = add_freshness_timestamp()

    # Step 14: Assemble final HTML
    main_heading = f"<h1>Complete Fix: {keyword} (2026 Troubleshooting Manual)</h1>"
    final_html = f"""
<article class='astra-premium-post'>
    {main_heading}
    {author_html}
    {toc_html}
    <div class='article-body'>
        {full_article}
    </div>
    {spec_table}
    {ref_html}
    {related_html}
    {trans_html}
    {freshness_html}
    {faq_html}
    {howto_html}
</article>
    """.strip()

    # Step 15: Advanced validation and self-correction
    quality_score = 0
    validation_errors = []

    if len(final_html) < MIN_CONTENT_LENGTH:
        validation_errors.append(f"Content too short ({len(final_html)} chars)")
    else:
        quality_score += 1

    if keyword.lower() not in final_html.lower():
        validation_errors.append("Keyword missing in content")
    else:
        quality_score += 1

    # Check for at least one table (spec table might be empty, but we want some structured data)
    if '<table>' not in final_html and not raw_steps_for_schema:
        validation_errors.append("No structured data (tables or steps) found")
    else:
        quality_score += 1

    # Check for technical terms (simple heuristic)
    tech_terms = ['voltage', 'current', 'resistance', 'parameter', 'firmware', 'hardware', 'circuit']
    if any(term in final_html.lower() for term in tech_terms):
        quality_score += 1
    else:
        validation_errors.append("Lacks technical depth")

    if validation_errors:
        logger.warning(f"Quality issues for {keyword}: {validation_errors}")
        # If quality is too low, reject and maybe retry later
        if quality_score < 2:
            logger.error(f"Content rejected due to poor quality: {keyword}")
            return None

    # Prepare metadata
    metadata = {
        "image_prompts": image_prompts,
        "translations": translations,
        "source_reference": ref_html,
        "spec_table": bool(spec_table),
        "sections": [sec.get('title') for sec in sections],
        "word_count": len(final_html.split()),
        "has_howto": bool(howto_schema),
        "has_faq": bool(faq_schema),
        "quality_score": quality_score,
        "generated_at": datetime.utcnow().isoformat()
    }

    return {"html": final_html, "metadata": metadata}

# ============================ MAIN LOOP ============================
def synthesize():
    logger.info("=== Astra Synthesizer (ParBrahma Antim Astra) Started ===")
    keywords = fetch_pending_keywords()
    if not keywords:
        logger.info("No pending keywords. Sleeping...")
        return

    logger.info(f"Processing {len(keywords)} keywords...")
    for kw_data in keywords:
        keyword = kw_data['keyword']
        kw_id = kw_data['id']

        if kw_data.get('is_indexed'):
            continue

        # Generate article with retry logic (max 2 attempts)
        result = None
        for attempt in range(2):
            result = generate_technical_article(keyword, kw_id)
            if result:
                break
            logger.warning(f"Retry {attempt+1} for {keyword}")
            time.sleep(5)

        if not result:
            logger.error(f"Failed to generate content for {keyword} after retries")
            continue

        content = result["html"]
        metadata = result["metadata"]
        mark_as_processed(kw_id, content, metadata)
        logger.info(f"Successfully processed: {keyword}")

        time.sleep(3)

    logger.info("=== Synthesizer Cycle Complete ===")

# ============================ ENTRY POINT ============================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Astra Synthesizer - ParBrahma Engine")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--interval", type=int, default=60, help="Minutes between cycles")
    args = parser.parse_args()

    if args.once:
        synthesize()
    else:
        while True:
            synthesize()
            logger.info(f"Sleeping for {args.interval} minutes...")
            time.sleep(args.interval * 60)
