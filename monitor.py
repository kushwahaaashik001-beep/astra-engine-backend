#!/usr/bin/env python3
"""
Astra Monitor – The Infinity Guardian (Level ∞+)
Monitors all Astra components: synthesizer, linker, pinger, and overall engine health.
Detects anomalies, runs advanced SEO audits, triggers auto‑recovery, and sends alerts.
Integrates with Telegram/Email and stores reports in Supabase.
"""

import os
import sys
import logging
import smtplib
import subprocess
import json
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import requests
from supabase import create_client, Client
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

# Load environment
load_dotenv()

# ============================ CONFIGURATION ============================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SITE_DOMAIN = os.getenv("SITE_DOMAIN", "https://your-astra-site.com")

# Alert thresholds
PENDING_THRESHOLD = 10000               # If pending keywords exceed this, warn
ZERO_OUTPUT_THRESHOLD_HOURS = 24        # If no new articles in last X hours, warn
MIN_ARTICLES_PER_DAY = 5                 # Expected minimum daily production
THIN_CONTENT_THRESHOLD = 1500            # Minimum characters for a good article
MAX_DUPLICATE_SLUGS = 3                   # Max allowed duplicate slugs
MAX_DEAD_LINKS = 5                        # Warn if more than X dead links

# Telegram bot (optional)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Email alerts (optional)
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO = os.getenv("EMAIL_TO")

# Auto‑recovery
ENABLE_AUTO_RECOVERY = os.getenv("ENABLE_AUTO_RECOVERY", "true").lower() == "true"
SCRIPTS_PATH = os.getenv("SCRIPTS_PATH", "./")  # directory containing synthesizer.py, linker.py, pinger.py

# Paths
PUBLIC_DIR = os.getenv("PUBLIC_DIR", "./public")
SITEMAP_PATH = os.path.join(PUBLIC_DIR, "sitemap.xml")
RSS_PATH = os.path.join(PUBLIC_DIR, "rss.xml")

# Store reports in Supabase
STORE_REPORTS = True
REPORT_TABLE = "engine_logs"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Monitor - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AstraMonitor")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    logger.error("Supabase credentials missing. Exiting.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ============================ ALERT FUNCTIONS ============================
def send_telegram_message(message: str):
    """Send alert via Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")

def send_email_alert(subject: str, body: str):
    """Send alert via email."""
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD, EMAIL_FROM, EMAIL_TO]):
        return
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = EMAIL_TO
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        logger.error(f"Email send failed: {e}")

def send_alert(message: str, subject: str = "Astra Monitor Alert"):
    """Send alert via all configured channels."""
    logger.warning(f"ALERT: {message}")
    send_telegram_message(f"🚨 {subject}\n{message}")
    send_email_alert(subject, message)

# ============================ HELPER FUNCTIONS ============================
def slugify(keyword: str) -> str:
    """Generate slug from keyword (matching linker's method)."""
    # Remove special chars, keep alphanumeric and spaces
    cleaned = re.sub(r'[^a-zA-Z0-9\s-]', '', keyword).strip().lower()
    # Replace spaces/hyphens with single hyphen
    return re.sub(r'[\s-]+', '-', cleaned)

# ============================ CORE METRICS ============================
def get_total_keywords() -> int:
    """Total keywords in astra_data."""
    try:
        res = supabase.table("astra_data").select("id", count="exact").execute()
        return res.count
    except Exception as e:
        logger.error(f"Failed to get total keywords: {e}")
        return -1

def get_pending_count() -> int:
    """Keywords not yet indexed."""
    try:
        res = supabase.table("astra_data").select("id", count="exact").eq("is_indexed", False).execute()
        return res.count
    except Exception as e:
        logger.error(f"Failed to get pending count: {e}")
        return -1

def get_recent_articles(hours: int = 24) -> int:
    """Count articles indexed in last X hours."""
    try:
        since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        res = supabase.table("astra_data") \
            .select("id", count="exact") \
            .eq("is_indexed", True) \
            .gte("updated_at", since) \
            .execute()
        return res.count
    except Exception as e:
        logger.error(f"Failed to get recent articles: {e}")
        return -1

def get_avg_score() -> float:
    """Average score of pending keywords (to gauge quality)."""
    try:
        res = supabase.table("astra_data") \
            .select("score") \
            .eq("is_indexed", False) \
            .limit(1000) \
            .execute()
        scores = [row['score'] for row in res.data if row.get('score')]
        return sum(scores) / len(scores) if scores else 0.0
    except Exception as e:
        logger.error(f"Failed to get avg score: {e}")
        return 0.0

def get_stuck_keywords(days: int = 7) -> int:
    """Keywords that have been pending for too long (older than X days)."""
    try:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        res = supabase.table("astra_data") \
            .select("id", count="exact") \
            .eq("is_indexed", False) \
            .lt("created_at", cutoff) \
            .execute()
        return res.count
    except Exception as e:
        logger.error(f"Failed to get stuck keywords: {e}")
        return -1

# ============================ ADVANCED AUDITS ============================

def audit_internal_links() -> List[Dict]:
    """
    Find articles that have no internal links despite having suitable candidates.
    """
    logger.info("Auditing internal links...")
    zero_link_articles = []
    try:
        # Fetch articles with content but internal_links is 0 or null
        res = supabase.table("astra_data") \
            .select("id, keyword, content, entity_brand") \
            .not_.is_("content", "null") \
            .or_("internal_links.is.null,internal_links.eq.0") \
            .limit(100) \
            .execute()
        for art in res.data:
            keyword = art['keyword']
            brand = art.get('entity_brand')
            # Check if any candidate exists (same brand) ignoring dead links
            if brand:
                # Simplified: count articles with same brand (excluding self)
                count_res = supabase.table("astra_data") \
                    .select("id", count="exact") \
                    .eq("entity_brand", brand) \
                    .neq("id", art['id']) \
                    .execute()
                if count_res.count > 0:
                    zero_link_articles.append({
                        "id": art['id'],
                        "keyword": keyword,
                        "brand": brand,
                        "candidate_count": count_res.count
                    })
    except Exception as e:
        logger.error(f"Internal link audit failed: {e}")
    logger.info(f"Found {len(zero_link_articles)} articles with zero internal links despite candidates.")
    return zero_link_articles

def audit_dead_links() -> List[str]:
    """
    Retrieve dead links from a dead_links table (if maintained by linker/pinger).
    """
    dead_urls = []
    try:
        res = supabase.table("dead_links").select("url").execute()
        dead_urls = [row['url'] for row in res.data]
    except Exception:
        # If table doesn't exist, return empty
        pass
    return dead_urls

def audit_thin_content() -> List[Dict]:
    """
    Find articles with very short content (below threshold).
    """
    thin = []
    try:
        res = supabase.table("astra_data") \
            .select("id, keyword, content") \
            .not_.is_("content", "null") \
            .limit(500) \
            .execute()
        for art in res.data:
            content = art['content']
            if content and len(content) < THIN_CONTENT_THRESHOLD:
                thin.append({
                    "id": art['id'],
                    "keyword": art['keyword'],
                    "length": len(content)
                })
    except Exception as e:
        logger.error(f"Thin content audit failed: {e}")
    return thin

def audit_duplicate_slugs() -> List[Tuple[str, int]]:
    """
    Detect duplicate slugs (two different keywords producing same slug).
    """
    try:
        res = supabase.table("astra_data").select("keyword").execute()
        slug_counts = {}
        for row in res.data:
            slug = slugify(row['keyword'])
            slug_counts[slug] = slug_counts.get(slug, 0) + 1
        duplicates = [(slug, count) for slug, count in slug_counts.items() if count > 1]
        return duplicates
    except Exception as e:
        logger.error(f"Duplicate slug audit failed: {e}")
        return []

def audit_html_errors() -> List[str]:
    """
    Check if any stored HTML is malformed (parsing errors).
    """
    errors = []
    try:
        res = supabase.table("astra_data") \
            .select("id, content") \
            .not_.is_("content", "null") \
            .limit(200) \
            .execute()
        for art in res.data:
            try:
                soup = BeautifulSoup(art['content'], 'html.parser')
                # Trigger parsing to detect errors
                soup.prettify()
            except Exception:
                errors.append(f"Malformed HTML in article {art['id']}")
    except Exception as e:
        logger.error(f"HTML audit failed: {e}")
    return errors

def audit_sitemap_vs_db() -> Dict[str, Any]:
    """
    Compare URLs in sitemap.xml with those in database that are indexed.
    """
    sitemap_urls = set()
    db_indexed_urls = set()
    try:
        if os.path.exists(SITEMAP_PATH):
            tree = ET.parse(SITEMAP_PATH)
            root = tree.getroot()
            namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            for url in root.findall('ns:url/ns:loc', namespaces=namespace):
                sitemap_urls.add(url.text.strip())
    except Exception as e:
        logger.error(f"Failed to parse sitemap: {e}")

    try:
        res = supabase.table("astra_data") \
            .select("keyword") \
            .eq("is_indexed", True) \
            .execute()
        for row in res.data:
            slug = slugify(row['keyword'])
            url = f"{SITE_DOMAIN}/troubleshoot/{slug}"
            db_indexed_urls.add(url)
    except Exception as e:
        logger.error(f"Failed to fetch indexed keywords: {e}")

    missing_in_sitemap = db_indexed_urls - sitemap_urls
    extra_in_sitemap = sitemap_urls - db_indexed_urls
    return {
        "missing_in_sitemap": list(missing_in_sitemap)[:20],
        "extra_in_sitemap": list(extra_in_sitemap)[:20],
        "total_db_indexed": len(db_indexed_urls),
        "total_sitemap": len(sitemap_urls)
    }

def audit_ping_history() -> List[Dict]:
    """
    Check recent ping_log entries for failures.
    """
    failures = []
    try:
        cutoff = (datetime.utcnow() - timedelta(days=1)).isoformat()
        res = supabase.table("ping_log") \
            .select("*") \
            .gte("timestamp", cutoff) \
            .execute()
        for log in res.data:
            results = json.loads(log['results']) if isinstance(log['results'], str) else log['results']
            for key, success in results.items():
                if not success:
                    failures.append({"timestamp": log['timestamp'], "target": key})
    except Exception as e:
        logger.error(f"Ping history audit failed: {e}")
    return failures

# ============================ AUTO‑RECOVERY ============================
def trigger_script(script_name: str):
    """Run a Python script as subprocess (non‑blocking)."""
    if not ENABLE_AUTO_RECOVERY:
        return
    script_path = os.path.join(SCRIPTS_PATH, script_name)
    if os.path.exists(script_path):
        try:
            subprocess.Popen([sys.executable, script_path, "--once"])
            logger.info(f"Triggered {script_name}")
        except Exception as e:
            logger.error(f"Failed to trigger {script_name}: {e}")
    else:
        logger.warning(f"Script {script_name} not found at {script_path}")

# ============================ HEALTH SCORE ============================
def compute_health_score(metrics: Dict[str, Any]) -> int:
    """
    Calculate a 0-100 health score based on various factors.
    """
    score = 100
    # Deductions
    if metrics['pending'] > PENDING_THRESHOLD:
        score -= 20
    if metrics['recent_24h'] < MIN_ARTICLES_PER_DAY:
        score -= 15
    if metrics['stuck_7d'] > 100:
        score -= 10
    if metrics['zero_link_count'] > 10:
        score -= 10
    if metrics['thin_content_count'] > 10:
        score -= 10
    if len(metrics['duplicate_slugs']) > MAX_DUPLICATE_SLUGS:
        score -= 10
    if metrics['dead_links_count'] > MAX_DEAD_LINKS:
        score -= 10
    if metrics['sitemap_missing'] > 10:
        score -= 10
    if metrics['ping_failures'] > 0:
        score -= 5 * min(metrics['ping_failures'], 5)
    return max(score, 0)

# ============================ REPORT GENERATION ============================
def generate_report() -> Dict[str, Any]:
    """Collect all metrics and return comprehensive report."""
    total = get_total_keywords()
    pending = get_pending_count()
    recent = get_recent_articles(ZERO_OUTPUT_THRESHOLD_HOURS)
    avg_score = get_avg_score()
    stuck = get_stuck_keywords(7)

    zero_links = audit_internal_links()
    thin = audit_thin_content()
    duplicates = audit_duplicate_slugs()
    dead_urls = audit_dead_links()
    html_errors = audit_html_errors()
    sitemap_diff = audit_sitemap_vs_db()
    ping_fails = audit_ping_history()

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_keywords": total,
        "pending": pending,
        "recent_24h": recent,
        "avg_score_pending": avg_score,
        "stuck_7d": stuck,
        "zero_link_count": len(zero_links),
        "zero_link_examples": zero_links[:5],
        "thin_content_count": len(thin),
        "thin_content_examples": thin[:5],
        "duplicate_slugs": duplicates[:10],
        "dead_links_count": len(dead_urls),
        "dead_link_examples": dead_urls[:10],
        "html_errors_count": len(html_errors),
        "html_errors": html_errors[:5],
        "sitemap_missing": len(sitemap_diff['missing_in_sitemap']),
        "sitemap_extra": len(sitemap_diff['extra_in_sitemap']),
        "sitemap_missing_examples": sitemap_diff['missing_in_sitemap'][:5],
        "ping_failures": len(ping_fails),
        "ping_fail_examples": ping_fails[:5],
        "alerts": []
    }

    # Build alerts
    if pending > PENDING_THRESHOLD:
        report["alerts"].append(f"Pending queue > {PENDING_THRESHOLD} ({pending})")
    if recent == 0 and pending > 0:
        report["alerts"].append(f"No new articles in last {ZERO_OUTPUT_THRESHOLD_HOURS}h but pending queue exists")
    if recent < MIN_ARTICLES_PER_DAY and pending > 100:
        report["alerts"].append(f"Low production: only {recent} articles in 24h")
    if stuck > 100:
        report["alerts"].append(f"{stuck} keywords stuck pending for >7 days")
    if len(zero_links) > 10:
        report["alerts"].append(f"{len(zero_links)} articles have zero internal links (linker issue)")
    if len(thin) > 10:
        report["alerts"].append(f"{len(thin)} articles are thin content (<{THIN_CONTENT_THRESHOLD} chars)")
    if len(duplicates) > MAX_DUPLICATE_SLUGS:
        report["alerts"].append(f"{len(duplicates)} duplicate slugs detected")
    if len(dead_urls) > MAX_DEAD_LINKS:
        report["alerts"].append(f"{len(dead_urls)} dead links found")
    if len(html_errors) > 0:
        report["alerts"].append(f"{len(html_errors)} articles have malformed HTML")
    if len(sitemap_diff['missing_in_sitemap']) > 10:
        report["alerts"].append(f"{len(sitemap_diff['missing_in_sitemap'])} indexed URLs missing from sitemap")
    if ping_fails:
        report["alerts"].append(f"{len(ping_fails)} recent ping failures")

    # Compute health score
    report["health_score"] = compute_health_score(report)

    return report

def print_report(report: Dict[str, Any]):
    """Pretty print report to logs."""
    logger.info("📊 --- ASTRA ENGINE HEALTH REPORT ---")
    logger.info(f"Timestamp       : {report['timestamp']}")
    logger.info(f"Health Score    : {report['health_score']}/100")
    logger.info(f"Total Keywords  : {report['total_keywords']}")
    logger.info(f"Pending Queue   : {report['pending']}")
    logger.info(f"Articles (24h)  : {report['recent_24h']}")
    logger.info(f"Avg Score (pend) : {report['avg_score_pending']:.2f}")
    logger.info(f"Stuck (>7d)     : {report['stuck_7d']}")
    logger.info(f"Zero‑link Arts   : {report['zero_link_count']}")
    logger.info(f"Thin Content     : {report['thin_content_count']}")
    logger.info(f"Duplicate Slugs  : {len(report['duplicate_slugs'])}")
    logger.info(f"Dead Links       : {report['dead_links_count']}")
    logger.info(f"HTML Errors      : {report['html_errors_count']}")
    logger.info(f"Sitemap Missing  : {report['sitemap_missing']}")
    logger.info(f"Ping Failures    : {report['ping_failures']}")
    if report['alerts']:
        logger.warning("🚨 ALERTS:")
        for alert in report['alerts']:
            logger.warning(f"   - {alert}")
    else:
        logger.info("✅ No alerts. System operational.")
    logger.info("----------------------------------------")

def store_report(report: Dict[str, Any]):
    """Save report to Supabase engine_logs table."""
    if not STORE_REPORTS:
        return
    try:
        supabase.table(REPORT_TABLE).insert(report).execute()
        logger.info(f"Report saved to {REPORT_TABLE}.")
    except Exception as e:
        logger.error(f"Failed to store report: {e}")

# ============================ AUTO‑RECOVERY DECISION ============================
def decide_auto_recovery(report: Dict[str, Any]):
    """
    Based on report, trigger appropriate scripts.
    """
    if not ENABLE_AUTO_RECOVERY:
        return
    # If pending > 0 and recent == 0, synthesizer might be stuck
    if report['pending'] > 0 and report['recent_24h'] == 0:
        logger.info("Auto‑recovery: Triggering synthesizer")
        trigger_script("synthesizer.py")
    # If many zero‑link articles, trigger linker
    if report['zero_link_count'] > 20:
        logger.info("Auto‑recovery: Triggering linker")
        trigger_script("linker.py")
    # If ping failures, trigger pinger
    if report['ping_failures'] > 3:
        logger.info("Auto‑recovery: Triggering pinger")
        trigger_script("pinger.py")

# ============================ MAIN ============================
def monitor():
    """Main monitoring function."""
    logger.info("📡 Astra Monitor Infinity starting...")
    report = generate_report()
    print_report(report)

    if report['alerts']:
        alert_msg = "\n".join(report['alerts'])
        send_alert(alert_msg, subject="Astra Engine Issues Detected")

    store_report(report)
    decide_auto_recovery(report)

if __name__ == "__main__":
    monitor()
