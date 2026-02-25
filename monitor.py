#!/usr/bin/env python3
"""
Astra Monitor - The Engine Guardian (Level ∞)
Author: Your Name
Purpose: Continuously monitor Astra engine health, detect anomalies, send alerts,
         and maintain a history of system status. Integrates with Telegram/Email.
"""

import os
import sys
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import requests
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment
load_dotenv()

# ============================ CONFIGURATION ============================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Alert thresholds
PENDING_THRESHOLD = 10000        # If pending keywords exceed this, warn
ZERO_OUTPUT_THRESHOLD_HOURS = 24 # If no new articles in last X hours, warn
MIN_ARTICLES_PER_DAY = 5         # Expected minimum daily production

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

# Whether to store reports in a separate table (create this table in Supabase)
STORE_REPORTS = True
REPORT_TABLE = "engine_logs"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Monitor - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AstraMonitor")

# ============================ SUPABASE CLIENT ============================
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

# ============================ HEALTH METRICS ============================
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
            .execute()  # sample
        scores = [row['score'] for row in res.data if row.get('score')]
        if scores:
            return sum(scores) / len(scores)
        return 0.0
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

# ============================ REPORT GENERATION ============================
def generate_report() -> Dict[str, Any]:
    """Collect all metrics and return as dict."""
    total = get_total_keywords()
    pending = get_pending_count()
    recent = get_recent_articles(ZERO_OUTPUT_THRESHOLD_HOURS)
    avg_score = get_avg_score()
    stuck = get_stuck_keywords(7)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_keywords": total,
        "pending": pending,
        "recent_24h": recent,
        "avg_score_pending": avg_score,
        "stuck_7d": stuck,
        "alerts": []
    }

    # Detect anomalies
    if pending > PENDING_THRESHOLD:
        report["alerts"].append(f"Pending queue > {PENDING_THRESHOLD} ({pending})")
    if recent == 0 and pending > 0:
        report["alerts"].append(f"No new articles in last {ZERO_OUTPUT_THRESHOLD_HOURS}h but pending queue exists")
    if recent < MIN_ARTICLES_PER_DAY and pending > 100:
        report["alerts"].append(f"Low production: only {recent} articles in 24h")
    if stuck > 100:
        report["alerts"].append(f"{stuck} keywords stuck pending for >7 days")

    return report

def print_report(report: Dict[str, Any]):
    """Pretty print report to logs."""
    logger.info("📊 --- ASTRA ENGINE HEALTH REPORT ---")
    logger.info(f"Timestamp       : {report['timestamp']}")
    logger.info(f"Total Keywords  : {report['total_keywords']}")
    logger.info(f"Pending Queue   : {report['pending']}")
    logger.info(f"Articles (24h)  : {report['recent_24h']}")
    logger.info(f"Avg Score (pend) : {report['avg_score_pending']:.2f}")
    logger.info(f"Stuck (>7d)     : {report['stuck_7d']}")
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

def monitor():
    logger.info("📡 Astra Monitor starting...")
    report = generate_report()
    print_report(report)

    # Send alerts if any
    if report['alerts']:
        alert_msg = "\n".join(report['alerts'])
        send_alert(alert_msg, subject="Astra Engine Issues Detected")

    # Store report
    store_report(report)

if __name__ == "__main__":
    monitor()
