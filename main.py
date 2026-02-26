import asyncio
import logging
import sys
import os
from dotenv import load_dotenv

# Files ko import kar rahe hain
try:
    from scout import main as run_scout
    from multiplier import main as run_multiplier
    from synthesizer import main as run_synthesizer
except ImportError as e:
    # Abhi files setup ho rahi hain, isliye error ko handle kar rahe hain
    pass

# Load Environment Variables
load_dotenv()

# Logging Setup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - AstraOS - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AstraMain")

# ==========================================================
# 🛑 SAFETY SWITCH: Isko True karne par hi machine chalegi
# ==========================================================
MACHINE_ACTIVE = False 
# ==========================================================

async def start_engine():
    logger.info("🌌 Astra Infinity Engine - Standby Mode")

    if not MACHINE_ACTIVE:
        logger.warning("⚠️ MACHINE_ACTIVE is set to False. Engine is locked for maintenance.")
        logger.info("ℹ️ Jab aapka code setup pura ho jaye, tab ise 'True' kar dena.")
        return

    # Check Credentials (Only runs if machine is active)
    if not os.getenv("SUPABASE_URL") or not os.getenv("GROQ_API_KEY"):
        logger.error("🚨 Missing API Keys in Environment Variables!")
        return

    try:
        # Phase 1: Hunting
        logger.info("--- Phase 1: Starting Scout ---")
        if asyncio.iscoroutinefunction(run_scout): await run_scout()
        else: run_scout()

        # Phase 2: Expanding
        logger.info("--- Phase 2: Starting Multiplier ---")
        run_multiplier()

        # Phase 3: Writing
        logger.info("--- Phase 3: Starting Synthesizer ---")
        if asyncio.iscoroutinefunction(run_synthesizer): await run_synthesizer()
        else: await asyncio.to_thread(run_synthesizer)

        logger.info("🏆 MISSION ACCOMPLISHED: All phases completed!")

    except Exception as e:
        logger.error(f"🚨 Engine Failure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(start_engine())
    except KeyboardInterrupt:
        logger.info("🛑 Stopped by user.")
