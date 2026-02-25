import logging
import sys
from scout import main as run_scout
from multiplier import main as run_multiplier
from synthesizer import main as run_synthesizer

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - AstraOS - %(levelname)s - %(message)s')
logger = logging.getLogger("AstraMain")

def start_engine():
    logger.info("🌌 Astra Infinity Engine Starting...")

    try:
        # Phase 1: Hunting
        logger.info("--- Phase 1: Starting Scout ---")
        # Yahan hum scout ka main function call karenge
        # run_scout() 

        # Phase 2: Expanding
        logger.info("--- Phase 2: Starting Multiplier ---")
        run_multiplier()

        # Phase 3: Writing
        logger.info("--- Phase 3: Starting Synthesizer ---")
        # run_synthesizer() 

        logger.info("🏆 All Phases Completed Successfully!")

    except Exception as e:
        logger.error(f"🚨 Engine Failure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_engine()
