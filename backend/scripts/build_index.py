import logging
import os
import sys
from pathlib import Path

# Add the project root ('backend/') to the Python path
# This allows the script to import modules from the 'app' package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting manual index build process...")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key or openai_api_key.strip() == "":
        logger.warning(
            "OPENAI_API_KEY environment variable is not set. Skipping index build."
        )
        logger.info("Index will be built at runtime when the API key is available.")
        sys.exit(0)

    try:
        from app.agents.knowledge_agent.main import build_index_from_scratch

        build_index_from_scratch()
        logger.info("Index build process completed successfully.")
    except Exception as e:
        logger.exception(f"Index build process failed: {e}")
        logger.info("Index will be built at runtime when the application starts.")
        sys.exit(0)
