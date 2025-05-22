import logging
import os
from datetime import datetime

def setup_logger(log_name="analyze_log"):
    os.makedirs("src/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"src/logs/{log_name}_{timestamp}.log"

    logging.basicConfig(
        filename=log_filename,
        level=logging.DEBUG,  # Set to DEBUG for verbose output
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger()
    return logger
