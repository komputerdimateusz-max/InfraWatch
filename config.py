# =========================================================
# FILE: src/infrawatch/config.py
# =========================================================
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

@dataclass(frozen=True)
class AppConfig:
    data_dir: Path
    log_level: str

def load_config() -> AppConfig:
    load_dotenv()
    data_dir = Path(os.getenv("EO_DATA_DIR", "./satellite_data")).resolve()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    return AppConfig(data_dir=data_dir, log_level=log_level)