# =========================================================
# FILE: tests/test_smoke.py
# =========================================================
from infrawatch.config import load_config

def test_load_config():
    cfg = load_config()
    assert cfg.log_level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}