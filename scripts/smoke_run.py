# =========================================================
# FILE: scripts/smoke_run.py
# =========================================================
from infrawatch.config import load_config

def main():
    cfg = load_config()
    print("InfraWatch config OK:", cfg)

if __name__ == "__main__":
    main()