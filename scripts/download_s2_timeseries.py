from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from infrawatch.ingestion.timeseries import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
