"""Put src/ on sys.path so scripts run without an install step."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
DATA = RESULTS / "data"
MODELS = RESULTS / "models"
for _d in (FIGURES, DATA, MODELS):
    _d.mkdir(parents=True, exist_ok=True)
