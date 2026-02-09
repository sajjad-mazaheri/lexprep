import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
WEB = ROOT / "web"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(WEB) not in sys.path:
    sys.path.insert(0, str(WEB))

# Prevent model warmup during test collection
os.environ.pop("LEXPREP_WARMUP", None)
