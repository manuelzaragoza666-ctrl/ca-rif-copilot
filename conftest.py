"""Makes `rif_copilot` importable when running the suite with pytest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
