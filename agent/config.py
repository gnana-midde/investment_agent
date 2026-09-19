"""Central paths for the investment agent."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# A hosted server points this at its disposable storage; locally it is ./data.
DATA = Path(os.getenv("INVESTMENT_AGENT_DATA_DIR", str(ROOT / "data"))).expanduser().resolve()
DATA.mkdir(parents=True, exist_ok=True)
