"""
Global pytest configuration and root path resolvers.
Ensures backend, tools, config, and datasets are seamlessly discovered during test execution.
"""
import os
import sys
from pathlib import Path

# Disable telemetry and online lookups that cause socket timeout in sandbox
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("ENABLE_DEMO_WORKSPACE", "false")

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"

for p in [str(BACKEND), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("PROJECT_ROOT", str(ROOT))
