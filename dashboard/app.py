"""Public Streamlit entrypoint for the global volatility deployment."""

# ruff: noqa: E402 - Streamlit needs the repository path bootstrap first.

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.global_app import _render


_render()
