"""Core logic for the shipping route risk tool. No Streamlit code in here,
so the same functions can be used by the UI, an AI agent, or ML models."""
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
