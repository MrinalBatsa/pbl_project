"""
utils.py
--------
Shared utility functions for the Credit Default XAI project.
Provides helpers for paths, logging, and plot saving.
"""

import os
import logging
from pathlib import Path

# ── Project-wide paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"

# ── Reproducibility ─────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COL = "default payment next month"

# ── Logging setup ───────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a consistently-configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s — %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    for d in (PLOTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_plot(fig, filename: str, dpi: int = 150) -> None:
    """
    Save a matplotlib figure to the plots directory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save.
    filename : str
        Name of the output file (e.g. 'roc_curve.png').
    dpi : int
        Resolution in dots per inch.
    """
    ensure_dirs()
    filepath = PLOTS_DIR / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    get_logger("utils").info("Plot saved → %s", filepath)
