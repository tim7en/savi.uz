"""Configuration helpers for savi.uz ingestion flows."""

from __future__ import annotations

import os


def get_alphavantage_api_key() -> str:
    """Return AlphaVantage API key from environment."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing ALPHAVANTAGE_API_KEY environment variable.")
    return api_key
