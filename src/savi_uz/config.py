"""Configuration helpers for savi.uz ingestion flows."""

from __future__ import annotations

import os
from pathlib import Path

#: Searched in order; the first name present wins.
FRED_KEY_NAMES = ("FRED_API_KEY", "FRED_API", "FRED_KEY")
ALPHAVANTAGE_KEY_NAMES = ("ALPHAVANTAGE_API_KEY", "ALPHAVANTAGE_API")
TIINGO_KEY_NAMES = ("TIINGO_API_KEY", "TIINGO_API", "TIINGO_TOKEN")

#: The SEC blocks default library user agents outright and asks that yours
#: identify you; set SEC_USER_AGENT to "project (you@example.com)".
SEC_USER_AGENT_NAMES = ("SEC_USER_AGENT", "SEC_UA")
DEFAULT_SEC_USER_AGENT = "savi-uz-research/1.0"


def load_dotenv(path: str | Path = ".env", override: bool = False) -> dict[str, str]:
    """Read a ``.env`` file into ``os.environ`` without taking a dependency.

    Existing environment variables win unless ``override`` is set, so a shell
    export always beats a stale file.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return {}

    loaded: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, raw_value = line.partition("=")
        name = name.strip().removeprefix("export ").strip()
        value = raw_value.strip().strip("\"'")
        if not name:
            continue
        loaded[name] = value
        if override or name not in os.environ:
            os.environ[name] = value
    return loaded


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def get_alphavantage_api_key() -> str:
    """Return AlphaVantage API key from environment."""
    load_dotenv()
    api_key = _first_env(ALPHAVANTAGE_KEY_NAMES)
    if not api_key:
        raise ValueError("Missing ALPHAVANTAGE_API_KEY environment variable.")
    return api_key


def get_tiingo_api_key() -> str:
    """Return the Tiingo API token from the environment or ``.env``."""
    load_dotenv()
    api_key = _first_env(TIINGO_KEY_NAMES)
    if not api_key:
        raise ValueError(
            "Missing TIINGO_API_KEY in the environment or .env -- "
            "free key at https://www.tiingo.com/account/api/token"
        )
    return api_key


def get_sec_user_agent() -> str:
    """User-Agent for data.sec.gov.

    Unlike the API keys this has a working default, because the SEC only
    requires that the agent not be a default library string -- but their fair
    -access policy asks for a contact address, so setting one is polite and
    keeps you off the rate-limit blocklist.
    """
    load_dotenv()
    return _first_env(SEC_USER_AGENT_NAMES) or DEFAULT_SEC_USER_AGENT


def get_fred_api_key() -> str:
    """Return the FRED/ALFRED API key from the environment or ``.env``.

    A key is what separates current values from vintages: the keyless CSV
    endpoint only ever serves the latest revision.
    """
    load_dotenv()
    api_key = _first_env(FRED_KEY_NAMES)
    if not api_key:
        raise ValueError(
            "Missing FRED API key. Set FRED_API_KEY (or FRED_API) in the environment "
            "or in .env -- free key at https://fredaccount.stlouisfed.org/apikeys"
        )
    return api_key
