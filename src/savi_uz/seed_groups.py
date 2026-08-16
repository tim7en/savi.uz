"""Hand-labelled risk groups used as the a-priori hypothesis to test against data.

Keys are theme labels, values are Binance base assets. The clustering script
compares this intuition against measured correlations: groups that hold together
confirm the theme, groups that scatter were never one risk factor to begin with.
"""

from __future__ import annotations

SEED_RISK_GROUPS: dict[str, tuple[str, ...]] = {
    "US mega-cap tech": ("AAPL", "MSFT", "GOOGL", "META", "AMZN"),
    "AI / semiconductors US": ("NVDA", "AMD", "AVGO", "MU", "MRVL", "LRCX", "KLAC", "AMAT"),
    "AI / semiconductors Asia": ("TSM", "SKHYNIX", "SAMSUNG", "HANMI", "GIGADEV", "ZHONGJI"),
    "China internet": ("BABA", "TENCENT", "HK0700", "MEITUAN", "KUAISHOU"),
    "China AI": ("MINIMAX", "ZHIPU"),
    "Consumer China": ("POPMART", "HK1810"),
    "Financials": ("JPM", "GS", "BX", "V"),
    "Crypto-linked equities": ("COIN", "MSTR", "HOOD", "IREN", "BMNR"),
    "Software / cyber": ("CRM", "NOW", "CRWD", "PANW", "PLTR"),
    "Healthcare": ("LLY", "NVO", "HIMS"),
    "Consumer / retail": ("WMT", "COST", "HD", "KO"),
    "Industrials / energy": ("CAT", "GEV", "XLE"),
    "Autos / mobility": ("TSLA", "RIVN", "HYUNDAI", "UBER"),
    "Space": ("RKLB", "ASTS"),
    "Broad indices": ("SPY", "QQQ", "IWM", "KODEX200"),
    "Country exposures": ("EWJ", "EWY", "EWT", "EWZ"),
    "Vol / rates diversifiers": ("UVXY", "TMF", "TBT"),
    "Precious metals": ("XAU", "XAG"),
}


def seed_base_assets() -> tuple[str, ...]:
    """Every base asset named in the seed table, de-duplicated and sorted."""
    return tuple(sorted({base for bases in SEED_RISK_GROUPS.values() for base in bases}))


def seed_group_by_base() -> dict[str, str]:
    """Reverse index from base asset to its hand-labelled theme."""
    return {base: label for label, bases in SEED_RISK_GROUPS.items() for base in bases}
