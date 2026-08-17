"""US-listed proxies for the non-US Binance trad-FI contracts.

133 of the 163 contracts are already US equities and need no proxy. The other 30
are Hong Kong and Korea listings, commodity futures and two pre-IPO names, and a
strategy that can only trade US hours needs a US-listed instrument that carries
the same risk.

The candidates here are hypotheses, not answers. Every one is scored against the
real underlying by :mod:`savi_uz.proxy_tracking` before it is used, because the
quality varies enormously: an ADR of the same company is close to fungible with
its ordinary shares, while a country ETF only shares the country.

Ordering within each tuple is the prior -- best structural match first -- and is
used only to break ties between candidates that measure the same.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Ranked by how much of the underlying's risk the instrument can carry.
#: ``adr`` is the same company; ``peer`` is a different company in the same
#: business; ``sector``/``country`` share only a factor.
PROXY_KINDS: tuple[str, ...] = ("direct", "adr", "peer", "sector", "country", "commodity")


@dataclass(frozen=True)
class ProxyCandidate:
    ticker: str
    kind: str
    rationale: str


def _c(ticker: str, kind: str, rationale: str) -> ProxyCandidate:
    return ProxyCandidate(ticker, kind, rationale)


#: Keyed by Binance base asset. Only non-US bases appear; US bases proxy to
#: themselves and are filled in by the builder.
US_PROXY_CANDIDATES: dict[str, tuple[ProxyCandidate, ...]] = {
    # -- Hong Kong / China ------------------------------------------------
    "HK0700": (
        _c("TCEHY", "adr", "Tencent ADR, same company"),
        _c("KWEB", "sector", "China internet ETF, Tencent is a top holding"),
        _c("MCHI", "country", "MSCI China"),
    ),
    "TENCENT": (
        _c("TCEHY", "adr", "Tencent ADR, same company"),
        _c("KWEB", "sector", "China internet ETF, Tencent is a top holding"),
        _c("MCHI", "country", "MSCI China"),
    ),
    "HK1810": (
        _c("XIACY", "adr", "Xiaomi ADR, same company"),
        _c("KWEB", "sector", "China internet/tech ETF"),
        _c("MCHI", "country", "MSCI China"),
    ),
    "MEITUAN": (
        _c("MPNGY", "adr", "Meituan ADR, same company"),
        _c("KWEB", "sector", "China internet ETF, Meituan is a top holding"),
        _c("MCHI", "country", "MSCI China"),
    ),
    "KUAISHOU": (
        _c("KSHTY", "adr", "Kuaishou ADR, same company but thinly traded"),
        _c("KWEB", "sector", "China internet ETF"),
        _c("MCHI", "country", "MSCI China"),
    ),
    "POPMART": (
        _c("PMRTY", "adr", "Pop Mart ADR, same company but thinly traded"),
        _c("KWEB", "sector", "China consumer internet"),
        _c("MCHI", "country", "MSCI China"),
    ),
    "GIGADEV": (
        _c("SMH", "sector", "Semiconductor ETF"),
        _c("SOXX", "sector", "Semiconductor ETF, different weighting"),
        _c("MCHI", "country", "MSCI China"),
    ),
    "ZHONGJI": (
        # Zhongji Innolight makes optical transceivers for AI data centres; the
        # closest US businesses are the optical module makers, not a China ETF.
        _c("COHR", "peer", "Coherent, optical transceivers for AI data centres"),
        _c("FN", "peer", "Fabrinet, optical module contract manufacturer"),
        _c("AAOI", "peer", "Applied Optoelectronics, optical networking"),
        _c("SMH", "sector", "Semiconductor ETF"),
    ),
    # Pre-IPO China AI names: no listing anywhere, so nothing can track them.
    # Kept so the report states that explicitly rather than omitting them.
    "MINIMAX": (_c("KWEB", "sector", "China internet ETF; pre-IPO, no real proxy"),),
    "ZHIPU": (_c("KWEB", "sector", "China internet ETF; pre-IPO, no real proxy"),),
    "CSOPSAMSUNG2L": (
        _c("MU", "peer", "Memory pure-play; the contract is 2x leveraged Samsung"),
        _c("EWY", "country", "MSCI Korea"),
    ),
    "CSOPSKHYNIX2L": (
        _c("MU", "peer", "Memory pure-play; the contract is 2x leveraged SK Hynix"),
        _c("EWY", "country", "MSCI Korea"),
    ),
    # -- Korea -------------------------------------------------------------
    # Samsung and SK Hynix have no liquid US ADR, so memory peers and the
    # country ETF are the only routes.
    "SAMSUNG": (
        _c("MU", "peer", "Micron, closest listed memory/HBM business"),
        _c("EWY", "country", "MSCI Korea, Samsung is the largest weight"),
        _c("SMH", "sector", "Semiconductor ETF"),
    ),
    "SKHYNIX": (
        _c("MU", "peer", "Micron, the memory/HBM pure-play comparison"),
        _c("EWY", "country", "MSCI Korea, SK Hynix is a top weight"),
        _c("SOXX", "sector", "Semiconductor ETF"),
    ),
    "SAMSUNGEM": (
        _c("EWY", "country", "MSCI Korea"),
        _c("SMH", "sector", "Semiconductor/components"),
    ),
    "HYUNDAI": (
        _c("EWY", "country", "MSCI Korea"),
        _c("CARZ", "sector", "Global auto manufacturers ETF"),
    ),
    "NAVER": (
        _c("EWY", "country", "MSCI Korea"),
    ),
    "LGELECTRONICS": (
        _c("EWY", "country", "MSCI Korea"),
    ),
    "HANMI": (
        _c("EWY", "country", "MSCI Korea"),
    ),
    "KODEX200": (
        # KODEX 200 tracks the KOSPI 200; EWY is the US-listed equivalent
        # exposure, so this is the one Korean name with a true index match.
        _c("EWY", "country", "MSCI Korea ETF, the US equivalent of KOSPI 200"),
    ),
    # -- Commodities -------------------------------------------------------
    "XAU": (
        _c("GLD", "commodity", "Spot gold ETF"),
        _c("IAU", "commodity", "Spot gold ETF, cheaper"),
        _c("GDX", "sector", "Gold miners, levered to the metal"),
    ),
    "XAG": (
        _c("SLV", "commodity", "Spot silver ETF"),
        _c("SIL", "sector", "Silver miners"),
    ),
    "XPT": (_c("PPLT", "commodity", "Physical platinum ETF"),),
    "XPD": (_c("PALL", "commodity", "Physical palladium ETF"),),
    "CL": (
        _c("USO", "commodity", "WTI futures ETF"),
        _c("XLE", "sector", "Energy equities"),
    ),
    "BZ": (
        _c("BNO", "commodity", "Brent futures ETF"),
        _c("USO", "commodity", "WTI futures ETF, close substitute"),
    ),
    "NATGAS": (
        _c("UNG", "commodity", "Henry Hub futures ETF"),
        _c("XOP", "sector", "Oil and gas E&P equities"),
    ),
    "COPPER": (
        _c("CPER", "commodity", "Copper futures ETF"),
        _c("COPX", "sector", "Copper miners"),
        _c("FCX", "peer", "Freeport-McMoRan, the liquid copper equity"),
    ),
    # -- No listing anywhere ----------------------------------------------
    "ANTHROPIC": (),
    "OPENAI": (),
}

#: Removing this from both sides separates a real tracking relationship from
#: two things that merely both rise when the US market does.
MARKET_FACTOR = "SPY"


def candidates_for(base_asset: str) -> tuple[ProxyCandidate, ...]:
    return US_PROXY_CANDIDATES.get(base_asset, ())


def proxy_tickers() -> tuple[str, ...]:
    """Every US ticker that has to be downloaded, plus the market factor."""
    tickers = {
        candidate.ticker
        for candidates in US_PROXY_CANDIDATES.values()
        for candidate in candidates
    }
    tickers.add(MARKET_FACTOR)
    return tuple(sorted(tickers))
