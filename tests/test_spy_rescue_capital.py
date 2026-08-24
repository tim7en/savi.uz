import pandas as pd
import pytest

from scripts.run_spy_rescue_capital_study import simulate


def test_temporary_bridge_is_repaid_with_fixed_premium_on_recovery():
    index = pd.date_range("2020-01-01", periods=3)
    returns = pd.Series([0.0, -0.50, 1.0], index=index)
    zero = pd.Series(0.0, index=index)

    _, stats = simulate(
        returns,
        zero,
        zero,
        levels=(1.0,),
        leverage_thresholds=(),
        reserve_thresholds=(),
        reserve_fractions=(),
        rescue_multiple=1.0,
        profit_sweep_frequency="annual",
        rescue_threshold=0.50,
        rescue_external_only=True,
        repay_rescue_on_recovery=True,
    )

    opening = 10_000 / 12
    bridge = opening / 2
    assert stats["rescue_calls"] == 1
    assert stats["rescue_exits"] == 1
    assert stats["total_rescue_external"] == pytest.approx(bridge)
    assert stats["total_rescue_repaid"] == pytest.approx(bridge * 1.10)
    assert stats["total_rescue_profit_retained"] == pytest.approx(bridge * 0.90)
    assert stats["ending_active_rescue"] == 0.0
