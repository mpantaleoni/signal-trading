"""Pure metric functions for backtesting.

All functions are stateless: they take equity curve Series and return scalars.
No scipy dependency — Acklam's rational approximation is used for the
inverse normal CDF (needed for DSR).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


# ── core metrics ───────────────────────────────────────────────────────────────

def cagr(equity: pd.Series) -> float:
    """Compound Annual Growth Rate.

    equity: Series indexed by date (or datetime), values are portfolio equity.
    Returns NaN if fewer than 2 data points or zero start value.
    """
    if len(equity) < 2:
        return float("nan")
    start_val = float(equity.iloc[0])
    end_val = float(equity.iloc[-1])
    if start_val <= 0:
        return float("nan")

    start_date = pd.Timestamp(equity.index[0])
    end_date = pd.Timestamp(equity.index[-1])
    years = (end_date - start_date).days / 365.25
    if years <= 0:
        return float("nan")

    return (end_val / start_val) ** (1.0 / years) - 1.0


def daily_returns(equity: pd.Series) -> pd.Series:
    """Compute daily log-returns (or simple pct_change) from equity curve."""
    return equity.pct_change().dropna()


def annualised_volatility(equity: pd.Series, trading_days: int = 252) -> float:
    """Annualised return volatility from daily equity."""
    ret = daily_returns(equity)
    if len(ret) < 2:
        return float("nan")
    return float(ret.std() * math.sqrt(trading_days))


def sharpe(equity: pd.Series, risk_free_rate: float = 0.0, trading_days: int = 252) -> float:
    """Annualised Sharpe ratio (excess return / annualised vol), using daily returns."""
    ret = daily_returns(equity)
    if len(ret) < 2:
        return float("nan")
    daily_rf = risk_free_rate / trading_days
    excess = ret - daily_rf
    vol = float(excess.std())
    if vol == 0:
        return float("nan")
    return float(excess.mean() / vol * math.sqrt(trading_days))


def sortino(equity: pd.Series, risk_free_rate: float = 0.0, trading_days: int = 252) -> float:
    """Annualised Sortino ratio (downside deviation only)."""
    ret = daily_returns(equity)
    if len(ret) < 2:
        return float("nan")
    daily_rf = risk_free_rate / trading_days
    excess = ret - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    down_std = float(downside.std())
    if down_std == 0:
        return float("nan")
    return float(excess.mean() / down_std * math.sqrt(trading_days))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown fraction (0.0 = no drawdown, 0.20 = 20% peak-to-trough)."""
    if len(equity) < 2:
        return 0.0
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(-dd.min())


def max_drawdown_duration(equity: pd.Series) -> int:
    """Maximum drawdown duration in calendar days (from peak to recovery or end)."""
    if len(equity) < 2:
        return 0

    roll_max = equity.cummax()
    in_dd = equity < roll_max

    max_dur = 0
    current_start = None
    for ts, val in in_dd.items():
        if val and current_start is None:
            current_start = ts
        elif not val and current_start is not None:
            dur = (pd.Timestamp(ts) - pd.Timestamp(current_start)).days
            max_dur = max(max_dur, dur)
            current_start = None

    if current_start is not None:
        dur = (pd.Timestamp(equity.index[-1]) - pd.Timestamp(current_start)).days
        max_dur = max(max_dur, dur)

    return max_dur


def total_return(equity: pd.Series) -> float:
    """Simple total return fraction."""
    if len(equity) < 2:
        return float("nan")
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    if start <= 0:
        return float("nan")
    return (end - start) / start


def yearly_returns(equity: pd.Series) -> dict[int, float]:
    """Per-calendar-year simple returns."""
    equity = equity.copy()
    equity.index = pd.to_datetime(equity.index)
    result: dict[int, float] = {}
    for year, group in equity.groupby(equity.index.year):
        if len(group) >= 2:
            result[int(year)] = float((group.iloc[-1] - group.iloc[0]) / group.iloc[0])
        elif len(group) == 1:
            result[int(year)] = 0.0
    return result


# ── Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) ─────────────────────

def _acklam_inv_normal(p: float) -> float:
    """Acklam's rational approximation to the inverse normal CDF.

    Accurate to ~1e-9 over (0,1).  Pure Python, no scipy.
    Reference: https://home.online.no/~pjacklam/notes/invnorm/
    """
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]

    p_lo = 0.02425
    p_hi = 1 - p_lo

    if p < p_lo:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= p_hi:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (DSR) from Bailey & Lopez de Prado (2014).

    Estimates the probability that the observed Sharpe is truly positive
    (i.e. not a lucky result from multiple testing).

    Parameters
    ----------
    observed_sharpe : annualised Sharpe ratio of the strategy
    n_trials        : number of strategy variants tested (feeds the SR* benchmark)
    n_observations  : number of return observations (e.g. trading days)
    skewness        : skewness of returns (default 0 = normal)
    kurtosis        : kurtosis of returns (default 3 = normal)

    Returns
    -------
    Probability in [0, 1]. Values > 0.95 indicate the Sharpe is likely genuine.
    """
    if n_trials < 1 or n_observations < 4:
        return float("nan")

    # Expected maximum SR over n_trials IID standard-normal trials
    # Bailey & LdP Eq. (6): E[max SR] ≈ (1-γ) * Φ^{-1}(1 - 1/n_trials) + γ * Φ^{-1}(1-1/(n_trials*e))
    euler_gamma = 0.5772156649
    z1 = _acklam_inv_normal(1.0 - 1.0 / max(n_trials, 2))
    z2 = _acklam_inv_normal(1.0 - 1.0 / (max(n_trials, 2) * math.e))
    sr_star = (1 - euler_gamma) * z1 + euler_gamma * z2

    # Variance of estimated SR given non-normality (Bailey & LdP Eq. 7)
    sr_var = (1.0 / (n_observations - 1)) * (
        1.0
        + (observed_sharpe ** 2) / 2.0 * (kurtosis - 1)
        - observed_sharpe * skewness
        + (observed_sharpe ** 2) / 4.0 * (kurtosis - 1)
    )
    # Simplified version (assume normal → kurtosis=3, skewness=0)
    sr_std = math.sqrt((1.0 + observed_sharpe ** 2 / 2.0) / (n_observations - 1))

    if sr_std <= 0:
        return float("nan")

    z = (observed_sharpe - sr_star) / sr_std
    return _normal_cdf(z)


# ── trade-level metrics ────────────────────────────────────────────────────────

def win_rate(trades: list[dict]) -> float:
    if not trades:
        return float("nan")
    wins = sum(1 for t in trades if t.get("pnl_usd", 0) > 0)
    return wins / len(trades)


def avg_win(trades: list[dict]) -> float:
    wins = [t["pnl_usd"] for t in trades if t.get("pnl_usd", 0) > 0]
    return float(np.mean(wins)) if wins else 0.0


def avg_loss(trades: list[dict]) -> float:
    losses = [t["pnl_usd"] for t in trades if t.get("pnl_usd", 0) <= 0]
    return float(np.mean(losses)) if losses else 0.0


def avg_holding_days(trades: list[dict]) -> float:
    if not trades:
        return float("nan")
    return float(np.mean([t.get("sessions_held", 0) for t in trades]))
