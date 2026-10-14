# CLAUDE.md — Short-Horizon Equity Signal System (v1)

Owner: Mauro Pantaleoni
Status: Phase 0 spec, approved for Phase 1 implementation

## 1. Purpose

Build a system that scans a universe of liquid stocks and ETFs once per trading day,
identifies entry and exit points for holds of roughly 2–20 trading days, and alerts the
owner on his iPhone. Version 1 is long-only, uses daily bars and executes manually.

The goal is NOT to maximise backtested return. The goal is a strategy whose net return
after costs and tax beats buy-and-hold SPY on a risk-adjusted basis, out-of-sample,
with a maximum drawdown of 20% or less.

## 2. Scope

In scope (v1):
- US large caps (historical S&P 500 members) plus a list of liquid ETFs
- Daily OHLCV data from free sources
- Two signal engines (A and B, section 4) and a shared risk layer
- Backtester, validation reports, daily scanner, Telegram alerts

Out of scope (v1): intraday data, short selling, options, news/sentiment, European stocks,
automated order execution, machine learning models. Do not add any of these without
explicit instruction.

## 3. Data

- Primary source: yfinance. Secondary source for cross-checks: Stooq.
- All data goes through a single `DataProvider` interface so a paid survivorship-free source
  (e.g. Norgate, EODHD) can replace yfinance later through configuration only.
- Cache everything locally (Parquet files plus an SQLite metadata table). Never re-download
  data that is already cached and valid.
- Historical S&P 500 membership is reconstructed from the public constituent-change history.
  The backtest may only trade a stock on dates when it was an index member.
- Members with no available price data (typically delisted) are logged in a
  `missing_coverage` report showing count and share of the universe per year.
- KNOWN BIAS: free data under-represents delisted losers. All stock-universe results are an
  optimistic upper bound. This must be stated on every report.
- Data integrity checks: missing bars, zero or negative prices, split/dividend adjustment
  jumps, stale prices. Failures are logged, never silently filled.
- ETF cross-check universe: a configurable list of liquid sector, country and broad-index ETFs
  (e.g. SPY, QQQ, IWM, XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLU, XLB, EFA, EEM, EWG, EWJ, TLT, GLD).

## 4. Strategy definitions

General timing rule for both engines: signals are computed from data up to and including
the close of day t. Orders are filled at the open of day t+1. No exceptions.

### Engine A — Trend-filtered pullback (holds of about 2–10 days)

Entry (all conditions true at close of day t):
- Stock is a current universe member and passes the liquidity filter
  (20-day average dollar volume above a configurable threshold)
- Close > 200-day SMA
- RSI(2) < `rsi_entry` (default 10)
- Regime filter: SPY close > SPY 200-day SMA

Exit (first condition hit, evaluated at close, filled next open):
- Close > 5-day SMA
- Time stop: `max_hold_days` trading days (default 10)
- Close < 200-day SMA

Tunable parameters (and ONLY these): `rsi_entry` in {5, 10}, `max_hold_days` in {5, 10}.

### Engine B — Short-term momentum in high-activity stocks (holds of about 1 month)

- Rebalance on the first trading day of each month.
- Activity filter: keep the top third of the universe by relative trading activity.
  Proxy: 21-day average volume divided by 252-day average volume. (The research uses share
  turnover = volume / shares outstanding; free historical shares-outstanding data is not
  reliable, so this proxy is a documented deviation.)
- Rank the filtered stocks by return over the past 21 trading days, skipping the most
  recent 2 days.
- Hold the top `top_n` stocks (default 10) until the next rebalance.
- Regime filter: same as Engine A.

Tunable parameter (and ONLY this): `top_n` in {5, 10, 15}.

### Shared risk layer

- Position sizing by volatility: each position sized so that a 2×ATR(20) move equals
  `risk_per_trade` of portfolio equity (default 0.75%), capped at 10% of equity per position.
- Maximum concurrent positions: Engine A 8, Engine B `top_n`, combined portfolio 15.
- Sector cap: max 3 positions per sector (current sector classification; documented limitation).
- Drawdown control: warning at 10% portfolio drawdown from peak; at 20%, no new entries and the
  system is flagged as STOPPED pending manual review.
- Earnings blackout (optional, off by default): no new entries within 3 trading days before a
  known earnings date, if reliable dates are available.

## 5. Costs and tax

- Commission: configurable per trade (fixed + percentage).
- Slippage: default 5 bp per side, configurable.
- Stress test: every result is also reported with costs doubled.
- Tax: German flat tax, default 26.375% (25% + solidarity surcharge), applied annually to net
  realised gains with loss carry-forward. Rate and annual allowance are configurable.

## 6. Validation protocol

- Development period: 2005-01-01 to 2018-12-31.
- Hold-out period: 2019-01-01 to present. Do NOT run anything on the hold-out period unless
  the owner explicitly instructs it. Each engine version gets one hold-out run.
- Walk-forward evaluation on the development period (e.g. 5-year train, 1-year test, rolling).
- Experiment registry: every backtest run is logged (parameters, period, metrics, git commit).
  The number of variants tested feeds a Deflated Sharpe Ratio calculation.
- Benchmark: buy-and-hold SPY over the same period, same cost and tax treatment.

Pass criteria (all must hold, on hold-out, net of costs and tax):
1. Sharpe ratio above the SPY benchmark
2. Maximum drawdown ≤ 20%
3. Positive total return
4. Still beats SPY on Sharpe with costs doubled
5. Same rules on the ETF universe are not loss-making

Standard report metrics: CAGR, volatility, Sharpe, Sortino, max drawdown and duration,
win rate, average win/loss, average holding period, turnover, exposure, number of trades,
costs paid, tax paid, deflated Sharpe, yearly returns table, equity curve vs SPY.

## 7. Architecture

- Python 3.11+, pandas, numpy, pyarrow, pytest, PyYAML. Ask before adding other dependencies.
- Event-driven daily backtester written in this repo (transparency over speed).
- Reuse plumbing from the owner's earlier ATS-MVP repo where it fits (config, SQLite).

```
config/          YAML: universe, engines, risk, costs, tax, alerts
src/data/        DataProvider interface, yfinance + Stooq providers, cache, integrity checks
src/universe/    historical S&P 500 membership, ETF list, liquidity filter
src/signals/     indicators, engine_a.py, engine_b.py
src/portfolio/   sizing, risk layer, position book
src/backtest/    event loop, fills, costs, tax, experiment registry
src/reports/     metrics, deflated Sharpe, HTML/Markdown reports
src/scanner/     daily run: update data, compute signals, produce orders list
src/alerts/      Telegram sender
tests/
```

## 8. Deployment (Phase 4)

- The scanner runs on a scheduled GitHub Actions workflow in a private repository, after
  the US close: 22:30 UTC on weekdays (works year-round across daylight-saving changes).
- The cache persists between runs (Actions cache or committed compact data).
- Output: Telegram message to the owner's iPhone with new entries, exits, position sizes,
  current positions, and drawdown status. Also a daily log file of hypothetical fills.
- Secrets (Telegram bot token, chat ID) live in GitHub Secrets, never in the repo.
- If the run fails, a failure alert is sent.

## 9. Rules for Claude Code

1. No look-ahead: nothing computed at day t may use data after the close of day t.
   Write explicit tests for this.
2. Never touch the hold-out period unless instructed.
3. Do not add strategies, parameters, indicators or features beyond this spec.
4. Every module gets unit tests; the test suite must pass before a phase is reported done.
5. One phase per session. End each session with a short report: what was built,
   test status, open issues, deviations from this spec.
6. Prefer clear, readable code over clever or fast code.
7. When the spec is ambiguous, ask rather than assume.

## 10. Phases and gates

| Phase | Content | Gate |
|---|---|---|
| 1 | Data layer, cache, universe membership, integrity checks | Data report reviewed |
| 2 | Backtester + Engine A + reports, development period only | Gate 1: Engine A passes walk-forward; one hold-out run passes |
| 3 | Engine B + combined portfolio | Gate 2: combination beats best single engine out-of-sample |
| 4 | Daily scanner on GitHub Actions + Telegram, paper trading 2–3 months | Gate 3: paper results consistent with backtest |
| 5 | Small live trading, manual execution at Fineco, live vs paper tracking | Ongoing review |

Before Phase 4: re-run the final validation on a paid survivorship-free data source.

## 11. Open decisions

- Liquidity threshold value
- Commission model for the owner's broker
- Whether to enable the earnings blackout
