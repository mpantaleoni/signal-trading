# CLAUDE.md — Short-Horizon Equity Signal System (v1)

Owner: Mauro Pantaleoni
Status: Phase 0 spec, approved for Phase 1 implementation

## 1. Purpose

Build a system that scans a universe of liquid stocks and ETFs once per trading day,
identifies entry and exit points for holds of roughly 2–20 trading days, and alerts the
owner on his iPhone. Version 1 is long/short, uses daily bars and executes manually.

The goal is NOT to maximise backtested return. The goal is a strategy whose net return
after costs and tax beats buy-and-hold SPY on a risk-adjusted basis, out-of-sample,
with a maximum drawdown of 30% or less.

## 2. Scope

In scope (v1):
- US large caps (historical S&P 500 members) plus a list of liquid ETFs
- Daily OHLCV data from free sources
- Momentum engine with long top-N and short bottom-N legs, shared risk layer
- Backtester, validation reports, daily scanner, Telegram alerts
- Short selling (manual execution, US equities only)

Out of scope (v1): intraday data, options, news/sentiment, European stocks,
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

## 4. Strategy definition

General timing rule: signals are computed from data up to and including the close of day t.
Orders are filled at the open of day t+1. No exceptions.

### Momentum Engine — Cross-sectional momentum (holds of about 1 month)

Rebalance on the first trading day of each month.

Signals (both required; ticker dropped if either is None):
- **12-1 momentum**: `(close[t-21] - close[t-273]) / close[t-273]`. Requires ≥ 273 rows.
- **52-week-high pct**: `close[t] / max(high[t-252:t])`. Requires ≥ 252 rows.
- **Composite rank**: rank each score percentile within the universe, then average:
  `0.5 × rank_12_1 + 0.5 × rank_52wh`.
  - **Long leg**: top `top_n` (default 15) descending.
  - **Short leg**: bottom `short_n` (default 10) ascending. No SMA filter applied (shorts target laggards).

Regime filter (asymmetric):
- SPY bull (12-1 momentum > 0 AND close > 200-day SMA): open/hold longs + open/hold shorts.
- SPY bear: exit all longs; short book remains active (bearish conditions benefit shorts).
- Drawdown stop: exit ALL positions (both sides) pending manual review.

Tunable parameters: `top_n` in {10, 15, 20}, `short_n` in {5, 10, 15}.

### Risk layer

- Position sizing (volatility-targeted):
  ```
  stock_annual_vol  = rolling 21-day std of daily log-returns × sqrt(252)
  position_value    = (target_portfolio_vol / N) × equity / max(stock_annual_vol, vol_floor)
  shares            = floor(position_value / price)
  cap               = floor(max_position_pct × equity / price)
  final_shares      = min(shares, cap)
  ```
  `target_portfolio_vol` default 0.15, `vol_floor` 0.05, N = top_n.
- Maximum concurrent positions: 20, counting long + short combined.
- Sector cap: max 3 positions per sector, long + short combined (current sector classification; documented limitation).
- Short borrow cost: 50 bps/year annualised, accrued daily against short notional (`config/costs.yaml`).
- Drawdown control: warning at 15% portfolio drawdown from peak; at 30%, no new entries and ALL
  positions (long and short) are exited. System is flagged as STOPPED pending manual review.
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
- Walk-forward evaluation on the development period: 9 folds (train 5 years, test 1 year,
  rolling). Param grid: `top_n ∈ {10, 15, 20}` × `short_n ∈ {5, 10, 15}`, n_trials = 81.
- Experiment registry: every backtest run is logged (parameters, period, metrics, git commit).
  The number of variants tested feeds a Deflated Sharpe Ratio calculation.
- Benchmark: buy-and-hold SPY over the same period, same cost and tax treatment.

Pass criteria (all must hold, on hold-out, net of costs and tax):
1. Sharpe ratio above the SPY benchmark
2. Maximum drawdown ≤ 35%
3. Positive total return
4. Still beats SPY on Sharpe with costs doubled
5. Same rules on the ETF universe are not loss-making

Target (development): Sharpe > 0.40, max drawdown < 30%, walk-forward avg Sharpe > 0.30.

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
src/signals/     indicators, engine_momentum.py
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

All resolved:

- **Liquidity threshold**: $5M 20-day average dollar volume (`config/universe.yaml`)
- **Commission model**: Fineco flat fee €12.95 per trade (one-way), 0% variable,
  5 bp slippage per side (`config/costs.yaml`)
- **Earnings blackout**: OFF for now; revisit in Phase 2 after measuring backtest impact
- **Short leg**: enabled, `short_n=10`, borrow cost 50 bps/year, asymmetric regime (long exits in bear, shorts remain)
