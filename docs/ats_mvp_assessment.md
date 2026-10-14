# Old Repository Assessment — Read-Only Audit for Phase 1

**Date:** 2026-10-14
**Purpose:** Classify every module in the four legacy repos as REUSE-AS-IS / ADAPT / DISCARD and flag data-handling risks before Phase 1 implementation.
**Agreed approach:** Fresh repo, selective porting. Do not fork any old repo as a base. Each ported piece gets its own tests and a commit message naming its origin.

---

## Repositories assessed

| Repo | Path | Description | Quality |
|------|------|-------------|---------|
| `ats_mvp_starter` | `/Users/mauro/ats_mvp_starter/` | ATS-MVP: EMA/SMA + sentiment + Degiro live loop | Prototype/MVP |
| `A_Crypto_trading` | `/Users/mauro/A_Crypto_trading/` | BTC Trader: hexagonal-arch crypto system, ~30 strategies | High |
| `trading_bot` | `/Users/mauro/trading_bot/` | Sentiment + technical signals, DEGIRO skeleton | Low/Prototype |
| `trading-bot-starter` | `/Users/mauro/trading-bot-starter/` | MA-crossover paper-trading starter | Starter-grade |

---

## Summary verdict

**Primary source for porting: `A_Crypto_trading` (btc_trader)**
It is the only production-quality codebase. ~80% of it is asset-class-agnostic and maps almost directly onto Phase 1 and beyond. The other three are educational prototypes with critical data-handling bugs; nothing should be ported from them except minor config/utility patterns.

---

## Cross-cutting risks found in all repos

| Risk category | Found in | Notes |
|---------------|----------|-------|
| **Look-ahead bias** | `trading_bot`, `trading-bot-starter`, `ats_mvp_starter` | Backtest uses full future data, or signal and fill on same bar |
| **Forward-fill without timestamp validation** | `ats_mvp_starter` (features.py, news.py) | `.ffill()` propagates end-of-day sentiment into future bars |
| **Adjusted-price confusion** | `trading-bot-starter` (`auto_adjust=True`), `ats_mvp_starter` | Retroactive price rewrites break backtests |
| **Timezone mixing** | `trading_bot`, `trading-bot-starter`, `ats_mvp_starter` | UTC data mixed with broker-local timestamps |
| **Same-bar signal execution** | All four repos | Signal generated from close t, order filled at close t — impossible live |

`A_Crypto_trading` is mostly clean on all five points. UTC throughout; 1-day sentiment lag; fills at bar close as a deliberate, documented assumption.

---

## Repo 1 — `ats_mvp_starter`

### Per-file verdicts

| File | Purpose | Verdict | Risks |
|------|---------|---------|-------|
| `configs/base.yaml` | Config schema (strategy params, live loop, broker, news) | ADAPT | Market hours hardcoded to US/NYC; otherwise good structure |
| `src/ats_mvp/config.py` | YAML → typed dataclasses | REUSE-AS-IS | Clean; `yaml.safe_load`; nested dicts handled correctly |
| `src/ats_mvp/data.py` | `load_prices()` + `load_sentiment_csv()` | ADAPT | `auto_adjust=False` is correct; sentiment CSV assumes `;` separator and `,` decimal (non-standard); no date-alignment validation |
| `src/ats_mvp/features.py` | SMA features + sentiment merge | **DISCARD** | **FORWARD-FILL BUG**: `.ffill()` on resampled sentiment propagates end-of-day articles into next-day bars. Same-bar signal risk. |
| `src/ats_mvp/strategy.py` | EMA + ATR + regime + news (`TAPlusSentiment`) | ADAPT | Same-bar signal risk (see above). ATR zero-check is good. Integer size cast can underflow. |
| `src/ats_mvp/strategy_simple.py` | Simpler SMA-cross variant | ADAPT | Same risks as above. |
| `src/ats_mvp/backtest_pipeline.py` | Load → feature → backtest orchestration | REUSE-AS-IS | Solid prepare → run → extract pattern. `dropna()` correct here. |
| `src/ats_mvp/run.py` | CLI entry for batch backtesting | REUSE-AS-IS | Simple and correct. |
| `src/ats_mvp/live_loop.py` | Live trading loop with market-hours logic | ADAPT | **Market-hours detection and UTC↔TZ scheduling logic is the most valuable piece in this repo.** Order logic uses SMA-only (doesn't match strategy.py). Recomputes full backtest each iteration (expensive). |
| `src/ats_mvp/news.py` | `NewsSentimentEngine` + FinBERT pipeline | ADAPT | Forward-fill risk; articles after close ffilled into next day. FinBERT model loaded fresh each time. Fix timestamp validation before using. |
| `src/ats_mvp/news_aggregator.py` | Standalone news fetcher (Google, GDELT, Reddit…) | DISCARD | Not integrated into ATS pipeline; optional deps without fallback. |
| `src/ats_mvp/brokers/degiro.py` | Degiro connector wrapper | ADAPT | Clean abstraction; handles OTP/TOTP. No position sizing, hardcoded `default_size`. Logout not guaranteed on exception. |
| `src/ats_mvp/utils.py` | `ensure_dir()` + Loguru logger | REUSE-AS-IS | Minimal, correct. |
| `tests/test_config.py` | Config loading test | REUSE-AS-IS | Basic but correct pattern. |
| `tests/test_features.py` | Feature engineering tests | DISCARD | Does not test the forward-fill risk or timezone handling. |

**What to port from `ats_mvp_starter`:**
1. Config architecture (`config.py` + YAML schema style) — superseded by Phase 1 plan's own YAML design, so use as reference only
2. Market-hours scheduling logic from `live_loop.py` — needed in Phase 4; save for then
3. Degiro broker wrapper pattern — needed in Phase 5

---

## Repo 2 — `A_Crypto_trading` (btc_trader)

### Per-file verdicts

| File | Purpose | Verdict | Notes |
|------|---------|---------|-------|
| `src/btc_trader/domain/models.py` | Candle, Order, Position, Trade (Pydantic, frozen) | REUSE-AS-IS | Fully portable; no crypto assumptions |
| `src/btc_trader/domain/enums.py` | OrderSide, OrderType, OrderStatus | REUSE-AS-IS | Generic |
| `src/btc_trader/domain/exceptions.py` | Exception hierarchy | REUSE-AS-IS | Generic |
| `src/btc_trader/engine/types.py` | Signal, StrategyContext, Strategy ABC | REUSE-AS-IS | Core abstraction; excellent design |
| `src/btc_trader/engine/backtest.py` | BacktestEngine (candle replay, YAML config) | REUSE-AS-IS | No crypto-specific logic. For daily equities verify no intra-bar fill assumption. |
| `src/btc_trader/engine/execution.py` | ExecutionSimulator: fills at close + slippage/fees | ADAPT | Fills entire qty at candle.close; acceptable for daily-bar system with next-open execution |
| `src/btc_trader/engine/portfolio.py` | Portfolio state (cash, positions, PnL) | REUSE-AS-IS | Multi-symbol, handles fees, realized/unrealized. Gold standard. |
| `src/btc_trader/engine/performance.py` | Backtest metrics | REUSE-AS-IS | Generic |
| `src/btc_trader/engine/_market_regime.py` | Bull/bear/lateral regime (MA + ATR) | REUSE-AS-IS | Pure, works on any OHLCV |
| `src/btc_trader/engine/order_manager.py` | Idempotent order placement + reconciliation | REUSE-AS-IS | Best-in-class pattern; prevents double-submits |
| `src/btc_trader/engine/paper.py` | Paper-trading engine | REUSE-AS-IS | Generic polling loop |
| `src/btc_trader/engine/live.py` | Live engine (CCXT-specific parts) | ADAPT | OrderManager pattern is excellent; replace CCXT calls |
| `src/btc_trader/services/risk_manager.py` | ConfigurableRiskManager: position limits, drawdown halts | REUSE-AS-IS | Robust and reusable; copy wholesale |
| `src/btc_trader/services/advanced_risk.py` | TradePlan, ManagedTrade, RiskSessionState | ADAPT | Good structure; use selectively |
| `src/btc_trader/strategies/indicators.py` | SMA, EMA, RSI(Wilder), MACD, ATR, ADX, slope, true-range | **REUSE-AS-IS — TIER 1** | **No look-ahead bias. Correct seeding (returns None until ready). Port this first.** |
| `src/btc_trader/strategies/divergence.py` | Pivot detection, bullish/bearish divergence | REUSE-AS-IS | Correct pivot definitions; safe |
| `src/btc_trader/strategies/momentum.py` | Normalized momentum score [0,1] | REUSE-AS-IS | Tiny, useful |
| `src/btc_trader/strategies/ma_cross.py` | Golden/death cross + stop-loss + min-hold | REUSE-AS-IS | Clean; portable to equities |
| `src/btc_trader/strategies/trend_breakout.py` | N-bar breakout + ATR trailing stop | REUSE-AS-IS | Correct for daily bars |
| `src/btc_trader/strategies/mean_reversion_bands.py` | Bollinger-band mean reversion | REUSE-AS-IS | Well-coded |
| `src/btc_trader/strategies/lateral_market.py` | 5 lateral strategies (AdaptiveMR, RangeBound, DCA, Rebalance, Grid) | REUSE-AS-IS | Excellent for sideways periods |
| `src/btc_trader/strategies/multi_timeframe_trend.py` | Weekly regime filter + daily breakout | REUSE-AS-IS | Aggregation helper works on any OHLCV |
| `src/btc_trader/strategies/regime_scaled_ma_cross.py` | MA cross with soft rebalancing in lateral regimes | REUSE-AS-IS | Advanced; clean regime switching |
| `src/btc_trader/strategies/rsi_macd_div_momentum.py` | RSI + MACD + divergence | REUSE-AS-IS | Complex but correct |
| `src/btc_trader/strategies/ma_cross_*.py` variants | Chop guard, confirmed entry, profit trail, slope filter | REUSE-AS-IS | Well-crafted variants; no look-ahead |
| `src/btc_trader/strategies/sentiment_*.py` | Fear-Greed breakout/pullback/recovery | ADAPT | Crypto-specific sentiment source; lag design (1-day) is correct and portable |
| `src/btc_trader/strategies/sentiment_data.py` | FearGreedSeries with conservative lag | REUSE-AS-IS | `latest_available(lag_days=1)` prevents look-ahead. Port the lag-pattern. |
| `src/btc_trader/strategies/trend_wyckoff_*.py` | Wyckoff analysis | ADAPT | Research-heavy; skip unless explicitly needed |
| `src/btc_trader/strategies/registry.py` | Strategy factory | REUSE-AS-IS | Standard pattern |
| `src/btc_trader/adapters/historical_cache.py` | Decorator for caching fetched candles | REUSE-AS-IS | Excellent pattern for wrapping any MarketDataPort |
| `src/btc_trader/adapters/paper_broker.py` | Paper-trading broker simulator | REUSE-AS-IS | Generic; no crypto assumptions |
| `src/btc_trader/adapters/fakes/` | FakeMarketData, FixedClock, InMemoryPersistence | REUSE-AS-IS | **Port test doubles; invaluable for offline unit tests** |
| `src/btc_trader/adapters/exchange_ccxt/` | CCXT market data + broker adapters | **DISCARD** | Crypto-specific; replace with equity data provider |
| `src/btc_trader/ports/` | BrokerPort, ClockPort, MarketDataPort, PersistencePort, NotifierPort | REUSE-AS-IS | Clean protocols; implement for equities |
| `src/btc_trader/observability/` | Metrics, hooks, FastAPI runtime state | ADAPT | Good patterns; no crypto-specific logic |

**What to port from `A_Crypto_trading` — priority order for Phase 1:**

For **Phase 1 (data layer)** specifically, the relevant pieces are:
1. `adapters/historical_cache.py` — decorator pattern for caching. Adapt to wrap `DataProvider` interface.
2. `adapters/fakes/` — test double patterns. Use to build `FakeProvider` for offline cache tests.
3. `domain/exceptions.py` — exception hierarchy pattern.
4. Config YAML structure (not the code; just the pattern).

For **Phase 2+** (backtester, signals):
- `strategies/indicators.py` — SMA, EMA, RSI(Wilder), ATR: copy with credit
- `engine/portfolio.py`, `engine/execution.py`, `engine/backtest.py`
- `services/risk_manager.py`
- Strategy frameworks (ma_cross, trend_breakout, mean_reversion_bands)

---

## Repo 3 — `trading_bot`

### Per-file verdicts

| File | Purpose | Verdict | Critical risks |
|------|---------|---------|----------------|
| `main.py` | Pipeline orchestration + threading | DISCARD | Undefined `df` variable; thread race conditions on shared SQLite; unexecutable |
| `strategies/backtest.py` | Backtest engine | DISCARD | **CRITICAL**: undefined `df` runs at module-level; dummy sentiment hardcoded to 0.0 |
| `strategies/simple_strategy.py` | Rule-based BUY/SELL/HOLD | DISCARD | Logic is trivial and subsumed by `A_Crypto_trading` strategies |
| `analysis/technical_indicators.py` | RSI, SMA | DISCARD | **CRITICAL**: `.dropna()` on Close misaligns index; RSI/SMA values shift to wrong dates |
| `analysis/sentiment_analysis.py` | VADER + TextBlob | DISCARD | Out of scope for v1 (CLAUDE.md §2: no news/sentiment) |
| `data_fetchers/fetch_stocks.py` | yfinance wrapper | DISCARD | Superseded by Phase 1 `DataProvider` interface |
| `data_fetchers/fetch_ansa.py` | ANSA RSS parser | DISCARD | Out of scope |
| `data_fetchers/fetch_newsfilter.py` | NewsFilter API | DISCARD | Out of scope |
| `data_fetchers/fetch_fineco.py` | Web scraper | DISCARD | Fragile; out of scope |
| `data_fetchers/stream_news_socketio.py` | Real-time news stream | DISCARD | Undefined function at line 29; missing import; out of scope |
| `db_models/models.py` | SQLAlchemy ORM | DISCARD | Pattern superseded by Phase 1 SQLite schema |
| `degiro_order.py` | DEGIRO broker wrapper | DISCARD | Hardcoded credentials; no position sizing |

**Nothing to port from `trading_bot`.**

---

## Repo 4 — `trading-bot-starter`

### Per-file verdicts

| File | Purpose | Verdict | Critical risks |
|------|---------|---------|----------------|
| `app/data.py` | yfinance wrapper | DISCARD | `auto_adjust=True` rewrites history retroactively |
| `app/strategy.py` | MA-crossover signal | DISCARD | Logic correct but trivial; subsumed |
| `app/bot.py` | Event loop | DISCARD | Same-bar signal execution; no freshness check |
| `app/storage.py` | SQLite persistence | DISCARD | Schema is superseded by Phase 1 design, though the parameterized-SQL safety pattern is good |
| `app/broker.py` | Paper account + positions | DISCARD | Position math is correct; pattern subsumed by `A_Crypto_trading` portfolio.py |
| `app/backtest.py` | Backtest runner | DISCARD | **CRITICAL look-ahead**: feeds all history at each step including future bars |

**Nothing to port from `trading-bot-starter`.**

---

## Final porting plan

### Phase 1 (this phase) — data layer

| What | From | Action |
|------|------|--------|
| Cache decorator pattern | `A_Crypto_trading/adapters/historical_cache.py` | Adapt into `src/data/cache.py`; replace candle model with our schema |
| Test-double / fake-provider pattern | `A_Crypto_trading/adapters/fakes/` | Adapt into `tests/fakes/`; implement `FakeDataProvider` for offline tests |
| Exception hierarchy style | `A_Crypto_trading/domain/exceptions.py` | Reference; write our own `src/exceptions.py` |
| Config YAML style | `ats_mvp_starter/configs/base.yaml` | Reference only; follow Phase 1 plan's own YAML spec |

### Phase 2+ (backtester, signals)

| What | From | Action |
|------|------|--------|
| `indicators.py` — SMA, EMA, RSI(Wilder), ATR | `A_Crypto_trading/strategies/indicators.py` | Port with tests; commit message to name origin |
| Portfolio accounting | `A_Crypto_trading/engine/portfolio.py` | Adapt to equity model (no shorts in v1) |
| Risk manager | `A_Crypto_trading/services/risk_manager.py` | Adapt thresholds to match CLAUDE.md §4 |
| Execution simulator | `A_Crypto_trading/engine/execution.py` | Adapt: fill at next-open, not same-bar close |
| Market-hours scheduler | `ats_mvp_starter/src/ats_mvp/live_loop.py` | Phase 4 only |

---

## Anti-patterns to avoid (synthesised from all four repos)

1. **Same-bar signal + fill**: Signal from close of day t must execute at open of day t+1. Test explicitly.
2. **`auto_adjust=True`**: Always use `auto_adjust=False`; store both `close` and `adj_close` explicitly.
3. **`.ffill()` without timestamp check**: Never forward-fill sentiment or any feature without validating that the source timestamp predates the bar start.
4. **`.dropna()` on intermediate series**: This shifts indices and causes silent misalignment. Fail loudly or handle NaN explicitly.
5. **Module-level executable code**: All logic must be in functions; never run code at import time.
6. **Recomputing full backtest per live iteration**: Use a rolling window; don't replay from year 2005 every minute.
7. **Hardcoded credentials**: Use environment variables or GitHub Secrets.
8. **Unsynchronised threads sharing SQLite**: Use a single-writer pattern or WAL mode with explicit locking.
9. **Assuming intraday timeframe**: Port indicators with parameters tuned for daily bars (e.g., ATR period 14 days, not 14 hours).
10. **Missing tests for the no-look-ahead invariant**: Write explicit tests that adding a future candle to history does not change any earlier signal.
