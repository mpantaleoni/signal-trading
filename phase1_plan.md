# Phase 1 Plan — Data layer, cache, universe membership, integrity checks

Status: PROPOSED, awaiting owner approval. No code written yet.
Drafted: 2026-10-14 (Windows session). Work continues on the Mac.

## 0. Handoff notes for the next session (read first)

1. Read `CLAUDE.md` fully, then this file.
2. **Assess the old ATS-MVP repo first** (read-only, time-boxed). Classify each module as
   reuse as-is / adapt / discard, and flag look-ahead or data-handling risks
   (forward-filled gaps, adjusted-price handling, timezone shifts, same-bar signals).
   Write the verdict to `docs/ats_mvp_assessment.md`.
   - Agreed approach: **fresh repo, selective porting**. Do not fork the old repo as a base.
     Each ported piece gets its own tests and a commit message naming its origin.
   - Revisit only if the old repo turns out to be very close to this spec.
3. Get the owner's answers to the decisions in section 7 (or "approved, defaults").
4. Check the environment: git installed, Python 3.11–3.13 (3.12 recommended), project `.venv`.
   (On the Windows machine git was missing and Python was 3.15 with no packages; pyarrow
   wheels for 3.15 may not exist.)
5. Then implement, committing in logical steps:
   scaffold+config → providers → cache → universe → integrity → report+CLI → tests/fixes.

Acceptance criteria (from owner):
1. CLI downloads/updates daily OHLCV for the universe (2005-01-01 to today) plus ETF list and
   SPY, using the cache.
2. A function returns S&P 500 members on any given date.
3. Markdown data report per year: universe size, members with data, members missing
   (missing_coverage), integrity failures.
4. All tests pass, incl. point-in-time membership and cache-not-re-downloading tests.
5. End with the session report per CLAUDE.md section 9.

Scope guard: Phase 1 only. No strategy, signal or backtest code.

## 1. Module / file structure

```
config/
  data.yaml            # date range, cache dir, provider choice, check thresholds
  universe.yaml        # ETF list, constituent source + checksum, ticker aliases
src/
  config.py            # YAML -> typed dataclasses
  cli.py               # argparse entry: python -m src.cli ...
  data/
    provider.py        # DataProvider protocol + canonical bar schema
    yfinance_provider.py
    stooq_provider.py  # stdlib urllib, no extra deps
    cache.py           # Parquet store + SQLite metadata, incremental logic
    calendar.py        # trading calendar derived from SPY bars (no extra dep)
    integrity.py       # checks -> IntegrityFailure records
    crosscheck.py      # yfinance vs Stooq close comparison on a sample
  universe/
    sp500.py           # load change history, build intervals, members_on(date)
    etfs.py            # ETF list from config
    symbols.py         # ticker normalisation (BRK.B -> BRK-B / brk-b.us) + aliases
  reports/
    data_report.py     # Markdown data report
data/                  # git-ignored: parquet/, meta.sqlite, raw/
reports/               # generated reports
tests/
```

CLI: `python -m src.cli data update [--start 2005-01-01] [--end today]`, `data report`,
`data crosscheck`, `universe members --date YYYY-MM-DD`.

## 2. DataProvider interface

```python
class DataProvider(Protocol):
    name: str
    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame: ...
```

- Canonical frame: tz-naive `DatetimeIndex` (session dates), columns
  `open, high, low, close, adj_close, volume, dividends, splits`.
- Empty frame = provider has no data. Network errors raise (never confused with "no data").
- yfinance: `auto_adjust=False, actions=True` (keeps close and adj_close). Drop today's bar if
  the US session hasn't closed, so partial bars never enter the cache.
- Stooq: CSV via `https://stooq.com/q/d/l/?s=<sym>.us&i=d`. Adjusted OHLCV only, no actions,
  so used for price-level cross-check only. RISK: Stooq may have started gating automated
  CSV downloads. Verify first; if blocked, report it and leave the cross-check as a stub
  rather than scrape around it.
- Provider choice via `data.yaml` (`provider: yfinance`), so a paid source can be swapped in
  through configuration only.

## 3. Cache design

- Parquet per symbol per provider: `data/parquet/<provider>/<SYMBOL>.parquet`.
- SQLite `symbols` table: `symbol, provider, first_date, last_date, rows, fetched_at,
  status (ok|no_data|error), adj_fingerprint, retry_after`.
- SQLite `integrity_failures`: `symbol, date, check, detail, run_id`. SQLite `runs`: run log.

Incremental update per symbol:
1. If `last_date >= last completed session`, make no network call.
2. Else fetch from `last_date - 10 sessions` to end (overlap window).
3. If overlap adj_close differs from cache beyond tolerance (new dividend/split rewrote the
   adjusted history), do a full re-download of that symbol and log why.
4. Else append and rewrite atomically (temp file + replace).

- `no_data` symbols (typically delisted) get a 30-day `retry_after`, so dead tickers aren't
  re-requested every run.
- Metadata/Parquet mismatch (row count, last date) or unreadable file means the entry is
  invalid; refetch and log.
- Throttling + retry with backoff (yfinance rate-limits bulk pulls; first run ~1,100 symbols).

## 4. S&P 500 constituent source and point-in-time membership

- Primary: GitHub repo `fja05680/sp500`, file "S&P 500 Historical Components & Changes"
  (CSV, one row per change date with the full ticker list on that date, back to 1996).
  Vendored into `data/raw/` with the commit and SHA-256 recorded in `universe.yaml`.
- Cross-check: Wikipedia "List of S&P 500 companies" changes table. Informational only;
  needs `lxml` or a hand-written parser (see decision D4).
- Reconstruction: snapshots become per-ticker `[start, end)` intervals.
  `members_on(d)` returns the latest snapshot with date <= d, so it never uses a later change
  (tested). A date before the first snapshot raises an error.
  Helpers: `membership_intervals()`, `ever_members(start, end)`.
- Ticker issues (renames like FB->META, reused tickers): hand-maintained alias map in
  `universe.yaml` maps historical ticker to data-source symbol. Unmapped cases go to
  `missing_coverage`, never guessed.

## 5. Integrity checks and data report

Checks (each records failures; nothing is changed or filled):
- Missing bars: SPY-calendar sessions inside the symbol's data range with no bar.
- Zero, negative or NaN prices.
- Adjustment jumps: raw close-to-close move that looks like a split ratio (x2, x3, x1/2...)
  with no recorded split; adjusted return above 40% in absolute size (configurable).
- Stale prices: 5 or more identical closes in a row (configurable).
- Proposed extras (need approval, rule 3): OHLC consistency, zero-volume days.

Report `reports/data_report.md`:
- Header: run date, provider, constituent-list commit, and the KNOWN BIAS statement verbatim.
- Per-year table: year, distinct members during year, with data, partial, missing,
  missing share, integrity failures by type.
  "With data" = bars on at least 90% of in-index sessions that year; "partial" = some bars
  but below that; "missing" = no bars.
- Lists: missing members with membership dates; top 20 symbols by integrity failures.
- ETF section: coverage + failures. Stooq cross-check summary.
- Data-quality information only; no returns or performance figures.

## 6. Tests

Unit tests offline. Network tests marked `@pytest.mark.network`, skipped by default.
- Membership: synthetic change history (before/on/after a change); before-first-snapshot
  raises; snapshots -> intervals -> snapshots round-trip; aliases; no-look-ahead (adding a
  future change leaves earlier `members_on` unchanged); real-file sanity (495-510 members on
  sample dates, a few known pre-2019 adds/removals).
- Cache (fake provider counting calls): second request for a cached range makes zero calls;
  newer end date fetches only tail + overlap; overlap mismatch triggers full refetch;
  corrupt Parquet / metadata mismatch triggers refetch; `no_data` respected until
  `retry_after`; today's partial bar never stored.
- Integrity: passing and failing synthetic series for each check.
- Providers: canonical-schema conversion on recorded sample responses.
- Report: built from a fixture cache; per-year counts match hand-computed values; bias
  statement present.
- CLI smoke test with the fake provider.

## 7. Decisions needed from the owner (defaults in brackets)

- D1 Price columns: store close, adj_close, dividends, splits as delivered; later phases
  build fully adjusted OHLC via `adj_close / close`. [yes]
- D2 Hold-out data: Phase 1 downloads and integrity-checks 2019+ data; report shows
  data-quality counts only, no backtests or returns. [OK]
- D3 Liquidity filter: threshold still open; defer to Phase 2. [defer]
- D4 Dependencies: approve `yfinance` (implied by spec). No `requests`/`lxml`; Stooq via
  `urllib`; Wikipedia cross-check skipped. [yes]
- D5 Extra checks: OHLC consistency + zero volume, flagged as a spec deviation. [yes]
- D6 "With data" threshold: 90% of in-index sessions per year. [90%]
- D7 Download scope: every ticker in the index at any point 2005-today (~1,100). [yes]
