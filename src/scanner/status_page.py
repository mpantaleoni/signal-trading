"""Generate docs/index.html + docs/status.json from a ScanReport.

The HTML page is mobile-first, dark-themed, no external dependencies.
Equity curve is rendered as an inline SVG polyline.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .run import ScanReport

DOCS_DIR = Path("docs")


def generate_status_page(report, state, output_dir: Path = DOCS_DIR) -> None:
    """Write index.html and status.json to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── status.json ──────────────────────────────────────────────────────────
    status_json = {
        "as_of_date": str(report.as_of_date),
        "equity": report.equity,
        "cash": report.cash,
        "initial_capital": report.initial_capital,
        "peak_equity": report.peak_equity,
        "drawdown_pct": round(report.drawdown_pct * 100, 2),
        "drawdown_status": report.drawdown_status,
        "engine": report.engine,
        "open_positions": report.open_positions,
        "pending_orders": report.pending_orders,
        "fills_today": [
            {
                "ticker": f.ticker, "side": f.side, "shares": f.shares,
                "price": f.price, "engine": f.engine, "reason": f.reason,
                "pnl_usd": f.pnl_usd,
            }
            for f in report.fills_today
        ],
        "trades_total": report.trades_total,
        "equity_history": state.equity_history[-60:],  # last 60 sessions
    }
    (output_dir / "status.json").write_text(
        json.dumps(status_json, indent=2, default=str), encoding="utf-8"
    )

    # ── HTML ─────────────────────────────────────────────────────────────────
    html = _render_html(report, state)
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def _sparkline_svg(history: list, width: int = 320, height: int = 60) -> str:
    """Render an equity curve as an SVG polyline."""
    if len(history) < 2:
        return ""
    values = [float(v) for _, v in history]
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0

    def x(i: int) -> float:
        return i / (len(values) - 1) * width

    def y(v: float) -> float:
        return height - ((v - lo) / span) * (height - 4) - 2

    points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    color = "#3fb950" if values[-1] >= values[0] else "#f85149"
    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;height:{height}px">'
        f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/>'
        f"</svg>"
    )


def _fmt_usd(v: float) -> str:
    return f"${v:,.0f}"


def _fmt_pct(v: float, signed: bool = False) -> str:
    sign = "+" if signed and v >= 0 else ""
    return f"{sign}{v:.1f}%"


def _positions_table(positions: list[dict]) -> str:
    if not positions:
        return "<p style='color:var(--muted)'>No open positions.</p>"
    rows = []
    for p in sorted(positions, key=lambda x: x["engine"]):
        pct = p["unrealized_pct"]
        cls = "green" if pct >= 0 else "red"
        sign = "+" if pct >= 0 else ""
        rows.append(
            f"<tr>"
            f"<td><b>{p['ticker']}</b></td>"
            f"<td><span class='badge badge-{p['engine'].lower()}'>{p['engine']}</span></td>"
            f"<td>{p['shares']}</td>"
            f"<td class='{cls}'>{sign}{pct:.1f}%</td>"
            f"<td>{p['sessions_held']}d</td>"
            f"</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Ticker</th><th>Eng</th><th>Shares</th><th>P&L</th><th>Held</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _orders_table(orders: list[dict]) -> str:
    if not orders:
        return "<p style='color:var(--muted)'>No orders queued.</p>"
    rows = []
    for o in orders:
        icon = "🟢" if o["side"] == "buy" else "🔴"
        rows.append(
            f"<tr>"
            f"<td>{icon} {o['side'].upper()}</td>"
            f"<td><b>{o['ticker']}</b></td>"
            f"<td>{o['shares']}</td>"
            f"<td><span class='badge badge-{o['engine'].lower()}'>{o['engine']}</span></td>"
            f"</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Side</th><th>Ticker</th><th>Shares</th><th>Eng</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _fills_table(fills) -> str:
    if not fills:
        return "<p style='color:var(--muted)'>No fills today.</p>"
    rows = []
    for f in fills:
        icon = "✅" if f.side == "buy" else "❌"
        pnl_str = ""
        if f.pnl_usd is not None:
            cls = "green" if f.pnl_usd >= 0 else "red"
            sign = "+" if f.pnl_usd >= 0 else ""
            pnl_str = f"<span class='{cls}'>{sign}${f.pnl_usd:,.0f}</span>"
        rows.append(
            f"<tr>"
            f"<td>{icon} {f.side.upper()}</td>"
            f"<td><b>{f.ticker}</b></td>"
            f"<td>{f.shares}</td>"
            f"<td>${f.price:.2f}</td>"
            f"<td>{pnl_str}</td>"
            f"</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Side</th><th>Ticker</th><th>Shares</th><th>Price</th><th>P&L</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _render_html(report, state) -> str:
    total_ret = (report.equity / report.initial_capital - 1) * 100
    ret_cls = "green" if total_ret >= 0 else "red"
    dd_cls = {"ok": "green", "warning": "yellow", "stopped": "red"}.get(report.drawdown_status, "muted")
    dd_icon = {"ok": "✅", "warning": "⚠️", "stopped": "🛑"}.get(report.drawdown_status, "")

    spark = _sparkline_svg(state.equity_history[-60:])
    spark_html = f'<div class="card">{spark}</div>' if spark else ""

    stopped_banner = ""
    if report.drawdown_status == "stopped":
        stopped_banner = '<div class="banner-stopped">🛑 SYSTEM STOPPED — drawdown limit hit. No new entries.</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <title>Signal Scanner</title>
  <style>
    :root{{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#e6edf3;--muted:#8b949e;
          --green:#3fb950;--red:#f85149;--yellow:#d29922;--blue:#58a6ff;--purple:#c792ea}}
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         font-size:14px;padding:12px;max-width:480px;margin:0 auto}}
    .updated{{color:var(--muted);font-size:11px;text-align:right;margin-bottom:10px}}
    .banner-stopped{{background:#3d0c0c;border:1px solid var(--red);border-radius:8px;
                     padding:10px 12px;margin-bottom:12px;color:var(--red);font-weight:600}}
    .grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}}
    .metric{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px}}
    .metric .label{{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.5px}}
    .metric .value{{font-size:18px;font-weight:600;margin-top:4px}}
    .card{{background:var(--card);border:1px solid var(--border);border-radius:8px;
           padding:12px;margin-bottom:12px}}
    h2{{font-size:15px;margin:16px 0 8px;color:var(--blue)}}
    table{{width:100%;border-collapse:collapse;font-size:13px}}
    th{{text-align:left;color:var(--muted);font-weight:normal;padding:5px 3px;border-bottom:1px solid var(--border)}}
    td{{padding:7px 3px;border-bottom:1px solid var(--border)}}
    tr:last-child td{{border-bottom:none}}
    .badge{{display:inline-block;padding:1px 5px;border-radius:3px;font-size:10px;font-weight:700}}
    .badge-a{{background:#1f3a5f;color:var(--blue)}}
    .badge-b{{background:#2d1f3a;color:var(--purple)}}
    .green{{color:var(--green)}}.red{{color:var(--red)}}.yellow{{color:var(--yellow)}}.muted{{color:var(--muted)}}
    p{{color:var(--muted);font-size:13px;padding:8px 0}}
  </style>
</head>
<body>
  <div class="updated">Updated {report.as_of_date} &nbsp;·&nbsp; Engine {report.engine.upper()}</div>
  {stopped_banner}

  <div class="grid">
    <div class="metric">
      <div class="label">Equity</div>
      <div class="value">{_fmt_usd(report.equity)}</div>
    </div>
    <div class="metric">
      <div class="label">Return</div>
      <div class="value {ret_cls}">{_fmt_pct(total_ret, signed=True)}</div>
    </div>
    <div class="metric">
      <div class="label">Drawdown</div>
      <div class="value {dd_cls}">{_fmt_pct(report.drawdown_pct * 100)} {dd_icon}</div>
    </div>
    <div class="metric">
      <div class="label">Cash</div>
      <div class="value">{_fmt_usd(report.cash)}</div>
    </div>
  </div>

  {spark_html}

  <h2>Positions ({len(report.open_positions)})</h2>
  <div class="card">{_positions_table(report.open_positions)}</div>

  <h2>Orders for tomorrow ({len(report.pending_orders)})</h2>
  <div class="card">{_orders_table(report.pending_orders)}</div>

  <h2>Fills today ({len(report.fills_today)})</h2>
  <div class="card">{_fills_table(report.fills_today)}</div>

</body>
</html>"""
