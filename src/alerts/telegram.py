"""Telegram alert sender and message formatter.

Secrets come from environment variables:
  TELEGRAM_BOT_TOKEN — bot token from @BotFather
  TELEGRAM_CHAT_ID   — your personal chat ID

Usage:
  alert = TelegramAlert.from_env()
  alert.send(format_report(scan_report))
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


@dataclass
class TelegramAlert:
    token: str
    chat_id: str

    @classmethod
    def from_env(cls) -> Optional["TelegramAlert"]:
        """Return None if env vars are not set (allows dry-run without crashing)."""
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            return None
        return cls(token=token, chat_id=chat_id)

    def send(self, text: str) -> bool:
        """Send a message. Returns True on success."""
        url = _TELEGRAM_API.format(token=self.token)
        try:
            resp = requests.post(
                url,
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=15,
            )
            if not resp.ok:
                logger.warning("Telegram error %s: %s", resp.status_code, resp.text[:200])
            return resp.ok
        except requests.RequestException as exc:
            logger.warning("Telegram send failed: %s", exc)
            return False

    def send_failure(self, error: str) -> None:
        """Send a failure alert (best-effort, does not raise)."""
        try:
            self.send(f"🚨 <b>Scanner failed</b>\n\n<code>{error[:500]}</code>")
        except Exception:
            pass


# ── message formatting ────────────────────────────────────────────────────────


def format_report(report) -> str:
    """Format a ScanReport into an HTML Telegram message.

    *report* is a ScanReport dataclass from src.scanner.run.
    """
    from datetime import date as _date

    lines: list[str] = []

    # Header
    dd_icon = {"ok": "✅", "warning": "⚠️", "stopped": "🛑"}.get(report.drawdown_status, "❓")
    lines.append(f"📊 <b>Signal Scanner</b> — {report.as_of_date}  [{report.engine.upper()}]")
    lines.append("")

    # Portfolio summary
    total_return_pct = (report.equity / report.initial_capital - 1) * 100
    ret_sign = "+" if total_return_pct >= 0 else ""
    lines.append("💰 <b>Portfolio</b>")
    lines.append(f"  Equity:   <b>${report.equity:,.0f}</b>  ({ret_sign}{total_return_pct:.1f}%)")
    lines.append(f"  Cash:     ${report.cash:,.0f}")
    lines.append(f"  Drawdown: {report.drawdown_pct * 100:.1f}%  {dd_icon}")
    if report.drawdown_status == "stopped":
        lines.append("  <b>STOPPED — no new entries until new equity peak</b>")
    lines.append("")

    # Fills today
    if report.fills_today:
        lines.append(f"📋 <b>Fills today ({len(report.fills_today)})</b>")
        for f in report.fills_today:
            icon = "✅" if f.side == "buy" else "❌"
            pnl_str = ""
            if f.pnl_usd is not None:
                pnl_sign = "+" if f.pnl_usd >= 0 else ""
                pnl_str = f"  P&L: {pnl_sign}${f.pnl_usd:,.0f}"
            lines.append(f"  {icon} {f.side.upper():4s} {f.ticker} ×{f.shares} @ ${f.price:.2f}  [{f.engine}]{pnl_str}")
        lines.append("")

    # Open positions
    if report.open_positions:
        lines.append(f"📌 <b>Positions ({len(report.open_positions)})</b>")
        for p in sorted(report.open_positions, key=lambda x: x["engine"]):
            pnl = p["unrealized_pnl"]
            pct = p["unrealized_pct"]
            sign = "+" if pnl >= 0 else ""
            lines.append(
                f"  {p['ticker']:6s}  ×{p['shares']:4d}  {sign}${pnl:,.0f} ({sign}{pct:.1f}%)"
                f"  [{p['engine']}, {p['sessions_held']}d]"
            )
        lines.append("")

    # Pending orders
    if report.pending_orders:
        lines.append(f"🔔 <b>Orders for tomorrow ({len(report.pending_orders)})</b>")
        for o in report.pending_orders:
            icon = "🟢" if o["side"] == "buy" else "🔴"
            lines.append(f"  {icon} {o['side'].upper():4s} {o['ticker']} ×{o['shares']}  [{o['engine']}]")
        lines.append("")
    else:
        lines.append("🔔 <b>No orders queued for tomorrow</b>")
        lines.append("")

    # Footer
    lines.append(f"<i>Trades all-time: {report.trades_total}</i>")

    return "\n".join(lines)
