"""Live trade journal: one CSV row per closed trade with everything needed to measure the edge
and find leaks (decision price, spread, slippage, best and worst move while open, exit reason,
signal strength at entry). Append-only; never raises into the trading code."""
from __future__ import annotations

import csv
import datetime as dt
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

FIELDS = ["symbol", "side", "source", "entry_time", "exit_time", "hold_min", "qty",
          "decision_price", "bid", "ask", "spread_bps", "entry_fill", "slippage_bps",
          "exit_fill", "pnl", "pnl_bps", "notional", "mfe_bps", "mae_bps", "reason",
          "tp", "sl", "sma_gap_bps", "rsi", "version", "strategy", "config_id"]

REASON_MAP = {
    "trend reversal": "signal", "watchdog stop": "watchdog", "manual close": "manual",
    "partial fill without protection": "partial-fill", "end-of-day flatten": "eod",
    "manual panic": "panic", "take profit": "target", "software stop": "stop", "carried overnight": "carried",
}


def _now_str(ts: Optional[float] = None) -> str:
    return dt.datetime.fromtimestamp(ts or time.time()).strftime("%Y-%m-%d %H:%M")


class TradeJournal:
    def __init__(self, path: str, version: str, strategy: str = "", config_id: str = ""):
        self.path = Path(path)
        self.version = version
        self.strategy = strategy
        self.config_id = config_id
        self.lock = threading.Lock()
        self._upgrade_header()
        self.pending: Dict[str, Dict[str, Any]] = {}   # submitted entries, waiting for a fill
        self.open: Dict[str, Dict[str, Any]] = {}      # filled positions being tracked
        self.reasons: Dict[str, Tuple[str, float]] = {}

    def _upgrade_header(self) -> None:
        """A journal written by an older version has fewer columns: rewrite it once with the
        current header (old rows get blank new columns) so new rows line up."""
        try:
            if not self.path.exists():
                return
            with self.path.open(newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames == FIELDS:
                    return
                rows = list(reader)
            tmp = self.path.with_suffix(".csv.tmp")
            with tmp.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in FIELDS})
            os.replace(tmp, self.path)
        except Exception:
            pass  # never stop the bot over the journal

    # -------- entry side --------
    def note_entry(self, symbol: str, is_long: bool, source: str, bid: float, ask: float,
                   limit: float, tp: float, sl: float, signal: Optional[Dict[str, Any]]) -> None:
        mid = (bid + ask) / 2 if bid and ask else 0.0
        fast, slow = (signal or {}).get("fast"), (signal or {}).get("slow")
        with self.lock:
            self.pending[symbol] = {
                "symbol": symbol, "side": "long" if is_long else "short", "source": source,
                "decision_price": (signal or {}).get("price") or (ask if is_long else bid),
                "bid": bid, "ask": ask, "spread_bps": (ask - bid) / mid * 1e4 if mid else None,
                "tp": tp, "sl": sl, "rsi": (signal or {}).get("rsi"),
                "sma_gap_bps": (fast - slow) / slow * 1e4 * (1 if is_long else -1) if fast and slow else None,
                "submitted": time.time(),
            }

    def note_fill(self, symbol: str, position: Any) -> None:
        try:
            qty = float(position.qty)
            fill = float(position.avg_entry_price)
        except Exception:
            return
        with self.lock:
            rec = self.pending.pop(symbol, None) or {"symbol": symbol, "side": "long" if qty > 0 else "short",
                                                     "source": "carried"}
            rec.update(entry_ts=time.time(), qty=abs(qty), entry_fill=fill, hi=fill, lo=fill)
            dp = rec.get("decision_price")
            if dp:
                sign = 1 if rec["side"] == "long" else -1
                rec["slippage_bps"] = (fill / dp - 1) * 1e4 * sign   # positive = paid more than the signal price
            self.open[symbol] = rec

    def mark(self, symbol: str, price: float, qty: Optional[float] = None) -> None:
        """Called every watchdog cycle: tracks the best/worst price while open, and the size
        (a partly filled entry can grow after the first snapshot)."""
        with self.lock:
            rec = self.open.get(symbol)
            if rec and price > 0:
                rec["hi"] = max(rec["hi"], price)
                rec["lo"] = min(rec["lo"], price)
                if qty:
                    rec["qty"] = qty

    def note_exit_reason(self, symbol: str, reason: str, source: str = "auto") -> None:
        label = "manual" if source == "manual" and reason == "manual close" else REASON_MAP.get(reason, reason)
        if reason.startswith("circuit breaker") or "breaker" in reason:
            label = "breaker"
        with self.lock:
            self.reasons[symbol] = (label, time.time())

    # -------- exit side --------
    def close(self, symbol: str, exit_fill: Optional[float], fallback_price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        with self.lock:
            rec = self.open.pop(symbol, None)
            reason, when = self.reasons.pop(symbol, (None, 0.0))
        if rec is None:
            return None
        px = exit_fill or fallback_price or rec["entry_fill"]
        long_ = rec["side"] == "long"
        sign = 1 if long_ else -1
        entry = rec["entry_fill"]
        if reason is None or time.time() - when > 600:
            tp, sl = rec.get("tp"), rec.get("sl")
            if tp and ((long_ and px >= tp * 0.999) or (not long_ and px <= tp * 1.001)):
                reason = "target"
            elif sl and ((long_ and px <= sl * 1.001) or (not long_ and px >= sl * 0.999)):
                reason = "stop"
            else:
                reason = "broker/other"
        hi, lo = max(rec["hi"], px), min(rec["lo"], px)
        best, worst = (hi, lo) if long_ else (lo, hi)
        pnl = (px - entry) * rec["qty"] * sign
        row = {
            **{k: rec.get(k) for k in ("symbol", "side", "source", "qty", "decision_price", "bid", "ask",
                                       "spread_bps", "entry_fill", "slippage_bps", "tp", "sl", "sma_gap_bps", "rsi")},
            "entry_time": _now_str(rec["entry_ts"]), "exit_time": _now_str(),
            "hold_min": (time.time() - rec["entry_ts"]) / 60, "exit_fill": px,
            "pnl": pnl, "notional": entry * rec["qty"], "pnl_bps": (px / entry - 1) * 1e4 * sign,
            "mfe_bps": (best / entry - 1) * 1e4 * sign, "mae_bps": (worst / entry - 1) * 1e4 * sign,
            "reason": reason, "version": self.version,
            "strategy": self.strategy, "config_id": self.config_id,
        }
        self._append(row)
        return row

    def _append(self, row: Dict[str, Any]) -> None:
        new = not self.path.exists()
        with self.lock, self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})

    def read(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.lock, self.path.open(newline="") as f:
            return list(csv.DictReader(f))
