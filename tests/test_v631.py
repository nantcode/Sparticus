"""V6.3.1 tests: the trade review tool and the hold-longer lab experiments. Offline, no data needed.

    python3 -m unittest discover -s tests -v
"""
import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import backtest as bt  # noqa: E402
import trade_review as tr  # noqa: E402
from sparticus_strategy import LIVE_DEFAULTS, config_id  # noqa: E402


# Every test runs in a throwaway folder, never in your bot folder: a test that creates the engine
# would otherwise open (and could rewrite) your real sparticus_journal.csv and log files.
_SAFE_CWD = []


def setUpModule():
    _SAFE_CWD.append((os.getcwd(), tempfile.TemporaryDirectory()))
    os.chdir(_SAFE_CWD[-1][1].name)


def tearDownModule():
    cwd, tmp = _SAFE_CWD.pop()
    os.chdir(cwd)
    tmp.cleanup()


def line(sym, otype, side, qty, filled, px, status, sub, fill):
    return f"{sym}\n{otype}\t{side}\t{qty}\t{filled}\t{px}\t{status}\taccess_key\t{sub}\t{fill}\tOct 09, 2026, 04:00:00 PM\t\n"


D = "Oct 09, 2026, "
# Newest first, like the dashboard.
PASTE = "\tAsset\tOrder Type\n" + "".join([
    line("KO", "Market", "sell", "56.00", "56.00", "88.90", "filled", D + "03:55:10 PM", D + "03:55:12 PM"),   # flatten
    line("NVDA", "Stop @ $225.11", "sell", "21.00", "21.00", "225.10", "filled", D + "01:01:09 PM", D + "01:40:00 PM"),
    line("NVDA", "Limit @ $238.89", "sell", "21.00", "0.00", "-", "canceled", D + "01:01:09 PM", "-"),
    line("NVDA", "Limit @ $229.70", "buy", "21.00", "21.00", "229.70", "filled", D + "01:01:09 PM", D + "01:01:10 PM"),
    line("AAPL", "Limit @ $340.00", "sell", "0.5", "0.5", "340.00", "filled", D + "12:00:00 PM", D + "12:30:00 PM"),  # target, fractional
    line("AAPL", "Limit @ $330.00", "buy", "0.5", "0.5", "329.90", "filled", D + "11:00:00 AM", D + "11:00:01 AM"),
    line("KO", "Limit @ $88.40", "buy", "56.00", "56.00", "88.40", "filled", D + "10:03:08 AM", D + "10:03:12 AM"),
    line("SPY", "Market", "sell", "6.00", "6.00", "778.00", "filled", D + "10:30:00 AM", D + "10:30:01 AM"),  # signal
    line("SPY", "Limit @ $780.00", "buy", "6.00", "6.00", "779.00", "filled", D + "10:10:00 AM", D + "10:10:01 AM"),
    line("COST", "Market", "sell", "5.00", "5.00", "947.01", "filled", D + "09:40:00 AM", D + "09:40:01 AM"),  # orphan
    line("TSLA", "Limit @ $400.00", "buy", "2.00", "2.00", "400.00", "filled", D + "03:50:00 PM", D + "03:50:01 PM"),  # left open
]) + "Page Size:\n\n500\n"


class TestTradeReview(unittest.TestCase):
    def setUp(self):
        self.orders = tr.parse_paste(PASTE)
        self.trades, self.orphans, self.open = tr.round_trips(self.orders)
        self.by = {t["sym"]: t for t in self.trades}

    def test_reads_every_order(self):
        self.assertEqual(len(self.orders), 11)
        self.assertEqual(sum(o["status"] == "filled" for o in self.orders), 10)

    def test_round_trips_and_exit_types(self):
        self.assertEqual(sorted(self.by), ["AAPL", "KO", "NVDA", "SPY"])
        self.assertEqual(self.by["KO"]["reason"], "eod")
        self.assertEqual(self.by["NVDA"]["reason"], "stop")
        self.assertEqual(self.by["AAPL"]["reason"], "target")
        self.assertEqual(self.by["SPY"]["reason"], "signal")

    def test_pnl_hold_and_fractional_shares(self):
        self.assertAlmostEqual(self.by["KO"]["pnl"], (88.90 - 88.40) * 56)
        self.assertAlmostEqual(self.by["AAPL"]["pnl"], (340.00 - 329.90) * 0.5)
        self.assertAlmostEqual(self.by["SPY"]["hold_min"], 20.0)
        self.assertEqual(self.by["AAPL"]["entry_limit"], 330.00)

    def test_history_edges_are_reported_not_guessed(self):
        self.assertEqual([o["sym"] for o in self.orphans], ["COST"])
        self.assertEqual(self.open, {"TSLA": 2.0})
        text = tr.report(self.trades, self.orphans, self.open, "orders.txt")
        self.assertIn("Still open at the end", text)
        self.assertIn("COST 5", text)

    def test_report_flags_brackets_that_never_fill(self):
        signal_only = [t for t in self.trades if t["reason"] in ("signal", "eod")]
        self.assertIn("Not one bracket filled", tr.report(signal_only, [], {}, "x"))
        self.assertNotIn("Not one bracket filled", tr.report(self.trades, [], {}, "x"))

    def test_verdicts(self):
        lose = [{"pnl": -3.0 + (i % 3), "notional": 5000.0, "hold_min": 10.0} for i in range(60)]
        self.assertIn("not bad luck", tr.verdict(tr.stats(lose)))
        self.assertIn("too few", tr.verdict(tr.stats(lose[:10])))
        flat = [{"pnl": (-1) ** i * 5.0, "notional": 5000.0, "hold_min": 10.0} for i in range(60)]
        self.assertIn("no edge visible", tr.verdict(tr.stats(flat)))

    def test_csv_export_works_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "orders.csv"
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Asset", "Order Type", "Side", "Qty", "Filled Qty", "Avg. Fill Price", "Status",
                            "Submitted At", "Filled At"])
                w.writerow(["SPY", "Market", "sell", "6", "6", "778.00", "filled", "2026-10-09 10:30:00", "2026-10-09 10:30:01"])
                w.writerow(["SPY", "Limit @ $780.00", "buy", "6", "6", "779.00", "filled", "2026-10-09 10:10:00", "2026-10-09 10:10:01"])
            trades, _, _ = tr.round_trips(tr.load_orders(path))
        self.assertEqual(len(trades), 1)
        self.assertAlmostEqual(trades[0]["pnl"], -6.0)

    def test_main_writes_the_report_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                Path("orders.txt").write_text(PASTE)
                import contextlib
                import io
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(tr.main(["orders.txt"]), 0)
                self.assertTrue(Path("reviews/review_2026-10-09_2026-10-09.md").exists())
                self.assertTrue(Path("reviews/trades_2026-10-09_2026-10-09.csv").exists())
            finally:
                os.chdir(cwd)


class TestHoldExperiments(unittest.TestCase):
    def test_registered_with_one_change_each(self):
        p, _ = bt.EXPERIMENTS["live_hold"]
        self.assertFalse(p.signal_exit)
        self.assertEqual(p.min_hold_minutes, LIVE_DEFAULTS.min_hold_minutes)
        q, _ = bt.EXPERIMENTS["hold30"]
        self.assertEqual(q.signal_exit, LIVE_DEFAULTS.signal_exit)
        self.assertEqual(q.min_hold_minutes, 30.0)
        if LIVE_DEFAULTS.signal_exit:
            self.assertNotEqual(config_id(p), config_id(LIVE_DEFAULTS))

    def test_live_bot_can_trade_them(self):
        # both only change settings the live bot already honours (no lab-only switches)
        for name in ("live_hold", "hold30"):
            p, _ = bt.EXPERIMENTS[name]
            self.assertEqual((p.strategy, p.bar_minutes, p.market_filter), ("sma_cross", 1, "none"))


if __name__ == "__main__":
    unittest.main()
