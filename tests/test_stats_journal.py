"""Tests for the statistics pack, the trade journal, and the live bot's journal hooks.

    python3 -m unittest discover -s tests -v
"""
import importlib.util
import os
import random
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparticus_stats as S  # noqa: E402
from sparticus_journal import TradeJournal  # noqa: E402


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


def trade(pnl, day="2026-10-01", hour="10", reason="signal", notional=1000.0, **kw):
    return {"pnl": pnl, "notional": notional, "exit_time": f"{day} {hour}:30", "entry_time": f"{day} {hour}:10",
            "reason": reason, **kw}


class TestStats(unittest.TestCase):
    def test_edge_numbers(self):
        s = S.summarize([trade(10), trade(-5), trade(-5), trade(20)])
        self.assertEqual(s["trades"], 4)
        self.assertAlmostEqual(s["win_rate"], 0.5)
        self.assertAlmostEqual(s["payoff"], 3.0)
        self.assertAlmostEqual(s["breakeven_win_rate"], 0.25)
        self.assertAlmostEqual(s["profit_factor"], 3.0)
        self.assertAlmostEqual(s["expectancy"], 5.0)
        self.assertAlmostEqual(s["expectancy_bps"], 50.0)
        self.assertEqual(s["verdict"], "too few trades to judge")

    def test_verdicts_follow_the_t_stat(self):
        rng = random.Random(3)
        real = [trade(rng.gauss(1.0, 5.0)) for _ in range(800)]
        noise = [trade(rng.gauss(0.0, 5.0)) for _ in range(800)]
        losing = [trade(rng.gauss(-1.0, 5.0)) for _ in range(800)]
        self.assertEqual(S.summarize(real)["verdict"], "likely a real edge")
        self.assertIn(S.summarize(noise)["verdict"], ("can't tell from luck yet", "no edge visible"))
        self.assertEqual(S.summarize(losing)["verdict"], "losing, and it's not bad luck")

    def test_trades_needed(self):
        s = S.summarize([trade(x) for x in [1.0, 3.0] * 50])  # mean 2, sd ~1
        self.assertEqual(s["trades_needed"], 2)  # (2 * 1.005 / 2)^2 rounds up to 2

    def test_risk_stats(self):
        ts = [trade(5, hour="10"), trade(-3, hour="11"), trade(-4, hour="12"), trade(-1, hour="13"), trade(6, hour="14")]
        s = S.summarize(ts)
        self.assertAlmostEqual(s["max_drawdown"], -8.0)
        self.assertEqual(s["longest_losing_streak"], 3)

    def test_breakdowns(self):
        rows = S.breakdown([trade(5, hour="10"), trade(-3, hour="10"), trade(7, hour="14")], S.by_hour)
        self.assertEqual([(k, n) for k, n, *_ in rows], [("10:00", 2), ("14:00", 1)])

    def test_mfe_capture(self):
        s = S.summarize([trade(5, notional=1000, mfe_bps=100), trade(-2, notional=1000, mfe_bps=50)])
        self.assertAlmostEqual(s["kept_of_best_move"], (50 - 20) / 150)


class FakePos:
    def __init__(self, qty, avg, price):
        self.qty, self.avg_entry_price, self.current_price = str(qty), str(avg), str(price)


class TestJournal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.j = TradeJournal(os.path.join(self.tmp.name, "j.csv"), "test")

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_long_trade(self):
        self.j.note_entry("NVDA", True, "auto", bid=99.98, ask=100.02, limit=100.02, tp=104.02, sl=98.02,
                          signal={"price": 100.0, "fast": 100.1, "slow": 100.0, "rsi": 62.0})
        self.j.note_fill("NVDA", FakePos(10, 100.05, 100.05))
        for px in (100.4, 99.7, 100.2):
            self.j.mark("NVDA", px)
        self.j.note_exit_reason("NVDA", "trend reversal", "auto")
        row = self.j.close("NVDA", exit_fill=100.10)
        self.assertEqual(row["reason"], "signal")
        self.assertAlmostEqual(row["slippage_bps"], 5.0, places=3)
        self.assertAlmostEqual(row["spread_bps"], 4.0, places=3)
        self.assertAlmostEqual(row["pnl"], 0.5, places=6)
        self.assertAlmostEqual(row["mfe_bps"], (100.4 / 100.05 - 1) * 1e4, places=4)
        self.assertAlmostEqual(row["mae_bps"], (99.7 / 100.05 - 1) * 1e4, places=4)
        self.assertAlmostEqual(row["sma_gap_bps"], 10.0, places=3)
        back = self.j.read()
        self.assertEqual(len(back), 1)
        self.assertEqual(back[0]["reason"], "signal")
        self.assertEqual(S.summarize(back)["trades"], 1)

    def test_reason_inferred_from_bracket_prices(self):
        self.j.note_entry("AMD", True, "auto", 99.99, 100.01, 100.01, tp=104.01, sl=98.01, signal=None)
        self.j.note_fill("AMD", FakePos(5, 100.0, 100.0))
        self.assertEqual(self.j.close("AMD", 104.01)["reason"], "target")
        self.j.note_entry("AMD", True, "auto", 99.99, 100.01, 100.01, tp=104.01, sl=98.01, signal=None)
        self.j.note_fill("AMD", FakePos(5, 100.0, 100.0))
        self.assertEqual(self.j.close("AMD", 97.95)["reason"], "stop")

    def test_short_signs_and_flatten_reason(self):
        self.j.note_entry("TSLA", False, "auto", 199.9, 200.1, 199.9, tp=191.9, sl=203.9, signal={"price": 200.0})
        self.j.note_fill("TSLA", FakePos(-10, 199.9, 199.9))
        self.j.mark("TSLA", 198.0)
        self.j.note_exit_reason("TSLA", "end-of-day flatten")
        row = self.j.close("TSLA", 199.0)
        self.assertGreater(row["pnl"], 0)
        self.assertGreater(row["mfe_bps"], 0)
        self.assertEqual(row["reason"], "eod")

    def test_carried_position_without_entry_context(self):
        self.j.note_fill("KO", FakePos(50, 70.0, 70.0))
        row = self.j.close("KO", None, fallback_price=70.5)
        self.assertEqual(row["source"], "carried")
        self.assertAlmostEqual(row["pnl"], 25.0)

    def test_unknown_symbol_is_ignored(self):
        self.assertIsNone(self.j.close("XYZ", 1.0))


def load_bot():
    """Import the live bot with the broker SDK stubbed out (it isn't needed for these checks)."""
    class Any_:
        def __init__(self, *a, **k): self.__dict__.update(k)
        def __getattr__(self, n): return Any_()
        def __call__(self, *a, **k): return Any_()

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
    enum = types.SimpleNamespace(SIP="sip", IEX="iex", BUY="buy", SELL="sell", DAY="day", GTC="gtc", OCO="oco",
                                 BRACKET="bracket", OPEN="open", CLOSED="closed", Minute="1Min")
    for n in ("alpaca", "alpaca.trading", "alpaca.data"):
        mod(n)
    mod("alpaca.trading.client", TradingClient=Any_)
    mod("alpaca.trading.requests", **{k: Any_ for k in ("MarketOrderRequest", "LimitOrderRequest", "TakeProfitRequest",
                                                         "StopLossRequest", "GetOrdersRequest", "ClosePositionRequest",
                                                         "StopOrderRequest")})
    mod("alpaca.trading.enums", OrderSide=enum, TimeInForce=enum, OrderClass=enum, QueryOrderStatus=enum)
    mod("alpaca.data.historical", StockHistoricalDataClient=Any_)
    mod("alpaca.data.requests", StockBarsRequest=Any_, StockLatestQuoteRequest=Any_)
    mod("alpaca.data.timeframe", TimeFrame=enum)
    mod("alpaca.data.enums", DataFeed=enum)
    # Never touch the real account, .env or Discord from a test: stub dotenv, blank the webhook,
    # and use dummy keys (the broker client is stubbed anyway).
    mod("dotenv", load_dotenv=lambda *a, **k: None)
    os.environ["DISCORD_WEBHOOK_URL"] = ""
    os.environ["ALPACA_API_KEY"] = "test-key"
    os.environ["ALPACA_SECRET_KEY"] = "test-secret"
    spec = importlib.util.spec_from_file_location("sparticus_bot", ROOT / "sparticus_v6_4.py")
    bot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bot)
    return bot


class TestLiveBotJournalHooks(unittest.TestCase):
    """End to end inside the real engine: a position appears, moves, closes at its target."""

    def test_snapshot_cycle_writes_a_journal_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)   # the bot's log file and journal land in the temp folder, not yours
            try:
                bot = load_bot()
                eng = bot.SparticusEngineV55()
                acct = types.SimpleNamespace(equity="100000", last_equity="100000", cash="100000",
                                             buying_power="200000", daytrade_count=0)
                state = {"positions": []}
                closed_order = types.SimpleNamespace(filled_at=time.time(), side="sell", filled_qty="10",
                                                     filled_avg_price="104.0")
                eng.trading_client = types.SimpleNamespace(
                    get_account=lambda: acct, get_all_positions=lambda: state["positions"],
                    get_orders=lambda req: [closed_order])
                eng.journal.note_entry("NVDA", True, "auto", 99.99, 100.01, 100.01, 104.01, 98.01,
                                       {"price": 100.0, "fast": 100.2, "slow": 100.0, "rsi": 61.0})
                eng.refresh_snapshot()                                     # nothing held yet
                p = types.SimpleNamespace(symbol="NVDA", qty="10", avg_entry_price="100.02", current_price="100.02",
                                          unrealized_pl="0", unrealized_plpc="0", market_value="1000.2", side="long")
                state["positions"] = [p]
                eng.refresh_snapshot()                                     # fill
                p.current_price = "103.5"
                eng.refresh_snapshot()                                     # watermark
                state["positions"] = []
                eng.refresh_snapshot()                                     # exit -> journal thread
                for _ in range(50):
                    if Path("sparticus_journal.csv").exists() and eng.journal.read():
                        break
                    time.sleep(0.05)
                rows = eng.journal.read()
                self.assertEqual(len(rows), 1)
                r = rows[0]
                self.assertEqual(r["symbol"], "NVDA")
                self.assertEqual(r["reason"], "target")
                self.assertAlmostEqual(float(r["exit_fill"]), 104.0)
                self.assertAlmostEqual(float(r["mfe_bps"]), (104.0 / 100.02 - 1) * 1e4, places=1)
                eng.refresh_stats()
                self.assertEqual(eng.stats_cache["all"]["trades"], 1)
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
