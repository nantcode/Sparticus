"""V6.3.2 tests: the safety fixes from your Oct 7 build (circuit breaker, watchdog stages that can't
take each other down, secret redaction, sign check). Offline, in a throwaway folder.

    python3 -m unittest discover -s tests -v
"""
import contextlib
import io
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import sparticus_strategy as strat  # noqa: E402
from test_stats_journal import load_bot  # noqa: E402

_SAFE_CWD = []


def setUpModule():
    _SAFE_CWD.append((os.getcwd(), tempfile.TemporaryDirectory()))
    os.chdir(_SAFE_CWD[-1][1].name)


def tearDownModule():
    cwd, tmp = _SAFE_CWD.pop()
    os.chdir(cwd)
    tmp.cleanup()


class TestCircuitBreaker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = load_bot()

    def test_trip_halts_the_day_and_flattens(self):
        eng = self.bot.SparticusEngineV55()
        flattened = threading.Event()
        eng.flatten_all_positions = lambda *a, **k: flattened.set() or True
        eng.autopilot_enabled = True
        eng.trip_circuit_breaker(-0.06)
        self.assertTrue(eng.circuit_breaker_tripped)
        self.assertFalse(eng.autopilot_enabled)
        self.assertTrue(flattened.wait(2))

    def test_one_failing_watchdog_stage_cannot_stop_the_others(self):
        bot, eng = self.bot, self.bot.SparticusEngineV55()
        loser = types.SimpleNamespace(symbol="KO", unrealized_plpc="-0.05", qty="10")
        eng.refresh_snapshot = lambda: (None, {"KO": loser})
        eng.session_info = lambda: {"open": True, "mins_to_close": 200}
        closed = threading.Event()
        eng.close_position_safely = lambda *a, **k: closed.set()
        eng.cancel_stale_entries = lambda: None
        saved = bot._check_circuit_breaker
        bot._check_circuit_breaker = lambda *a: (_ for _ in ()).throw(AttributeError("boom"))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                bot.watchdog_cycle(eng)
        finally:
            bot._check_circuit_breaker = saved
        self.assertTrue(closed.wait(2), "the -3% stop must still fire when another stage fails")


class TestRedaction(unittest.TestCase):
    def test_keys_and_webhook_tokens_never_reach_the_log(self):
        bot = load_bot()
        text = bot.redact("key=test-key secret=test-secret url: /api/webhooks/1234567890/AbC-def_123XYZ")
        self.assertNotIn("test-key", text)
        self.assertNotIn("test-secret", text)
        self.assertNotIn("AbC-def_123XYZ", text)
        self.assertIn("/api/webhooks/<redacted>", text)


class TestSignCheck(unittest.TestCase):
    def test_positive_stop_settings_refuse_to_start(self):
        saved = strat.LIVE_DEFAULTS
        strat.LIVE_DEFAULTS = replace(saved, hard_stop_pct=0.03)
        try:
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stdout(io.StringIO()):
                load_bot()
            self.assertEqual(cm.exception.code, 4)
        finally:
            strat.LIVE_DEFAULTS = saved


class TestTestsStayOutOfYourFolder(unittest.TestCase):
    def test_suite_runs_in_a_throwaway_folder(self):
        self.assertNotEqual(Path.cwd().resolve(), ROOT.resolve())
        self.assertFalse(Path("sparticus_journal.csv").resolve().parent == ROOT.resolve())


if __name__ == "__main__":
    unittest.main()
