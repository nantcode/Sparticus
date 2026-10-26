#!/bin/bash
# One command: test your current settings against the alternatives on about 6 months of history,
# then print what to do (keep everything, or exactly which lines to change).
#
#   ./optimize.sh                      first run downloads market data: give it a while
#   ./optimize.sh --start 2026-01-01   extra options go straight to backtest.py
#
# Run it with the market closed (evenings or weekends). It only reads market data; it never trades.

cd "$(dirname "$0")" || exit 1
PY="${SPARTICUS_PYTHON:-python3}"
EXPERIMENTS="live,hold5,live_hold,tight_brackets,atr_2to1,atr_2to1_hold,cap2,cooldown60,orb,spy_momentum"

"$PY" backtest.py --experiments "$EXPERIMENTS" "$@" || exit $?
echo
"$PY" lab_verdict.py lab_results "$EXPERIMENTS"
echo
echo "Full report: lab_results/summary.md"
