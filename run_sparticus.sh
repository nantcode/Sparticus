#!/bin/bash
# Keeps Sparticus running for the whole trading day on a Mac.
#   * caffeinate stops the Mac from going to sleep while the bot runs
#     (keep it plugged in, and keep the lid open: closing the lid still sleeps a laptop)
#   * if the bot crashes or the health monitor finds it stuck, it restarts after 15 seconds
#   * Ctrl+C stops everything
#
# First time only:  chmod +x run_sparticus.sh
# Every day:        ./run_sparticus.sh          (no window, most reliable)
#             or:   ./run_sparticus.sh hud      (with the brain HUD; closing the window stops the bot and, during market hours, closes its positions first)

cd "$(dirname "$0")" || exit 1
BOT="sparticus_v6_4.py"
PY="${SPARTICUS_PYTHON:-python3}"   # the autostart sets this to the Python you installed with
if [ "$1" = "hud" ]; then ARGS=(); else ARGS=(--headless); fi
trap 'echo; echo "Stopped by you."; exit 0' INT TERM

# caffeinate (keeps a Mac awake) only exists on macOS; elsewhere run the bot directly.
if command -v caffeinate >/dev/null 2>&1; then KEEP_AWAKE=(caffeinate -is); else KEEP_AWAKE=(); fi

while true; do
  "${KEEP_AWAKE[@]}" "$PY" "$BOT" "${ARGS[@]}"
  code=$?
  case $code in
    0|130) echo "Sparticus exited normally."; break ;;
    4)     echo "Sparticus stopped on a setup problem (missing keys, or another copy is already running). Not restarting."; break ;;
    *)     echo "Sparticus exited with code $code. Restarting in 15 seconds (Ctrl+C to cancel)..."; sleep 15 ;;
  esac
done
