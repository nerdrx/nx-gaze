#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
    gaze_python="${NX_GAZE_PYTHON:-}"
    if [[ -z "$gaze_python" ]]; then
        for candidate in python3.12 python3.11 python3; do
            if command -v "$candidate" >/dev/null && "$candidate" -c "import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 12)" 2>/dev/null; then
                gaze_python="$candidate"
                break
            fi
        done
    fi
    if [[ -z "$gaze_python" ]]; then
        echo "Install Python 3.11 or 3.12, then run NX_GAZE_PYTHON=python3.12 ./setup.sh" >&2
        exit 1
    fi
    "$gaze_python" -c "import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 12), \"Python 3.11 or 3.12 required\""
    "$gaze_python" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
