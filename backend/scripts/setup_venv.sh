#!/usr/bin/env bash
# Create a Python 3.11+ virtualenv for ChordLift backend (matches Docker).
set -euo pipefail

cd "$(dirname "$0")/.."
VENV_DIR="${VENV_DIR:-.venv}"

pick_python() {
  for cmd in /opt/homebrew/opt/python@3.11/libexec/bin/python3.11 python3.12 python3.11 python3; do
    if command -v "$cmd" >/dev/null 2>&1; then
      version=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
      major=${version%%.*}
      minor=${version#*.}
      if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
        echo "$cmd"
        return 0
      fi
    fi
  done
  return 1
}

if ! PYTHON=$(pick_python); then
  echo "Python 3.11+ required. Install with: brew install python@3.11" >&2
  exit 1
fi

echo "Using $PYTHON ($($PYTHON --version))"
"$PYTHON" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Optional ML engine (ensemble: lv-chordia + stem bar vote):"
echo "  pip install -r requirements-ml.txt"
echo ""
echo "Activate: source $VENV_DIR/bin/activate"
