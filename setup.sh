#!/usr/bin/env bash
# Optional shortcut for the manual steps in README.md.
# Read this file before you run it.
#
# It only:
#   - looks for Python 3.11+
#   - installs Python 3.12 with winget if none is found
#   - creates .venv in this folder
#   - installs requirements.txt into that virtualenv
#   - prints the command that starts the server
#
# It does not install Ollama, download a model, change git settings,
# or write anything outside this folder except that one winget install.

set -euo pipefail
cd "$(dirname "$0")"

python_ok() {
  "$@" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >/dev/null 2>&1
}

PY=()

pick() {
  if python_ok "$@"; then
    PY=("$@")
    return 0
  fi
  return 1
}

consider() {
  local candidate="$1"
  if [[ -f "$candidate" ]]; then
    pick "$candidate" || true
  fi
}

if command -v py >/dev/null 2>&1; then
  pick py -3 || true
fi
if [[ ${#PY[@]} -eq 0 ]] && command -v python3 >/dev/null 2>&1; then
  pick python3 || true
fi
if [[ ${#PY[@]} -eq 0 ]] && command -v python >/dev/null 2>&1; then
  pick python || true
fi
if [[ ${#PY[@]} -eq 0 && -n "${LOCALAPPDATA:-}" ]]; then
  consider "${LOCALAPPDATA}/Programs/Python/Python312/python.exe"
  if [[ ${#PY[@]} -eq 0 ]] && command -v cygpath >/dev/null 2>&1; then
    consider "$(cygpath -u "${LOCALAPPDATA}/Programs/Python/Python312/python.exe")"
  fi
fi

if [[ ${#PY[@]} -eq 0 ]]; then
  echo "No Python 3.11+ found. Installing Python 3.12 with winget..."
  winget install --id Python.Python.3.12 -e --accept-package-agreements --accept-source-agreements --disable-interactivity || echo "winget could not install Python. Checking again for an interpreter."
  candidates=()
  if [[ -n "${LOCALAPPDATA:-}" ]]; then
    candidates+=("${LOCALAPPDATA}/Programs/Python/Python312/python.exe")
    if command -v cygpath >/dev/null 2>&1; then
      candidates+=("$(cygpath -u "${LOCALAPPDATA}/Programs/Python/Python312/python.exe")")
    fi
  fi
  win_local="$(cmd.exe //c "echo %LOCALAPPDATA%\\Programs\\Python\\Python312\\python.exe" | tr -d '\r')"
  candidates+=("$win_local")
  if command -v cygpath >/dev/null 2>&1; then
    candidates+=("$(cygpath -u "$win_local")")
  fi
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      pick "$candidate" && break
    fi
  done
  if [[ ${#PY[@]} -eq 0 ]] && command -v py >/dev/null 2>&1; then
    pick py -3 || true
  fi
fi

if [[ ${#PY[@]} -eq 0 ]]; then
  echo "Python 3.11+ is still not available."
  echo "Install it from https://www.python.org/downloads/ and run this script again."
  exit 1
fi

echo "Using: ${PY[*]}"
if [[ ! -x .venv/Scripts/python.exe ]]; then
  "${PY[@]}" -m venv .venv
fi
.venv/Scripts/python.exe -m pip install -r requirements.txt

echo
echo "Setup finished. Start the app with:"
echo "  .venv/Scripts/python -m uvicorn app.main:app --reload"
echo "Then open http://127.0.0.1:8000"
