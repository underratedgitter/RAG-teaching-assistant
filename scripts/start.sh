#!/usr/bin/env bash
# One command from a fresh clone to a running dashboard: make the virtualenv,
# install what requirements.txt asks for, check ffmpeg and Ollama, then launch.
# Safe to run again: it only installs when requirements.txt has changed.
#
#   RAG_NO_GUI=1      set everything up and check that the dashboard imports,
#                     without opening the window (CI and smoke tests)
#   RAG_AUTO_PULL=1   pull missing Ollama models without asking
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

venv=.venv
py="$venv/bin/python"
stamp="$venv/.requirements.sha256"

say() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

in_range() {
  "$1" -c 'import sys; sys.exit(not ((3, 11) <= sys.version_info[:2] <= (3, 14)))' 2>/dev/null
}

create_venv() {
  if command -v uv >/dev/null 2>&1; then
    say "Creating $venv with uv"
    uv venv --quiet --python '>=3.11,<3.15' "$venv"
    return
  fi
  local candidate
  for candidate in python3.13 python3.12 python3.14 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && in_range "$candidate"; then
      say "Creating $venv with $candidate"
      "$candidate" -m venv "$venv"
      return
    fi
  done
  die "Python 3.11 to 3.14 not found. Install it from https://www.python.org/downloads/ or install uv (https://docs.astral.sh/uv/) and run this again."
}

requirements_hash() {
  "$py" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' requirements.txt
}

install_failed() {
  die "Installing requirements failed (error above). A Python with no prebuilt wheels for your machine is the usual cause: delete $venv and run this again with Python 3.11 to 3.13, or with uv installed, which fetches one."
}

install_requirements() {
  say "Installing requirements"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --quiet --python "$py" -r requirements.txt || install_failed
  else
    "$py" -m pip install --quiet --disable-pip-version-check -r requirements.txt || install_failed
  fi
  requirements_hash > "$stamp"
}

[[ -x $py ]] || create_venv
in_range "$py" || die "$venv uses a Python outside 3.11 to 3.14. Delete it (rm -rf $venv) and run this again."

if [[ ! -f $stamp ]] || [[ "$(cat "$stamp")" != "$(requirements_hash)" ]]; then
  install_requirements
fi

say "Checking ffmpeg, Ollama and models"
"$py" preflight.py || say "Something above is missing. The dashboard still opens, but processing or answering needs it."

if [[ ${RAG_NO_GUI:-0} == 1 ]]; then
  "$py" -c 'import dashboard' || die "dashboard.py does not import. If the error mentions tkinter, install your Python's Tk package (macOS: brew install python-tk, Debian/Ubuntu: sudo apt install python3-tk)."
  say "RAG_NO_GUI=1: setup finished and the dashboard imports; not opening the window."
  exit 0
fi

say "Starting the dashboard"
exec "$py" dashboard.py
