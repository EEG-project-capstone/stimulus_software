#!/bin/bash

# One-time environment setup for Stimulus Software
# Supports: macOS, Linux (Debian/Ubuntu), ChromeOS (Linux enabled)
# Does NOT require conda or any pre-installed Python manager.

set -euo pipefail

VENV_DIR=".venv"

# ── Helpers ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { printf "${GREEN}[setup]${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}[setup] WARN:${NC} %s\n" "$*"; }
die()  { printf "${RED}[setup] ERROR:${NC} %s\n" "$*" >&2; exit 1; }

# ── Check if a Python version is usable (3.11+) ────────────────────────────────
python_ok() {
  local cmd="$1"
  command -v "$cmd" &>/dev/null || return 1
  local major minor
  read -r major minor < <("$cmd" -c \
    "import sys; print(sys.version_info.major, sys.version_info.minor)" 2>/dev/null) \
    || return 1
  [[ "$major" -gt 3 ]] || { [[ "$major" -eq 3 ]] && [[ "$minor" -ge 11 ]]; }
}

# ── OS detection ───────────────────────────────────────────────────────────────
OS="$(uname -s)"
case "$OS" in
  Darwin) PLATFORM="macos" ;;
  Linux)  PLATFORM="linux" ;;
  *)      die "Unsupported OS: $OS. Please install Python 3.12+ and dependencies manually." ;;
esac
info "Detected platform: $PLATFORM"

# ── macOS: install via Homebrew ────────────────────────────────────────────────
setup_macos() {
  if ! command -v brew &>/dev/null; then
    info "Homebrew not found — installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add Homebrew to PATH for the rest of this script
    [[ -x /opt/homebrew/bin/brew ]] && eval "$(/opt/homebrew/bin/brew shellenv)"
    [[ -x /usr/local/bin/brew    ]] && eval "$(/usr/local/bin/brew shellenv)"
  fi

  info "Installing Python 3.12, PortAudio, and ffmpeg via Homebrew..."
  brew install python@3.12 python-tk@3.12 portaudio ffmpeg 2>/dev/null || true
  # Ensure python3.12 is on PATH (brew link may be needed)
  brew link --overwrite python@3.12 2>/dev/null || true
}

# ── Linux / ChromeOS: install via apt ─────────────────────────────────────────
setup_linux() {
  command -v apt-get &>/dev/null \
    || die "apt-get not found. Only Debian/Ubuntu/ChromeOS Linux is supported by this script."

  info "Updating apt package lists..."
  sudo apt-get update -qq

  # Check if we already have a suitable Python version
  if python_ok python3; then
    info "python3 already available ($(python3 --version))"
  else
    # python3.12 may not be in default repos on older Ubuntu — try deadsnakes PPA
    if ! apt-cache show python3.12 &>/dev/null 2>&1; then
      info "python3.12 not in default repos — trying deadsnakes PPA..."
      if command -v software-properties-common &>/dev/null || sudo apt-get install -y software-properties-common 2>/dev/null; then
        sudo add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || warn "Could not add PPA, will attempt to use system Python"
        sudo apt-get update -qq 2>/dev/null || true
      fi
    fi
  fi

  info "Installing system packages (python3, venv, tk, portaudio, ffmpeg)..."
  # Try to install python3-tk, but don't fail if it doesn't exist as python3.12-tk
  sudo apt-get install -y \
    python3-venv \
    python3-tk \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg 2>/dev/null || {
    warn "Some packages failed to install, attempting alternative names..."
    sudo apt-get install -y \
      portaudio19-dev \
      libsndfile1 \
      ffmpeg 2>/dev/null || true
  }
}

[[ "$PLATFORM" == "macos" ]] && setup_macos || setup_linux

# ── Find a usable Python 3.11+ ────────────────────────────────────────────────

PYTHON=""
for candidate in \
    python3.12 \
    python3.11 \
    python3 \
    python \
    /opt/homebrew/bin/python3.12 \
    /opt/homebrew/opt/python@3.12/bin/python3.12 \
    /usr/local/bin/python3.12; do
  if python_ok "$candidate"; then
    PYTHON="$candidate"
    break
  fi
done

[[ -n "$PYTHON" ]] \
  || die "Python 3.11+ not found after installation. Check your PATH or install manually."
info "Using Python: $PYTHON ($($PYTHON --version))"

# ── Create / update virtual environment ───────────────────────────────────────
if [[ -d "$VENV_DIR" ]]; then
  info "Virtual environment '${VENV_DIR}' already exists — updating..."
else
  info "Creating virtual environment in '${VENV_DIR}'..."
  "$PYTHON" -m venv "$VENV_DIR"
fi

# ── Install Python dependencies ───────────────────────────────────────────────
info "Upgrading pip..."
"$VENV_DIR/bin/pip" install --upgrade pip -q

info "Installing Python dependencies from pyproject.toml..."
DEPS=$("$VENV_DIR/bin/python" -c \
  "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(' '.join(d['project']['dependencies']))")
# shellcheck disable=SC2086
"$VENV_DIR/bin/pip" install $DEPS -q

info ""
info "Setup complete. Run ./run.sh to start the application."
