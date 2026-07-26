#!/usr/bin/env bash
# Build PlotaViz.app for macOS with PyInstaller.
#
# Qt stays dynamically linked (PyInstaller's default), which is what keeps the LGPL relinking
# obligation satisfiable — see THIRD_PARTY_LICENSES.md. Code signing and notarization are a
# separate, later step; the notes at the bottom cover them.

set -euo pipefail

cd "$(dirname "$0")/.."

APP_NAME="PlotaViz"
# PyInstaller runs its target as a standalone top-level script, which breaks plotaviz/main.py's
# relative import (`from . import __version__`). This shim imports the package properly first.
ENTRY="scripts/pyinstaller_entry.py"
# `pip install -e .` (PEP 660) installs plotaviz behind a custom import-hook finder rather than a
# plain directory in site-packages. PyInstaller's static analysis can't walk that finder, so it
# silently drops the whole package instead of erroring — `--paths .` points it at the real source
# directory instead, which it can walk normally like any non-editable install.

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: this script builds a macOS .app and must run on macOS." >&2
  exit 1
fi

PYINSTALLER="${PYINSTALLER:-pyinstaller}"
if ! command -v "$PYINSTALLER" >/dev/null 2>&1; then
  if [[ -x .venv/bin/pyinstaller ]]; then
    PYINSTALLER=.venv/bin/pyinstaller
  else
    echo "error: pyinstaller not found. Install it with: pip install -e '.[build]'" >&2
    exit 1
  fi
fi

echo "==> Cleaning previous build"
rm -rf build "dist/${APP_NAME}.app"

# QtWebEngine is the reason this bundle is large. Excluding the Qt modules PlotaViz never touches
# claws back a few hundred megabytes; leave WebEngine itself in, since the interactive chart view
# depends on it and the app falls back to matplotlib only when it is genuinely absent.
EXCLUDES=(
  --exclude-module PySide6.Qt3DCore
  --exclude-module PySide6.Qt3DRender
  --exclude-module PySide6.QtBluetooth
  --exclude-module PySide6.QtCharts
  --exclude-module PySide6.QtDataVisualization
  --exclude-module PySide6.QtMultimedia
  --exclude-module PySide6.QtNfc
  --exclude-module PySide6.QtQuick3D
  --exclude-module PySide6.QtSensors
  --exclude-module PySide6.QtSerialPort
  --exclude-module PySide6.QtTest
  --exclude-module tkinter
  --exclude-module PyQt5
  --exclude-module PyQt6
)

echo "==> Building ${APP_NAME}.app"
"$PYINSTALLER" \
  --noconfirm \
  --clean \
  --windowed \
  --name "$APP_NAME" \
  --paths . \
  --osx-bundle-identifier "com.plotaviz.app" \
  --add-data "plotaviz/core/rules.yaml:plotaviz/core" \
  --collect-data plotly \
  --hidden-import plotaviz.main \
  --hidden-import plotaviz.ui.main_window \
  "${EXCLUDES[@]}" \
  "$ENTRY"

if [[ ! -d "dist/${APP_NAME}.app" ]]; then
  echo "error: build finished but dist/${APP_NAME}.app is missing." >&2
  exit 1
fi

SIZE=$(du -sh "dist/${APP_NAME}.app" | cut -f1)
echo
echo "==> Built dist/${APP_NAME}.app (${SIZE})"
echo
cat <<'NOTES'
Notes
-----
* Most of the bundle size is QtWebEngine's embedded Chromium. Removing the interactive view
  entirely would roughly halve it; PlotaViz keeps it because losing zoom/hover is a real
  downgrade. `--exclude-module` lines above trim the Qt modules that go unused.

* Qt is dynamically linked. Ship THIRD_PARTY_LICENSES.md alongside any distribution — the LGPL
  requires the attribution and the ability to relink.

* Code signing and notarization (do this before distributing outside your own machine):

    codesign --deep --force --options runtime \
      --sign "Developer ID Application: YOUR NAME (TEAMID)" dist/PlotaViz.app

    ditto -c -k --keepParent dist/PlotaViz.app dist/PlotaViz.zip
    xcrun notarytool submit dist/PlotaViz.zip \
      --apple-id you@example.com --team-id TEAMID --password APP_SPECIFIC_PASSWORD --wait
    xcrun stapler staple dist/PlotaViz.app

  Unsigned builds run, but Gatekeeper will make the first launch unpleasant for anyone else.
NOTES
