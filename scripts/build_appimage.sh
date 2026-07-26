#!/usr/bin/env bash
# Build a PlotaViz AppImage — the primary universal Linux artifact.
#
# The recipe is: PyInstaller produces a one-folder bundle, that folder becomes an AppDir with a
# desktop entry and an icon, and appimagetool packs the AppDir. Qt stays dynamically linked so
# the LGPL relinking obligation remains satisfiable (see THIRD_PARTY_LICENSES.md).
#
# Build on the OLDEST distribution you intend to support. glibc is forward-compatible, not
# backward-compatible, so an AppImage built on Ubuntu 24.04 will not start on 22.04.

set -euo pipefail

cd "$(dirname "$0")/.."

APP_NAME="PlotaViz"
# PyInstaller runs its target as a standalone top-level script, which breaks plotaviz/main.py's
# relative import (`from . import __version__`). This shim imports the package properly first.
ENTRY="scripts/pyinstaller_entry.py"
# `pip install -e .` (PEP 660) installs plotaviz behind a custom import-hook finder rather than a
# plain directory in site-packages. PyInstaller's static analysis can't walk that finder, so it
# silently drops the whole package instead of erroring — `--paths .` (below) points it at the real
# source directory instead, which it can walk normally like any non-editable install.
VERSION="$(python3 -c 'import re,pathlib; print(re.search(r"__version__ = \"([^\"]+)\"", pathlib.Path("plotaviz/__init__.py").read_text()).group(1))')"
ARCH="$(uname -m)"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "error: AppImages are built on Linux." >&2
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
rm -rf build dist/AppDir "dist/${APP_NAME}"

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

echo "==> Building the one-folder bundle"
"$PYINSTALLER" \
  --noconfirm \
  --clean \
  --name "$APP_NAME" \
  --paths . \
  --add-data "plotaviz/core/rules.yaml:plotaviz/core" \
  --collect-data plotly \
  --hidden-import plotaviz.main \
  --hidden-import plotaviz.ui.main_window \
  "${EXCLUDES[@]}" \
  "$ENTRY"

echo "==> Assembling the AppDir"
APPDIR="dist/AppDir"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" \
         "$APPDIR/usr/share/icons/hicolor/256x256/apps"
cp -r "dist/${APP_NAME}/." "$APPDIR/usr/bin/"

cat > "$APPDIR/usr/share/applications/plotaviz.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=PlotaViz
Comment=Automatic data analytics and visualization
Exec=PlotaViz %f
Icon=plotaviz
Categories=Science;Education;DataVisualization;
Terminal=false
MimeType=text/csv;application/vnd.ms-excel;application/json;
EOF
cp "$APPDIR/usr/share/applications/plotaviz.desktop" "$APPDIR/plotaviz.desktop"

# A generated placeholder icon keeps the build self-contained; replace it with real artwork in
# plotaviz/resources/plotaviz.png when there is some.
ICON_SRC="plotaviz/resources/plotaviz.png"
if [[ -f "$ICON_SRC" ]]; then
  cp "$ICON_SRC" "$APPDIR/usr/share/icons/hicolor/256x256/apps/plotaviz.png"
else
  python3 - <<'PY'
import struct, zlib, pathlib

size = 256
rows = []
for y in range(size):
    row = bytearray([0])
    for x in range(size):
        # A simple diagonal gradient stands in for real artwork.
        row += bytes((40 + x // 3, 90 + y // 4, 160, 255))
    rows.append(bytes(row))

def chunk(kind, data):
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

png = (
    b"\x89PNG\r\n\x1a\n"
    + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress(b"".join(rows), 9))
    + chunk(b"IEND", b"")
)
target = pathlib.Path("dist/AppDir/usr/share/icons/hicolor/256x256/apps/plotaviz.png")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes(png)
PY
fi
cp "$APPDIR/usr/share/icons/hicolor/256x256/apps/plotaviz.png" "$APPDIR/plotaviz.png"

cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
export PATH="$HERE/usr/bin:$PATH"
# Qt's own plugins ship inside the bundle; a host QT_PLUGIN_PATH would break them.
unset QT_PLUGIN_PATH
exec "$HERE/usr/bin/PlotaViz" "$@"
EOF
chmod +x "$APPDIR/AppRun"

echo "==> Fetching appimagetool"
TOOL="build/appimagetool-${ARCH}.AppImage"
mkdir -p build
if [[ ! -x "$TOOL" ]]; then
  curl -fsSL -o "$TOOL" \
    "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-${ARCH}.AppImage"
  chmod +x "$TOOL"
fi

echo "==> Packing the AppImage"
OUTPUT="dist/${APP_NAME}-${VERSION}-${ARCH}.AppImage"
# ARCH is read by appimagetool; --appimage-extract-and-run avoids needing FUSE on CI runners.
ARCH="$ARCH" "$TOOL" --appimage-extract-and-run "$APPDIR" "$OUTPUT"

SIZE=$(du -sh "$OUTPUT" | cut -f1)
echo
echo "==> Built ${OUTPUT} (${SIZE})"
echo
cat <<'NOTES'
Notes
-----
* QtWebEngine's Chromium is most of that size. It is also the most fragile part of an AppImage:
  its sandbox needs either unprivileged user namespaces or --no-sandbox. If the interactive view
  fails to start on a user's system, PlotaViz falls back to the matplotlib renderer rather than
  refusing to open.

* Build on the oldest glibc you intend to support.

* Other packaging targets, if you want distro-native builds:

  Debian/Ubuntu (.deb): use `fpm` over the PyInstaller output, or dh-virtualenv.
      fpm -s dir -t deb -n plotaviz -v VERSION --depends libxcb-cursor0 \
          dist/PlotaViz/=/opt/plotaviz/

  Fedora/RHEL (.rpm): same, with `-t rpm` and `--depends xcb-util-cursor`.

  Arch (AUR PKGBUILD):

      pkgname=plotaviz
      pkgver=VERSION
      pkgrel=1
      pkgdesc="Automatic data analytics and visualization"
      arch=('any')
      url="https://github.com/saadibnainan/plotaviz"
      license=('MIT')
      depends=('python' 'pyside6' 'python-pandas' 'python-polars' 'python-plotly'
               'python-matplotlib' 'python-pyarrow' 'python-keyring' 'python-yaml')
      makedepends=('python-build' 'python-installer' 'python-setuptools' 'python-wheel')
      source=("$pkgname-$pkgver.tar.gz::$url/archive/v$pkgver.tar.gz")
      build()   { cd "$pkgname-$pkgver"; python -m build --wheel --no-isolation; }
      package() { cd "$pkgname-$pkgver"; python -m installer --destdir="$pkgdir" dist/*.whl; }

  The AUR route is the nicest of the three: Arch already packages PySide6, so the package stays
  tiny and Qt is shared with the rest of the system.
NOTES
