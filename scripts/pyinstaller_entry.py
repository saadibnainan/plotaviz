"""PyInstaller entry point.

PyInstaller runs its target script as a standalone top-level module, not as part of the package
it lives in — so ``plotaviz/main.py``'s relative import (``from . import __version__``) fails at
runtime with "attempted relative import with no known parent package" once bundled, even though
it works fine for every other entry path (``python -m plotaviz.main``, the ``plotaviz`` console
script, `pytest`). This tiny shim is the standard fix: it imports ``plotaviz`` as a real package
first, so the package's own relative imports resolve normally, then hands off to ``main()``.
"""

from __future__ import annotations

import sys

from plotaviz.main import main

if __name__ == "__main__":
    sys.exit(main())
