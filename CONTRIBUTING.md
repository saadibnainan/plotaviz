# Contributing to PlotaViz

Thanks for helping out. This document covers dev setup, project conventions, and the two
extension points people most often want to touch: **selector rules** and **LLM providers**.

## Naming convention (please respect it)

Display name is **PlotaViz**. Everything machine-readable is lowercase `plotaviz` — repo,
package directory, import name, CLI command, PyPI name. Session files use the `.pviz` extension.

## Dev setup

```bash
git clone https://github.com/saadibnainan/plotaviz.git
cd plotaviz
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

Run the app:

```bash
make run          # or: python -m plotaviz.main
```

Run the checks:

```bash
make lint         # ruff check + ruff format --check + mypy plotaviz/core
make test         # pytest
```

On Linux you need Qt's system libraries for the GUI:

```bash
sudo apt-get install -y libegl1 libxkbcommon-x11-0 libxcb-cursor0 \
  libxcb-icccm4 libxcb-keysyms1 libxcb-shape0
```

## Architecture rules

The one hard rule: **`plotaviz/core/` must never import Qt.** Everything in `core/` is headless
and unit-testable, which is what makes the CLI mode and the test suite cheap. UI code lives in
`plotaviz/ui/` and talks to `core/` through plain data objects.

The other structural invariant is the **`ChartSpec`** (`core/spec.py`). It is the single contract
consumed by the plotter, the code generator, the session file, and the LLM layer. If you add a
chart capability, extend `ChartSpec` — do not pass side-channel state between modules.

Preprocessing is an **ordered, replayable list of steps** (`core/preprocess.py`), not in-place
mutation. That design is what makes session replay, undo/redo, and code generation possible from
one implementation. New cleaning behaviour should be a new `Step` subclass, not an inline
transform.

## Adding a selector rule

Rules live in `plotaviz/core/rules.yaml` and are loaded by `core/selector.py`.

1. Add a rule entry. `match` describes the data shape by role counts; `candidates` lists the
   chart types it proposes with a base score.

```yaml
rules:
  - name: datetime_x_numeric
    match: { datetime: 1, numeric: 1 }
    candidates:
      - { chart: line, score: 0.92, why: "One datetime and one numeric column — a time series." }
      - { chart: area, score: 0.70, why: "Area emphasises cumulative magnitude over time." }
```

2. If your chart type is new, add a builder to `core/plotter.py`
   (`_build_plotly_<chart>` and `_build_mpl_<chart>`) and a template branch in `core/codegen.py`.
3. Scoring adjustments (cardinality penalties, skew bonuses, and so on) live under `scoring:` in
   the same YAML, so tuning weights needs no code change.
4. Add a case to `tests/test_selector.py` asserting your rule ranks first for the shape it targets.

## Adding an LLM provider

Implement `plotaviz.core.llm.base.Provider`:

```python
class MyProvider(Provider):
    name = "myprovider"
    default_model = "my-model-1"

    def complete(self, system: str, user: str, *, timeout: float) -> str: ...
```

Then register it in `plotaviz/core/llm/__init__.py::PROVIDERS`. The base class already handles
prompt construction, JSON extraction, and schema validation of the returned chart spec — a
provider only has to turn two strings into one string.

Two constraints that are not negotiable:

- The model returns a **structured chart spec (JSON), never executable code.** Specs are validated
  against the real dataframe schema before rendering.
- The model receives **schema, summary statistics, and a handful of sample rows only** — never the
  full dataset.

## Commit / PR conventions

- Branch off `main`; PRs are squash-merged, so one PR is one commit in history.
- CI must be green: `ruff check`, `ruff format --check`, `mypy plotaviz/core`, `pytest`.
- Tests are required for anything in `core/`. GUI code is exercised loosely; do not block on it.
- Keep sample datasets under a few MB.

## Reporting bugs

Use the issue templates. For data-dependent bugs, a minimal CSV that reproduces the problem is
worth more than a stack trace.
