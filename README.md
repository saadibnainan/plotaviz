# PlotaViz

[![CI](https://github.com/saadibnainan/plotaviz/actions/workflows/ci.yml/badge.svg)](https://github.com/saadibnainan/plotaviz/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/plotaviz)](https://pypi.org/project/plotaviz/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Qt: LGPLv3](https://img.shields.io/badge/Qt-LGPLv3-orange)](THIRD_PARTY_LICENSES.md)
[![Code style: ruff](https://img.shields.io/badge/style-ruff-261230)](https://docs.astral.sh/ruff/)

**A cross-platform desktop app that works out the right chart for your data, draws it, and hands
you the Python code that reproduces it.**

Open a CSV. PlotaViz cleans it, works out what each column actually is, ranks the charts that suit
the data — and tells you *why* it picked each one. Then it exports the image, or the standalone
script.

It runs entirely offline. An LLM is optional and only ever adds two things: a second opinion when
the ranking is close, and a query bar you can type "revenue by region over time" into.

<!-- Screenshots go here. For a GUI project this is the single biggest factor in whether anyone
     tries it, so add them as soon as there is a working chart to show. -->

| | |
|---|---|
| ![Main window](docs/screenshots/main-window.png) | ![Recommendations](docs/screenshots/recommendations.png) |
| *Preview, cleaning report, and the chart* | *Ranked alternatives, each with its reasoning* |

---

## Why it exists

Most plotting tools make you decide the chart first and then fight the syntax. PlotaViz inverts
that: describe nothing, get a ranked set of charts that suit the shape of your data, and switch
between them in one click. The parts that usually go wrong are treated as first-class problems:

- **Type inference is wrong constantly.** IDs read as measures, dates read as text. So the
  inferred types are shown up front and are editable *before* analysis proceeds.
- **Auto-generated charts hide their reasoning.** Every recommendation carries a plain-language
  justification, and the alternatives are real charts you can switch to, not a dropdown.
- **Big data silently breaks charts.** Above ~100k points a scatter freezes a browser view. PlotaViz
  downsamples (LTTB for time series) and says *"showing a sample of 100,000 of 8,400,000 points"*
  on the chart and in the export.
- **You can't take the result anywhere.** Export the Python and it runs without PlotaViz installed.

---

## Install

Python 3.11 or newer.

### From PyPI (all platforms)

```bash
pip install plotaviz
plotaviz
```

### From source (contributing / latest main)

```bash
git clone https://github.com/saadibnainan/plotaviz.git
cd plotaviz
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
plotaviz
```

### macOS

Apple Silicon and Intel are both supported.

```bash
brew install python@3.11          # if you need it
pip install plotaviz
```

A prebuilt `.app` is attached to each [release](https://github.com/saadibnainan/plotaviz/releases).
Builds are not yet notarized, so the first launch needs **right-click → Open**.

To build one yourself:

```bash
make build-macos                  # produces dist/PlotaViz.app
```

### Linux

Qt needs a few system libraries that pip cannot install for you:

```bash
# Debian / Ubuntu
sudo apt install python3-venv libegl1 libxkbcommon-x11-0 libxcb-cursor0 \
                 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0

# Fedora
sudo dnf install python3-devel libglvnd-egl xcb-util-cursor libxkbcommon-x11

# Arch
sudo pacman -S python libxkbcommon-x11 xcb-util-cursor
```

Then either `pip install plotaviz`, or grab the **AppImage** from the releases page:

```bash
chmod +x PlotaViz-*.AppImage
./PlotaViz-*.AppImage
```

To build one: `make build-appimage`. Recipes for `.deb`, `.rpm`, and an **AUR PKGBUILD** are in
[`scripts/build_appimage.sh`](scripts/build_appimage.sh).

### Optional extras

```bash
pip install "plotaviz[llm]"     # Anthropic / OpenAI / Gemini / Ollama providers
pip install "plotaviz[fast]"    # pyqtgraph fast path for very large numeric plots
pip install -e ".[dev]"         # tests, ruff, mypy, pre-commit — source checkout only
```

---

## Quickstart

```bash
plotaviz                                  # open the app
plotaviz samples/sales_timeseries.csv     # open the app with a file
```

1. Drag a dataset in, or use **File → Open**.
2. Read the **cleaning report** — what was renamed, filled, deduplicated, flagged.
3. Check the **Types** tab and correct anything inference got wrong.
4. Pick a chart from **Charts**. The reasoning for each is right there.
5. Narrow the data in **Filters** — per-column widgets, or a raw pandas query.
6. **File → Export** for the image, the interactive HTML, or the Python script.
7. **File → Save session** writes a `.pviz` file that restores all of it later.

### Command line

`core/` is completely headless, so the whole pipeline works without a display — useful in a
Makefile, a cron job, or CI.

```bash
# What is in this file?
plotaviz --input data.csv --describe

# What should I plot?
plotaviz --input data.csv --recommend

# Just make the chart
plotaviz --input data.csv --auto --export chart.png

# Give me the code instead
plotaviz --input data.csv --auto --export-code plot.py --code-flavour plotly

# Take control
plotaviz --input data.csv --chart box --x region --y revenue \
         --query "revenue > 1000" --export chart.pdf --dpi 300 --size 12x8
```

`--json` makes `--describe` and `--recommend` machine-readable.

---

## How it decides

Two layers, deliberately separate.

**Rules** ([`plotaviz/core/rules.yaml`](plotaviz/core/rules.yaml)) map a data *shape* to candidate
charts. They are broad on purpose — a date column plus a measure proposes line, area, and scatter,
and does not try to pick between them:

| Shape | Candidates |
|---|---|
| 1 numeric | histogram, KDE |
| 1 categorical | bar (low cardinality), treemap (high) |
| 2 numeric | scatter, with correlation and trendline |
| numeric × categorical | bar, box, violin |
| datetime × numeric | line, area |
| 2 categorical | grouped/stacked bar, heatmap |
| many numeric | correlation heatmap, pair plot |

**Scoring** then ranks those candidates against your actual data — cardinality, missingness,
distribution skew, correlation strength, readability limits. This is where *"you have 400
categories, a bar chart will be unreadable"* gets expressed, as a penalty rather than a
prohibition.

The output is a **chart spec**: the single contract that the renderer, the code generator, the
session file, and the LLM layer all consume. There is no other way to describe a chart in this
codebase.

### Extending the rules

Adding a chart usually means editing YAML, not Python:

```yaml
rules:
  - name: datetime_numeric
    requires: { datetime: 1, numeric: 1 }
    candidates:
      - chart: line
        base: 0.94
        map: { x: datetime.0, y: numeric.0, color: "?categorical.0" }
        why: "A date column paired with a measure is a time series."
```

`numeric.0` means "the most promising numeric column"; a `?` prefix makes a slot optional. Weights
and readability caps live under `scoring:` in the same file. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the whole procedure.

---

## The LLM layer is optional

**PlotaViz works fully offline with no provider configured.** The rules engine answers instantly,
and it is the fallback whenever a model call fails, times out, or returns something that does not
validate. A model is never on the critical path.

When you do configure one, it does two jobs:

1. Breaks a tie when the local scores are close enough that the ranking is a coin flip.
2. Powers the natural-language query bar.

Both return a **structured chart spec (JSON), never executable code.** The spec is validated
against your real column names before anything renders; a hallucinated column is a clear error
message, not a crash.

**What gets sent:** column names, summary statistics, and up to 5 sample rows. Never the dataset.
You are asked for consent before the first remote request.

**Where keys live:** your OS keychain, via `keyring`. Never a config file, never a session file,
never the repo. Keys are redacted from logs and error messages.

### Local models with Ollama (nothing leaves your machine)

```bash
# install ollama from https://ollama.com, then
ollama pull llama3.1
ollama serve
```

In **Settings → LLM provider**, choose **Ollama**. No API key, no network, no consent prompt —
this is the right choice for sensitive data.

Hosted providers: Anthropic (Claude), OpenAI, and Google Gemini. Install with
`pip install -e ".[llm]"`, then paste a key in Settings.

---

## Features

- **Loading** — CSV, TSV, Excel, JSON, NDJSON, Parquet. Delimiter and encoding sniffing. Files
  over a configurable threshold read through polars and profile on a sample.
- **Cleaning** — an ordered, replayable step list: name normalization, type coercion,
  missing-value strategies, IQR/z-score outlier flagging (non-destructive by default),
  deduplication. Every step reports what it changed.
- **Type override** — inferred roles shown and editable before analysis proceeds.
- **Filters** — range widgets, multi-selects, date ranges, and a raw pandas-query bar. Filters are
  pipeline steps, so they serialize into sessions and generated code for free.
- **Charts** — 15 types, interactive via Plotly in a WebEngine view, with a matplotlib canvas
  fallback when WebEngine is unavailable.
- **Export** — PNG, SVG, PDF, JPEG, WebP at up to 1200 DPI; self-contained interactive HTML; or a
  standalone Python script in matplotlib or Plotly flavour.
- **Sessions** — `.pviz` files store the recipe, not the data, and warn if the source changed.
- **Guardrails** — automatic downsampling with a visible notice, top-N category capping,
  virtualized preview, debounced re-render.

### Sample datasets

Five, each exercising a different path through the selector:

| File | Exercises |
|---|---|
| `sales_timeseries.csv` | time series; messy names, missing values, duplicates |
| `iris_like.csv` | correlated numeric pairs → scatter, correlation heatmap |
| `survey_responses.csv` | two categorical dimensions → grouped bar, heatmap |
| `sensor_readings.csv` | one skewed numeric column with real outliers → histogram, box |
| `city_population.json` | JSON loading, high-cardinality categorical → treemap |

Regenerate with `make samples`.

---

## Architecture

```
plotaviz/
├── main.py                 # entry point; GUI or CLI
├── ui/                     # PySide6 — may import Qt
│   ├── main_window.py      # layout and wiring; no analysis logic
│   ├── interactive_view.py # ALL QtWebEngine use is isolated here
│   ├── nl_query_bar.py     # natural-language input + editable interpretation
│   ├── filter_panel.py     # per-column filters + query bar
│   ├── type_override.py    # correct inferred column types
│   ├── export_dialog.py    # format, DPI, size, code flavour
│   ├── settings_dialog.py  # provider, keys, thresholds
│   ├── table_model.py      # virtualized preview
│   └── workers.py          # QThread plumbing, progress, cancellation
└── core/                   # HEADLESS — no Qt anywhere, unit-tested
    ├── loader.py           # file → DataFrame (pandas or polars)
    ├── preprocess.py       # the replayable step list
    ├── profiler.py         # column roles + statistics
    ├── selector.py         # rules + scoring engine
    ├── rules.yaml          # the editable half of selection
    ├── spec.py             # ChartSpec — the shared contract
    ├── plotter.py          # Plotly + matplotlib, and the guardrails
    ├── codegen.py          # standalone script generation
    ├── exporter.py         # PNG/SVG/PDF/HTML
    ├── session.py          # .pviz save/load
    ├── analysis.py         # orchestration facade shared by GUI and CLI
    └── llm/                # pluggable providers behind one ABC
```

Two invariants hold the design together:

- **`core/` never imports Qt.** This is what makes the CLI nearly free and the engine testable —
  and it is [checked by a test](tests/test_cli.py), not just by convention.
- **Preprocessing is a list of steps, not mutations.** One design gives undo/redo, session replay,
  and code generation at once.

---

## Development

```bash
make dev      # venv + dev extras + pre-commit hooks
make test     # pytest
make lint     # ruff check + ruff format --check + mypy plotaviz/core
make run      # launch the app
make samples  # regenerate sample datasets
```

The test suite covers loading, step replay and serialization, profiling and identifier detection,
selector scoring, spec validation, both renderers, session round-trips, the mocked LLM provider
contract, and — for every chart type — that the **generated script actually executes and produces
an image**.

See [CONTRIBUTING.md](CONTRIBUTING.md) for adding selector rules and LLM providers.

---

## Roadmap

Designed for, not built yet: auto EDA report export (HTML/PDF), a multi-chart dashboard grid, a
chart customization panel (titles, palettes, axis scales), undo/redo across the step stack,
colorblind-safe palettes and OS-following dark mode, and joining two datasets on a key.

Explicitly out of scope for v1: plugin architecture, localization, real-time streaming, and live
database connectors.

---

## Licensing

PlotaViz is **MIT** licensed (see [LICENSE](LICENSE)).

It uses **PySide6 (Qt), which is LGPLv3**. Distributed bundles link Qt dynamically so it can be
relinked, and ship attribution. If you redistribute a build, keep
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) with it — that file covers the Qt obligation
and credits Plotly (MIT), matplotlib, pandas, polars, Apache Arrow, and the rest.

Qt is a trademark of The Qt Company Ltd. PlotaViz is not affiliated with or endorsed by them.

Security policy and vulnerability reporting: [SECURITY.md](SECURITY.md).
