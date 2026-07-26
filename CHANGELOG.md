# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-26

### Added

- **Data loading** for CSV, TSV, Excel, JSON/NDJSON, and Parquet, with automatic delimiter and
  encoding sniffing. Files above a configurable size threshold load through polars.
- **Replayable preprocessing pipeline** — normalize column names, type coercion with user
  overrides, missing-value strategies, IQR/z-score outlier flagging, deduplication, and filters,
  all modelled as an ordered step list.
- **Dataset profiler** — per-column role classification (numeric, categorical, datetime, boolean,
  high-cardinality text), cardinality, missingness, skew, and a correlation matrix. Profiles on a
  sample for large files.
- **Chart selection engine** — rules mapping data shape to candidate charts, plus a scoring layer
  ranking them by cardinality, missingness, skew, correlation strength, and readability. Rules and
  weights live in an editable `rules.yaml`. Every recommendation carries a human-readable
  justification.
- **Interactive charts** via Plotly in a `QWebEngineView`, with a matplotlib canvas fallback when
  WebEngine is unavailable.
- **Static export** to PNG, SVG, and PDF at configurable DPI through matplotlib.
- **Generated code export** — a standalone, runnable Python script reproducing load →
  preprocessing steps → chart, in matplotlib or Plotly flavour, with no `plotaviz` import.
- **Natural-language query bar** — free text becomes a validated chart spec; the interpreted spec
  is shown to the user and stays editable.
- **LLM layer** with pluggable Anthropic, OpenAI, Gemini, and Ollama providers. The model returns
  a structured chart spec, never executable code, and receives only schema, summary statistics,
  and sample rows. API keys are stored in the OS keyring.
- **Session save/load** to `.pviz` JSON files capturing source path and hash, type overrides,
  preprocessing steps, filters, chart spec, and view settings. Reopening warns if the source file
  changed.
- **Column type override panel** applied before analysis proceeds.
- **Filter panel** with per-column widgets and a raw pandas-query bar, applied live to the chart.
- **Performance guardrails** — automatic downsampling above ~100k points (LTTB for time series)
  with a visible sampling notice, top-N category capping, a virtualized preview table, and
  debounced re-render.
- **CLI mode** — `plotaviz --input data.csv --auto --export chart.png` for scripting and CI.
- Four sample datasets, each exercising a different chart path.

[Unreleased]: https://github.com/saadibnainan/plotaviz/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/saadibnainan/plotaviz/releases/tag/v0.1.0
