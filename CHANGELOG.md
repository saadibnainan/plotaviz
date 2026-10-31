# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Fixes from the full-stack and QA audit, and the first features it recommended — see
[docs/AUDIT.md](docs/AUDIT.md) for the details and the open recommendations.

### Added

- **Chart builder without an LLM**: the chart/x/y/colour/aggregation row is always available.
- **File → Open with options…**: delimiter, encoding, decimal mark, thousands separator, header
  line, and Excel sheet — saved in sessions and replayed by generated scripts.
- **Cleaning tab**: missing values, outliers (off/flag/drop, method, threshold), duplicate
  removal, and the full step list with "Remove step".
- **Edit → Undo / Redo** for filters, type overrides, cleaning, and removed steps.
- **Customize** panel: title, axis labels, log scales, category limit, sort order, histogram
  bins, and a colour-blind-safe (Okabe–Ito) palette — honoured by sessions and generated code.

### Changed

- **Missing values are no longer filled by default** (GUI, Python API, and CLI `--missing`
  default `keep`). Choose a strategy in the Cleaning tab or with `--missing median` etc.
- HTML export embeds plotly.js, so exported pages work offline.
- `plotaviz data.csv --export out.png` (and other output flags) run headless.
- "Other" categories are ranked by the charted aggregation; a log axis is only suggested for
  positive data; pandas is bounded below 4; LLM model suggestions refreshed.

### Fixed

- **Date filters no longer empty the dataset.** Date pickers start at the column's real range;
  previously they sat at 2000-01-01 and, because building the panel fired a filter pass, opening
  any dataset with a date column could leave zero rows. The end date is now inclusive.
- A yes/no column with a blank cell no longer makes the whole file fail to load.
- European-format numbers (`1,5`, `1.234,56`) in semicolon CSVs are read as numbers instead of
  being classified as dates; short codes such as `t13` or `M2` are no longer read as dates either.
- Cancelling a task hides the progress bar and re-enables the query bar.
- Opening a dataset after a session no longer makes Ctrl+S overwrite that session file.
- Filtering, changing a type, or any other replay keeps the chart you chose instead of switching
  back to the top recommendation; loads no longer render twice.
- A rejected filter is rolled back completely instead of breaking every later replay; a filter
  that matches nothing says so.
- Filter and type changes run off the UI thread and are queued, not dropped, while busy.
- The category filter lists every value (up to 200), not just the ten most frequent.
- "Other" bucketing works on `category` columns (Parquet/polars).
- "Reset to detected" really restores the inferred type; restored overrides survive the next Apply.
- Correlation heatmaps and pair plots exclude identifier columns; both renderers use 5 pair-plot
  columns.
- Generated scripts now match the app: mixed-format dates, yes/no booleans, category casts,
  colour-NaN rows, top-N + "Other" capping, `FillMissing` columns, and the file's delimiter,
  encoding, and decimal format.
- Export asks before replacing an existing file.
- JSON "object of arrays" files load as a table; the `.xls` error names `xlrd`.
- Restored sessions show their filters in the filter widgets and restore the active tab; a session
  whose filters match nothing in changed data still opens (without them).
- Decimal-comma detection no longer turns dotted dates into integers or reads US `1,250` as 1.25.
- Static chart re-renders no longer leak matplotlib figures.

- Static (matplotlib) view no longer stacks an extra navigation toolbar above the chart on
  every re-render.
- Interactive charts with a legend (colour splits, pies) no longer draw the title over the
  legend; the title gets its own row at the top of the figure.

### Security

- Consent to send data to an LLM provider is per provider and never pre-ticked. The previous
  global setting is discarded, so you are asked again.
- An Ollama host that is not on this machine requires consent like any remote provider.
- Values from identifier and free-text columns are withheld from LLM prompts, and the consent
  text now states exactly what is sent.
- Opening a session that contains query filters shows the expressions and asks before pandas
  evaluates them.
- Titles are escaped in generated scripts.

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
