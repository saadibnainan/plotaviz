# PlotaViz v0.1.0 — full-stack & QA audit

Audit of `main` at `eeb24c1`, covering every module in `plotaviz/core` and `plotaviz/ui`, the CLI,
the test suite, and CI. Findings were confirmed by reproducing them (in-memory pandas runs,
pytest-qt, and a headless launch of the GUI on the bundled samples) or against upstream
documentation where noted.

**Status key:** ✅ fixed on this branch, with a regression test · ◐ partly fixed · ○ open
(recommendation only).
**Evidence key:** *Repro* reproduced · *Code* read directly in source · *Docs* upstream docs.

## Summary

The `core/` engine is well designed — the replayable step list, the single `ChartSpec` contract,
and the headless/Qt split are the right architecture and made most fixes small. The defects
clustered in two places:

1. **GUI wiring.** The UI layer had 2 tests, and the main workflow was broken in ways a single
   end-to-end run would show. Opening the bundled `sales_timeseries.csv` left the analysis with
   **0 rows** within half a second, with no user input (P0-1).
2. **Claims the code did not meet.** "Generated code cannot drift", "profile on a sample",
   "nothing leaves this machine" (Ollama), and the consent text were each untrue in specific,
   testable ways.

After the fixes: 432 tests pass (372 before), `ruff`, `ruff format --check` and `mypy plotaviz/core`
are clean, the CI smoke steps pass, and the generated script runs for every sample dataset in both
flavours.

## P0 — breaks the core workflow

| # | Defect | Where | Evidence | Status |
|---|---|---|---|---|
| 1 | **Date filter emptied the dataset.** `QDateEdit` defaults to 2000-01-01 with a minimum of 1752-09-14 ([Qt docs](https://doc.qt.io/qt-6/qdatetimeedit.html)); the pickers were never set to the data's range, so every date column always looked filtered to 2000-01-01. Worse, the numeric spin boxes connected their signals *before* setting initial values, so building the panel fired the debounce and applied that filter on load. A date-only end bound also excluded rows later that day. | `ui/filter_panel.py`, `core/profiler.py` | Repro, Docs | ✅ |
| 2 | **Any boolean column with a blank cell made the file fail to load.** `is_numeric_dtype(boolean)` is True, the median is 1.0, and `fillna(1.0)` raises on a boolean column → `PreprocessError` in GUI and CLI. | `core/preprocess.py` `FillMissing` | Repro | ✅ |
| 3 | **European numbers became dates.** In a `;`-CSV, `1,5` loads as text and `format="mixed"` parses 100% of such values as dates, so measures were coerced to timestamps. No decimal-comma support existed. | `core/profiler.py`, `core/loader.py` | Repro | ✅ |
| 4 | **Cancel left the UI stuck.** A cancelled worker emitted neither `finished` nor `failed`, so the progress bar/Cancel button stayed up and a cancelled NL query left the query bar disabled. | `ui/workers.py`, `ui/main_window.py` | Repro | ✅ |
| 5 | **Ctrl+S overwrote the wrong `.pviz`.** `session_path` was never cleared when a plain dataset was opened (and was set from the worker thread, even for a restore that was later cancelled). | `ui/main_window.py` | Repro | ✅ |
| 6 | **Every replay replaced the user's chart.** `setCurrentRow(0)` ran with signals live → `currentItemChanged` → `choose(recommendations[0])`; `Analysis.rerun` independently reset any rules-sourced spec. Loads also rendered twice. | `ui/main_window.py`, `core/analysis.py` | Repro | ✅ |

## P1 — wrong output or data-loss risk

| # | Defect | Where | Evidence | Status |
|---|---|---|---|---|
| 7 | **A rejected filter poisoned the pipeline.** `replace_filters()` mutated before `rerun()` raised, with no rollback; the empty result was also already assigned. Every later replay and the saved session inherited it. | `core/analysis.py` | Repro | ✅ full rollback; "No rows match the current filters." instead of a profiler error |
| 8 | **Filter changes ran the whole pipeline + profile + ranking on the UI thread**, and every render rewrites/reloads ~4 MB of HTML embedding plotly.js. | `ui/main_window.py`, `ui/interactive_view.py` | Code | ◐ replays now run on the worker and are queued, not dropped, when busy. Rendering is still on the UI thread; see *Performance* below |
| 9 | **Categorical filter listed only the top 10 values**; deselecting one silently dropped every unlisted category. Columns with >200 values get no filter (the promised search box does not exist). | `ui/filter_panel.py`, `core/profiler.py` | Code | ◐ all values up to 200 are listed; >200 still open |
| 10 | **"Other" bucketing crashed on `category` dtype** (what Parquet/polars return): `Cannot setitem on a Categorical with a new category`. | `core/plotter.py` | Repro | ✅ |
| 11 | **Generated code drifted from the app**: dates parsed without `format="mixed"` (3/3 vs 1/3 parsed on mixed input); boolean coercion `astype('boolean')` crashed on yes/no; `dropna()` also dropped colour-NaN rows; top-30 + "Other" capping absent; `FillMissing` ignored `columns`; non-UTF-8 and `;`-delimited decimal-comma files were re-read with defaults. | `core/preprocess.py`, `core/codegen.py`, `core/loader.py` | Repro | ✅ (the matplotlib treemap layout still differs from the app's) |
| 12 | **Type "Reset to detected" did not undo the coercion**, and after a session restore a previous override counted as "detected" and was dropped on the next Apply. | `ui/main_window.py`, `ui/type_override.py` | Repro | ✅ `Analysis.inferred_types` + `set_type_overrides()` |
| 13 | **Correlation heatmap / pair plot included identifier columns** (`order_id`); renderers disagreed (6 vs 5 pair-plot columns). | `core/plotter.py`, `core/selector.py` | Repro | ✅ measure columns pinned in the spec |
| 14 | **"Sample" of large files is the first N rows**, so a time-sorted file shows only its earliest period; JSON/Excel have no size guard. | `core/loader.py` | Code | ○ |
| 15 | **Export silently overwrote files.** Default code-export path for `sales.csv` is `sales.py` beside it — plausibly the user's own script. | `ui/export_dialog.py` | Code | ✅ confirmation before replacing |

## Security and privacy

| # | Finding | Evidence | Status |
|---|---|---|---|
| 16 | **Consent auto-ticked.** Selecting Ollama ticked the shared consent box; switching to a hosted provider kept it — consent the user never gave. Consent was also global across providers. | Code, Repro | ✅ per-provider (`llm/consent/<provider>`), never pre-ticked; the old global flag is removed, so users are asked again |
| 17 | **Remote Ollama bypassed consent.** `sends_data_off_machine = False` was hard-coded while the host is editable; a LAN or internet host received data over plain HTTP with no prompt, while the UI said "nothing leaves this machine". | Code | ✅ loopback detection (`localhost`, `127.0.0.0/8`, `::1`); anything else needs consent |
| 18 | **Session files carry pandas query expressions** that run through `DataFrame.query`. pandas' own docstring warns `eval` "can run arbitrary code which can make you vulnerable to code injection". A CVE was filed (CVE-2024-9880) but its status is disputed/rejected, so exploitability is **not** asserted here. | Docs | ✅ GUI shows the expressions and asks before applying ("Open without them" is the default) |
| 19 | **Prompt contents did not match the consent text.** The text said "up to 5 sample rows"; the prompt actually sent per-column example values, top values with counts, and min/max — including from identifier and free-text columns (names, e-mails). `from_question(sample_rows=…)` was dead code. | Code | ✅ values withheld for identifier/free-text columns; one shared `CONSENT_TEXT` states exactly what is sent; parameter wired through |
| 20 | **Unescaped title in the generated script's docstring.** A `"""` or `\U` in a (model-supplied) title produced a script that does not compile. Not code execution — the following `from __future__` line forces a compile error first. | Repro | ✅ escaped |
| 21 | **Startup could block ~2 s** on an Ollama HTTP probe (and a keyring read) on the UI thread. | Code | ✅ no network probe on the UI thread |

## Also fixed while testing

- **Short codes classified as dates.** dateutil reads `t13` as "day 13" and `M2` as "2 seconds", so
  a team/machine-code column became a datetime. Date detection now also requires a date-shaped
  string (two digit groups, a month name with a number, or `YYYYMMDD`).
- Numeric filter spin boxes round to their display precision, so an untouched maximum like
  `1.005` produced a filter excluding the top rows.
- Date filters on timezone-aware columns raised on comparison.
- `TaskRunner` could clear a newly started task's references when a stale "finished" arrived.
- JSON "object of arrays" (`{"a": [...], "b": [...]}`) loaded as one row of lists.
- `.xls` import error told users to install `openpyxl`; it needs `xlrd`.
- A `mypy plotaviz/core` error under current pandas-stubs.

## P2 — open, smaller

- `performance/max_points` is saved in Settings but never read.
- `plotaviz data.csv --export out.png` opens the GUI and ignores `--export`; `--dpi`/`--size` are
  not range-checked.
- HTML export uses `include_plotlyjs="cdn"`, so it needs internet — the README calls it
  self-contained and the app offline.
- Session `view` (renderer, tab, geometry) is saved but never restored; sessions store absolute
  paths with no "locate file" prompt; large files are hashed twice on session open.
- Filters restored from a session are applied but not reflected in the filter widgets.
- `_cap_categories` ranks by sum even when the aggregation is mean.
- `log_y` is suggested on skew without checking `min > 0`, so zero values vanish.
- Median imputation is on by default with no "leave as-is" option — it fabricates values in time
  series. Consider making "leave missing" the default.
- pandas has no upper bound (pandas 3 is current; the suite passes on 3.1). Hard-coded LLM model
  lists are stale.

## Performance

1. Cache the cleaned frame *before* filters; a filter change should replay only the filter steps,
   not CoerceTypes → Dedupe → FillMissing → FlagOutliers, and should not re-rank unless asked.
2. Load plotly.js once into a persistent WebEngine page and update via `Plotly.react` instead of
   writing and loading a ~4 MB HTML file per render.
3. Build figures off the UI thread (`prepare()` + figure construction are pure).
4. Replace first-N "sampling" with a random or stratified sample (or push aggregation into polars
   so charts reflect the full file).

## Feature proposals (ranked by value ÷ effort)

1. **Manual chart builder that works offline.** The editable chart/x/y/colour/agg row already
   exists in the query bar but is hidden until an LLM answers, so offline users can only pick from
   ≤8 presets. Show it always.
2. **Import options dialog** — delimiter, encoding, decimal/thousands, day-first dates, header row,
   Excel sheet picker. `LoadResult.read_options` (added here) already carries these into codegen.
3. **Cleaning controls in the GUI** — missing-value strategy (plus "leave as-is"), outliers
   flag/drop/off, dedupe on/off. These are already `Step` parameters and CLI flags.
4. **Editable step list with undo/redo** — the pipeline design supports it directly.
5. **Chart customization** — title, axis labels, log scales, top-N, sort, bins, colour-blind-safe
   palette, all as `spec.options` so sessions and codegen pick them up for free.
6. **Session quality-of-life** — relative paths, "locate missing file", unsaved-changes prompt,
   restore the saved view, reflect restored filters in the widgets.
7. **LLM transparency** — "preview what will be sent", a real key check (list models; sends no
   data), model lists fetched from the provider.
8. **Preview table** — sort, search, per-column stats panel.
9. **Roadmap items** — EDA report export, multi-chart dashboard, dataset join, dark mode.
10. **CLI** — `--rules custom.yaml`, `--sheet`, batch mode over a glob, offline HTML.

## Test coverage

Added `tests/test_audit_regressions.py` (core) and `tests/test_ui_regressions.py` (pytest-qt)
— 60 tests, each reproducing one finding above. The UI tests redirect `QSettings` to a temp
directory. Remaining gaps worth closing:

- NL query bar, recommendations list, and export flow have no end-to-end test.
- CI runs generated code only for the iris sample; run it for every chart type × flavour and
  compare the plotted rows with `prepare()` (the new tests do this for the drift cases).
- No test exercises the polars large-file path with real sampling.
