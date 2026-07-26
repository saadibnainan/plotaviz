"""The main window — layout, menus, and the wiring between the UI and ``core``.

This module holds no analysis logic. Every decision about how to clean, profile, rank, or draw
lives in :mod:`plotaviz.core`; the window's job is to move data between widgets and that engine,
run anything slow on a worker thread, and turn :class:`~plotaviz.core.errors.PlotaVizError` into
a dialog a person can act on. No traceback ever reaches the user.

Layout, left to right: the data preview and cleaning report on the left, the chart in the middle
under the natural-language bar, and the recommendations, type overrides, and filters on the
right.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings, Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableView,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import __app_name__, __version__
from ..core import loader
from ..core.analysis import Analysis
from ..core.errors import PlotaVizError
from ..core.llm import LLMAssistant, get_provider
from ..core.session import SESSION_SUFFIX, Session
from ..core.spec import ChartSpec
from .export_dialog import ExportDialog
from .filter_panel import FilterPanel
from .interactive_view import ChartView
from .nl_query_bar import NLQueryBar
from .settings_dialog import SettingsDialog
from .table_model import DataFrameModel
from .type_override import TypeOverridePanel
from .workers import TaskRunner

#: How many recent files to remember.
MAX_RECENT = 8

#: File dialog filter covering everything the loader understands.
_OPEN_FILTER = (
    "Data files (*.csv *.tsv *.txt *.xlsx *.xls *.json *.ndjson *.jsonl *.parquet);;"
    "CSV (*.csv *.tsv *.txt);;Excel (*.xlsx *.xls);;JSON (*.json *.ndjson *.jsonl);;"
    "Parquet (*.parquet);;All files (*)"
)


class MainWindow(QMainWindow):
    """PlotaViz's main window.

    Args:
        parent: Parent widget.
        initial_file: Dataset or ``.pviz`` session to open on launch.
    """

    def __init__(
        self, parent: QWidget | None = None, *, initial_file: str | Path | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(__app_name__)
        self.resize(1500, 940)
        self.setAcceptDrops(True)

        self.settings = QSettings("plotaviz", "plotaviz")
        self.analysis: Analysis | None = None
        self.session_path: Path | None = None
        self.runner = TaskRunner(self)

        self._build_central()
        self._build_docks()
        self._build_menus()
        self._build_status_bar()
        self._restore_geometry()
        self._refresh_llm_state()

        if initial_file:
            self.open_path(initial_file)

    # ------------------------------------------------------------------ construction

    def _build_central(self) -> None:
        """Chart area with the natural-language bar above it."""
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.nl_bar = NLQueryBar(central)
        self.nl_bar.question_asked.connect(self._on_question)
        self.nl_bar.spec_edited.connect(self._on_spec_edited)
        layout.addWidget(self.nl_bar)

        prefer_static = self.settings.value("performance/prefer_static", False, type=bool)
        self.chart_view = ChartView(central, prefer_static=prefer_static)
        layout.addWidget(self.chart_view, stretch=1)

        self.chart_note = QLabel("", central)
        self.chart_note.setWordWrap(True)
        self.chart_note.setStyleSheet("color: palette(mid); font-size: 11px; padding: 4px 10px;")
        self.chart_note.hide()
        layout.addWidget(self.chart_note)

        self.setCentralWidget(central)

    def _build_docks(self) -> None:
        """The left data dock and the right analysis dock."""
        # --- left: preview + cleaning report
        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)

        self.preview = QTableView(left)
        self.preview_model = DataFrameModel(parent=self)
        self.preview.setModel(self.preview_model)
        self.preview.setAlternatingRowColors(True)
        self.preview.setSelectionBehavior(QTableView.SelectionBehavior.SelectColumns)
        self.preview.horizontalHeader().setStretchLastSection(True)
        self.preview.setSortingEnabled(False)

        self.preview_status = QLabel("No data loaded.", left)
        self.preview_status.setStyleSheet("color: palette(mid); font-size: 11px;")

        self.cleaning_report = QTextEdit(left)
        self.cleaning_report.setReadOnly(True)
        self.cleaning_report.setMaximumHeight(190)
        self.cleaning_report.setPlaceholderText("The cleaning report appears here after loading.")

        data_tabs = QTabWidget(left)
        preview_page = QWidget(data_tabs)
        preview_layout = QVBoxLayout(preview_page)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(self.preview)
        preview_layout.addWidget(self.preview_status)
        data_tabs.addTab(preview_page, "Preview")
        data_tabs.addTab(self.cleaning_report, "Cleaning report")

        left_layout.addWidget(data_tabs)

        self.data_dock = QDockWidget("Data", self)
        self.data_dock.setWidget(left)
        self.data_dock.setObjectName("data_dock")
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.data_dock)

        # --- right: recommendations, types, filters
        self.recommendations = QListWidget(self)
        self.recommendations.itemActivated.connect(self._on_recommendation_chosen)
        self.recommendations.currentItemChanged.connect(self._on_recommendation_chosen)

        recommend_page = QWidget(self)
        recommend_layout = QVBoxLayout(recommend_page)
        recommend_layout.setContentsMargins(6, 6, 6, 6)
        caption = QLabel(
            "Ranked by how well each chart suits this data. Select one to switch.", recommend_page
        )
        caption.setWordWrap(True)
        caption.setStyleSheet("color: palette(mid); font-size: 11px;")
        recommend_layout.addWidget(caption)
        recommend_layout.addWidget(self.recommendations, stretch=1)

        self.why_label = QTextEdit(recommend_page)
        self.why_label.setReadOnly(True)
        self.why_label.setMaximumHeight(140)
        self.why_label.setPlaceholderText("The reasoning behind the selected chart appears here.")
        recommend_layout.addWidget(self.why_label)

        self.refine_button = QPushButton("Ask the model for a second opinion", recommend_page)
        self.refine_button.clicked.connect(self._on_refine)
        recommend_layout.addWidget(self.refine_button)

        self.type_panel = TypeOverridePanel(self)
        self.type_panel.types_changed.connect(self._on_types_changed)

        self.filter_panel = FilterPanel(self)
        self.filter_panel.filters_changed.connect(self._on_filters_changed)

        self.analysis_tabs = QTabWidget(self)
        self.analysis_tabs.addTab(recommend_page, "Charts")
        self.analysis_tabs.addTab(self.type_panel, "Types")
        self.analysis_tabs.addTab(self.filter_panel, "Filters")

        self.analysis_dock = QDockWidget("Analysis", self)
        self.analysis_dock.setWidget(self.analysis_tabs)
        self.analysis_dock.setObjectName("analysis_dock")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.analysis_dock)

        self.resizeDocks(
            [self.data_dock, self.analysis_dock], [430, 400], Qt.Orientation.Horizontal
        )

    def _build_menus(self) -> None:
        """File, View, and Help menus."""
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("&Open dataset…", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.choose_file)
        file_menu.addAction(open_action)

        self.recent_menu = file_menu.addMenu("Open &recent")
        self._rebuild_recent_menu()

        file_menu.addSeparator()

        open_session = QAction("Open &session…", self)
        open_session.triggered.connect(self.choose_session)
        file_menu.addAction(open_session)

        self.save_session_action = QAction("&Save session", self)
        self.save_session_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_session_action.triggered.connect(self.save_session)
        self.save_session_action.setEnabled(False)
        file_menu.addAction(self.save_session_action)

        self.save_session_as_action = QAction("Save session &as…", self)
        self.save_session_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.save_session_as_action.triggered.connect(lambda: self.save_session(ask=True))
        self.save_session_as_action.setEnabled(False)
        file_menu.addAction(self.save_session_as_action)

        file_menu.addSeparator()

        self.export_action = QAction("&Export…", self)
        self.export_action.setShortcut("Ctrl+E")
        self.export_action.triggered.connect(self.export)
        self.export_action.setEnabled(False)
        file_menu.addAction(self.export_action)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.data_dock.toggleViewAction())
        view_menu.addAction(self.analysis_dock.toggleViewAction())
        view_menu.addSeparator()
        settings_action = QAction("&Settings…", self)
        settings_action.setShortcut(QKeySequence.StandardKey.Preferences)
        settings_action.triggered.connect(self.open_settings)
        view_menu.addAction(settings_action)

        help_menu = self.menuBar().addMenu("&Help")
        docs_action = QAction("Documentation", self)
        docs_action.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://github.com/saadibnainan/plotaviz"))
        )
        help_menu.addAction(docs_action)
        about_action = QAction(f"About {__app_name__}", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _build_status_bar(self) -> None:
        """Status message, progress bar, and cancel button."""
        self.progress = QProgressBar(self)
        self.progress.setMaximumWidth(200)
        self.progress.setTextVisible(False)
        self.progress.hide()

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self._cancel_task)
        self.cancel_button.hide()

        container = QWidget(self)
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.progress)
        row.addWidget(self.cancel_button)

        self.statusBar().addPermanentWidget(container)
        self.statusBar().showMessage("Ready. Open a dataset to begin.")

    # ------------------------------------------------------------------ opening files

    def choose_file(self) -> None:
        """Show the open dialog."""
        start = str(self.settings.value("paths/last_dir", str(Path.home())))
        path, _ = QFileDialog.getOpenFileName(self, "Open dataset", start, _OPEN_FILTER)
        if path:
            self.open_path(path)

    def choose_session(self) -> None:
        """Show the open dialog for session files."""
        start = str(self.settings.value("paths/last_dir", str(Path.home())))
        path, _ = QFileDialog.getOpenFileName(
            self, "Open session", start, f"PlotaViz sessions (*{SESSION_SUFFIX})"
        )
        if path:
            self.open_path(path)

    def open_path(self, path: str | Path) -> None:
        """Open a dataset or a session file, on a worker thread."""
        path = Path(path).expanduser()
        if not path.exists():
            self._error("File not found", f"There is no file at {path}.")
            return

        self.settings.setValue("paths/last_dir", str(path.parent))
        if path.suffix.lower() == SESSION_SUFFIX:
            self._run(self._load_session_task, path, label=f"Restoring {path.name}…")
        else:
            self._run(self._load_dataset_task, path, label=f"Loading {path.name}…")

    def _load_dataset_task(self, path: Path, progress: Any = None) -> tuple[Analysis, str | None]:
        """Worker body: load and clean a dataset."""
        if progress:
            progress(15, "Reading the file…")
        analysis = Analysis.from_file(
            path,
            large_file_mb=float(self.settings.value("performance/large_file_mb", 500)),
        )
        if progress:
            progress(90, "Choosing a chart…")
        return analysis, None

    def _load_session_task(self, path: Path, progress: Any = None) -> tuple[Analysis, str | None]:
        """Worker body: restore a session."""
        if progress:
            progress(10, "Reading the session…")
        session = Session.load(path)
        if progress:
            progress(40, "Replaying the cleaning steps…")
        analysis, warning = Analysis.from_session(session)
        self.session_path = path
        return analysis, warning

    def _on_loaded(self, result: tuple[Analysis, str | None]) -> None:
        """Populate every panel once loading finishes."""
        analysis, warning = result
        self.analysis = analysis

        self.preview_model.set_frame(analysis.df, analysis.profile)
        self.preview_status.setText(self.preview_model.status_text())
        self.cleaning_report.setPlainText(
            (analysis.result.cleaning_report() if analysis.result else "")
            + ("\n\n" + "\n".join(analysis.load_result.notes) if analysis.load_result.notes else "")
        )

        if analysis.profile:
            self.type_panel.set_profile(analysis.profile, analysis.type_overrides)
            self.filter_panel.set_profile(analysis.profile)
        self.nl_bar.set_columns(list(analysis.df.columns))

        self._populate_recommendations()
        self._render_current()

        name = analysis.load_result.path.name
        self.setWindowTitle(f"{name} — {__app_name__}")
        self.statusBar().showMessage(analysis.load_result.summary())
        self._remember_recent(analysis.load_result.path)

        for action in (self.export_action, self.save_session_action, self.save_session_as_action):
            action.setEnabled(True)

        if warning:
            self._warn("Session restored with differences", warning)

    # ------------------------------------------------------------------ charting

    def _populate_recommendations(self) -> None:
        """Fill the ranked list from the analysis."""
        self.recommendations.blockSignals(True)
        self.recommendations.clear()

        if self.analysis:
            for spec in self.analysis.recommendations:
                mapping = " · ".join(
                    part
                    for part in (
                        f"x: {spec.x}" if spec.x else "",
                        f"y: {spec.y}" if spec.y else "",
                        f"colour: {spec.color}" if spec.color else "",
                        f"{spec.agg}" if spec.agg else "",
                    )
                    if part
                )
                item = QListWidgetItem(f"{spec.chart}   ({spec.score:.0%})\n{mapping}")
                item.setData(Qt.ItemDataRole.UserRole, spec)
                item.setToolTip(spec.why)
                self.recommendations.addItem(item)

        self.recommendations.blockSignals(False)
        if self.recommendations.count():
            self.recommendations.setCurrentRow(0)
            self._show_why(self.analysis.spec if self.analysis else None)

    def _on_recommendation_chosen(self, item: QListWidgetItem | None, *_: object) -> None:
        """Switch to a chart the user picked from the ranked list."""
        if item is None or self.analysis is None:
            return
        spec = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(spec, ChartSpec):
            return
        try:
            self.analysis.choose(spec)
        except PlotaVizError as exc:
            self._error("That chart does not fit this data", str(exc))
            return
        self._show_why(spec)
        self._render_current()

    def _on_spec_edited(self, spec: ChartSpec) -> None:
        """Redraw after the user tweaks the interpreted spec in the query bar."""
        if self.analysis is None:
            return
        try:
            self.analysis.choose(spec)
        except PlotaVizError as exc:
            self.nl_bar.show_status(str(exc), error=True)
            return
        self._show_why(spec)
        self._render_current()

    def _show_why(self, spec: ChartSpec | None) -> None:
        """Display the justification for the current chart."""
        self.why_label.setPlainText(spec.why if spec and spec.why else "")

    def _render_current(self) -> None:
        """Draw the active chart, surfacing any preparation notes."""
        if self.analysis is None or self.analysis.spec is None:
            self.chart_view.show_message("No chart is selected yet.")
            return
        try:
            notes = self.chart_view.render_spec(self.analysis.df, self.analysis.spec)
        except PlotaVizError as exc:
            self.chart_view.show_message(str(exc))
            self.statusBar().showMessage("The chart could not be drawn.")
            return

        if notes:
            self.chart_note.setText("  ".join(notes))
            self.chart_note.show()
        else:
            self.chart_note.hide()
        self.statusBar().showMessage(self.chart_view.renderer_note())

    # ------------------------------------------------------------------ panel callbacks

    def _on_types_changed(self, overrides: dict[str, str]) -> None:
        """Re-run the pipeline with corrected column types."""
        if self.analysis is None:
            return
        self.analysis.type_overrides = dict(overrides)
        for column, role in overrides.items():
            for step in self.analysis.pipeline:
                if hasattr(step, "types"):
                    step.types[column] = role
                    break
        self._rerun("Applying types…")

    def _on_filters_changed(self, filters: list[Any]) -> None:
        """Re-run the pipeline with new filters."""
        if self.analysis is None:
            return
        try:
            self.analysis.set_filters(list(filters))
        except PlotaVizError as exc:
            self.filter_panel.show_query_error(str(exc))
            return
        self._after_rerun()

    def _rerun(self, label: str) -> None:
        """Replay the pipeline on a worker thread."""
        if self.analysis is None:
            return
        self._run(self._rerun_task, label=label, on_finished=lambda _: self._after_rerun())

    def _rerun_task(self, progress: Any = None) -> None:
        """Worker body: replay the pipeline."""
        if progress:
            progress(30, "Replaying the cleaning steps…")
        if self.analysis:
            self.analysis.rerun()

    def _after_rerun(self) -> None:
        """Refresh every panel after a pipeline replay."""
        if self.analysis is None:
            return
        self.preview_model.set_frame(self.analysis.df, self.analysis.profile)
        self.preview_status.setText(self.preview_model.status_text())
        if self.analysis.result:
            self.cleaning_report.setPlainText(self.analysis.result.cleaning_report())
        self.nl_bar.set_columns(list(self.analysis.df.columns))
        self._populate_recommendations()
        self._render_current()

    # ------------------------------------------------------------------ LLM

    def _assistant(self) -> LLMAssistant:
        """Build an assistant from the current settings."""
        name = str(self.settings.value("llm/provider", "") or "")
        if not name:
            return LLMAssistant(None)
        kwargs: dict[str, Any] = {}
        model = str(self.settings.value("llm/model", "") or "")
        if model:
            kwargs["model"] = model
        if name == "ollama":
            kwargs["host"] = str(self.settings.value("llm/ollama_host", "http://localhost:11434"))
        try:
            provider = get_provider(name, **kwargs)
        except PlotaVizError:
            return LLMAssistant(None)
        return LLMAssistant(provider, consent=self.settings.value("llm/consent", False, type=bool))

    def _refresh_llm_state(self) -> None:
        """Enable or disable the query bar to match the configured provider."""
        assistant = self._assistant()
        self.nl_bar.set_enabled_for_provider(
            assistant.available, assistant.provider.name if assistant.provider else ""
        )
        self.refine_button.setEnabled(assistant.available)

    def _on_question(self, question: str) -> None:
        """Turn a natural-language question into a chart."""
        if self.analysis is None or self.analysis.profile is None:
            self.nl_bar.show_status("Open a dataset first.", error=True)
            return

        assistant = self._assistant()
        profile = self.analysis.profile
        self.nl_bar.set_busy(True)

        def task(progress: Any = None) -> Any:
            if progress:
                progress(50, "Asking the model…")
            return assistant.from_question(profile, question)

        def done(result: Any) -> None:
            self.nl_bar.set_busy(False)
            spec = result.spec
            try:
                self.analysis.choose(spec) if self.analysis else None
            except PlotaVizError as exc:
                self.nl_bar.show_status(str(exc), error=True)
                return
            self.nl_bar.show_spec(spec)
            self._show_why(spec)
            self._render_current()

        def failed(message: str, _trace: str) -> None:
            self.nl_bar.set_busy(False)
            self.nl_bar.show_status(message, error=True)

        self._run(task, label="Asking the model…", on_finished=done, on_failed=failed)

    def _on_refine(self) -> None:
        """Ask the model to break a tie between close candidates."""
        if self.analysis is None or self.analysis.profile is None:
            return
        assistant = self._assistant()
        profile = self.analysis.profile
        candidates = list(self.analysis.recommendations)

        def task(progress: Any = None) -> ChartSpec:
            if progress:
                progress(50, "Asking the model…")
            return assistant.refine(profile, candidates)

        def done(spec: ChartSpec) -> None:
            if self.analysis is None:
                return
            try:
                self.analysis.choose(spec)
            except PlotaVizError as exc:
                self._error("The model suggested a chart that does not fit", str(exc))
                return
            self._show_why(spec)
            self._render_current()

        self._run(task, label="Asking the model…", on_finished=done)

    # ------------------------------------------------------------------ export / session

    def export(self) -> None:
        """Export the chart as an image, an HTML page, or a Python script."""
        if self.analysis is None or self.analysis.spec is None:
            return

        source = self.analysis.load_result.path
        dialog = ExportDialog(self, default_name=source.stem or "chart", start_dir=source.parent)
        if dialog.exec() != ExportDialog.DialogCode.Accepted:
            return

        target = dialog.path()
        mode = dialog.mode
        analysis = self.analysis

        def task(progress: Any = None) -> Path:
            if progress:
                progress(40, f"Writing {target.name}…")
            if mode == "code":
                from ..core import codegen

                return codegen.write(target, analysis.generate_code(flavour=dialog.flavour()))
            if mode == "html":
                from ..core import exporter

                return exporter.export_html(analysis.df, analysis.spec, target)  # type: ignore[arg-type]
            from ..core import exporter

            return exporter.export_image(
                analysis.df,
                analysis.spec,
                target,
                options=dialog.options(),  # type: ignore[arg-type]
            )

        def done(written: Path) -> None:
            self.statusBar().showMessage(f"Saved {written}")
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Information)
            box.setWindowTitle("Export complete")
            box.setText(f"Saved {written.name}")
            box.setInformativeText(str(written))
            reveal = box.addButton("Show in folder", QMessageBox.ButtonRole.ActionRole)
            box.addButton(QMessageBox.StandardButton.Ok)
            box.exec()
            if box.clickedButton() is reveal:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(written.parent)))

        self._run(task, label=f"Exporting {target.name}…", on_finished=done)

    def save_session(self, *, ask: bool = False) -> None:
        """Save the current state to a ``.pviz`` file."""
        if self.analysis is None:
            return

        target = self.session_path
        if target is None or ask:
            suggested = self.analysis.load_result.path.with_suffix(SESSION_SUFFIX)
            chosen, _ = QFileDialog.getSaveFileName(
                self, "Save session", str(suggested), f"PlotaViz sessions (*{SESSION_SUFFIX})"
            )
            if not chosen:
                return
            target = Path(chosen)

        view = {
            "renderer": self.chart_view.renderer,
            "active_tab": self.analysis_tabs.currentIndex(),
            "geometry": bytes(self.saveGeometry().toBase64().data()).decode(),
        }
        try:
            written = self.analysis.to_session(view).save(target)
        except PlotaVizError as exc:
            self._error("Could not save the session", str(exc))
            return

        self.session_path = written
        self.statusBar().showMessage(f"Session saved to {written}")

    # ------------------------------------------------------------------ settings / about

    def open_settings(self) -> None:
        """Show the settings dialog and apply anything that changed."""
        dialog = SettingsDialog(self, settings=self.settings)
        if dialog.exec() == SettingsDialog.DialogCode.Accepted:
            self._refresh_llm_state()
            self.chart_view.set_prefer_static(
                self.settings.value("performance/prefer_static", False, type=bool)
            )
            self._render_current()

    def show_about(self) -> None:
        """Show version and licensing information."""
        QMessageBox.about(
            self,
            f"About {__app_name__}",
            f"<h3>{__app_name__} {__version__}</h3>"
            "<p>Automatic data analytics and visualization.</p>"
            "<p>PlotaViz is MIT licensed. It uses Qt via PySide6, which is licensed under the "
            "LGPLv3 and is dynamically linked; see THIRD_PARTY_LICENSES for attribution. "
            "Charts are drawn with Plotly and matplotlib.</p>"
            f"<p>Chart renderer in use: <b>{self.chart_view.renderer}</b>.</p>",
        )

    # ------------------------------------------------------------------ recent files

    def _remember_recent(self, path: Path) -> None:
        """Push a path onto the recent-files list."""
        recent = [str(p) for p in self.settings.value("paths/recent", [], type=list)]
        entry = str(path)
        recent = [entry] + [p for p in recent if p != entry]
        self.settings.setValue("paths/recent", recent[:MAX_RECENT])
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        """Refresh the Open Recent submenu."""
        self.recent_menu.clear()
        recent = [str(p) for p in self.settings.value("paths/recent", [], type=list)]
        if not recent:
            empty = QAction("Nothing yet", self)
            empty.setEnabled(False)
            self.recent_menu.addAction(empty)
            return
        for entry in recent:
            action = QAction(Path(entry).name, self)
            action.setToolTip(entry)
            action.triggered.connect(lambda _=False, p=entry: self.open_path(p))
            self.recent_menu.addAction(action)
        self.recent_menu.addSeparator()
        clear = QAction("Clear list", self)
        clear.triggered.connect(
            lambda: (self.settings.setValue("paths/recent", []), self._rebuild_recent_menu())
        )
        self.recent_menu.addAction(clear)

    # ------------------------------------------------------------------ drag and drop

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept a drag carrying one file PlotaViz can open."""
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if len(urls) == 1:
            suffix = Path(urls[0].toLocalFile()).suffix.lower()
            if suffix in loader.SUPPORTED_SUFFIXES or suffix == SESSION_SUFFIX:
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        """Open a dropped file."""
        urls = event.mimeData().urls()
        if urls:
            self.open_path(urls[0].toLocalFile())
            event.acceptProposedAction()

    # ------------------------------------------------------------------ task plumbing

    def _run(
        self,
        fn: Any,
        *args: Any,
        label: str = "Working…",
        on_finished: Any = None,
        on_failed: Any = None,
    ) -> None:
        """Run a callable on the worker thread with progress and cancellation."""
        if self.runner.busy:
            self.statusBar().showMessage("Still finishing the previous task…")
            return

        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.show()
        self.cancel_button.show()
        self.statusBar().showMessage(label)

        def finished(result: Any) -> None:
            self._end_task()
            handler = on_finished or self._on_loaded
            handler(result)

        def failed(message: str, trace: str) -> None:
            self._end_task()
            if on_failed:
                on_failed(message, trace)
            else:
                self._error("Something went wrong", message, detail=trace)

        def progressed(percent: int, message: str) -> None:
            self.progress.setValue(percent)
            if message:
                self.statusBar().showMessage(message)

        self.runner.start(
            fn,
            *args,
            on_finished=finished,
            on_failed=failed,
            on_progress=progressed,
        )

    def _end_task(self) -> None:
        """Hide the progress widgets."""
        self.progress.hide()
        self.cancel_button.hide()

    def _cancel_task(self) -> None:
        """Ask the running task to stop."""
        self.runner.cancel()
        self.statusBar().showMessage("Cancelling…")

    # ------------------------------------------------------------------ dialogs

    def _error(self, title: str, message: str, *, detail: str = "") -> None:
        """Show an error the user can act on. Tracebacks stay behind "Show details"."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(title)
        parts = message.split("\n\n", 1)
        box.setText(parts[0])
        if len(parts) > 1:
            box.setInformativeText(parts[1])
        if detail:
            box.setDetailedText(detail)
        box.exec()

    def _warn(self, title: str, message: str) -> None:
        """Show a non-blocking advisory."""
        QMessageBox.information(self, title, message)

    # ------------------------------------------------------------------ window state

    def _restore_geometry(self) -> None:
        """Restore window size and dock layout from the last session."""
        geometry = self.settings.value("window/geometry")
        state = self.settings.value("window/state")
        if geometry:
            self.restoreGeometry(geometry)
        if state:
            self.restoreState(state)

    def closeEvent(self, event: Any) -> None:
        """Persist window state, stop background work, and clean up temp files."""
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState())
        self.settings.sync()
        self.runner.cancel()
        self.runner.wait()
        self.chart_view.cleanup()
        super().closeEvent(event)


def launch(initial_file: str | Path | None = None) -> int:
    """Create the application and show the window.

    Args:
        initial_file: Dataset or session to open on startup.

    Returns:
        The Qt exit code.
    """
    app = QApplication.instance() or QApplication([])
    app.setApplicationName(__app_name__)
    app.setApplicationDisplayName(__app_name__)
    app.setOrganizationName("plotaviz")

    window = MainWindow(initial_file=initial_file)
    window.show()
    return int(app.exec())
