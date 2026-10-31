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

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QByteArray, QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QGuiApplication,
    QKeySequence,
)
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableView,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import __app_name__, __version__
from ..core import loader
from ..core.analysis import Analysis
from ..core.errors import PlotaVizError
from ..core.llm import LLMAssistant, OllamaProvider, get_provider
from ..core.plotter import MAX_POINTS
from ..core.preprocess import QueryFilter
from ..core.report import export_report
from ..core.session import SESSION_SUFFIX, Session
from ..core.spec import ChartSpec
from .cleaning_panel import CleaningPanel
from .customize_panel import CustomizePanel, apply_customization
from .dashboard_view import DashboardView, build_tiles
from .export_dialog import ExportDialog
from .filter_panel import FilterPanel
from .import_dialog import ImportDialog
from .interactive_view import ChartView, RenderResult, build_render
from .join_dialog import JoinDialog
from .nl_query_bar import NLQueryBar
from .settings_dialog import SettingsDialog
from .table_model import DataFrameModel, column_summary_html
from .theme import apply_theme
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
        # True once anything changed since the last load or save; drives the close/open prompt.
        self._dirty = False
        #: Ask before discarding unsaved changes. Automated tests switch this off: a modal
        #: dialog during teardown would block a headless run.
        self.ask_before_discarding = True
        self._session_view: dict[str, Any] = {}
        self.runner = TaskRunner(self)
        # Charts and dashboard tiles are built on their own workers so drawing never blocks the
        # window or waits behind a replay. Only the newest request matters: a request made while
        # one is running sets the "again" flag and is redone when the worker goes idle.
        self.render_runner = TaskRunner(self)
        self.render_runner.idle.connect(self._render_if_requested)
        self._render_again = False
        self.tile_runner = TaskRunner(self)
        self.tile_runner.idle.connect(self._tiles_if_requested)
        self._tiles_again = False
        # Bumped whenever what should be on screen changes; a worker result built for an older
        # generation is dropped instead of drawn over the newer state.
        self._render_generation = 0
        self._tiles_generation = 0
        # Work requested while another task was running, in order. Entries with the same key
        # replace each other — a burst of filter edits needs one replay with the final state —
        # but different kinds of change (a type override, then a filter) are all kept.
        self._pending: list[tuple[str | None, Callable[[], None]]] = []
        self.runner.idle.connect(self._drain_pending)

        self._build_central()
        self._build_docks()
        self._build_menus()
        self._build_status_bar()
        self._dark = False
        self._apply_theme()
        hints = QGuiApplication.styleHints()
        if hasattr(hints, "colorSchemeChanged"):
            hints.colorSchemeChanged.connect(self._on_system_scheme_changed)
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
        self.chart_view.static_fallback_needed.connect(lambda: self._render_current(changed=False))
        layout.addWidget(self.chart_view, stretch=1)

        self.chart_note = QLabel("", central)
        self.chart_note.setWordWrap(True)
        self.chart_note.setStyleSheet("color: palette(mid); font-size: 11px; padding: 4px 10px;")
        self.chart_note.hide()
        layout.addWidget(self.chart_note)

        # Chart | Dashboard. The dashboard only redraws while it is visible (or when it is next
        # shown), so filtering stays as fast as before when the user is on the Chart tab.
        self.dashboard_view = DashboardView(self)
        self.dashboard_view.open_requested.connect(self._open_tile)
        self.dashboard_view.remove_requested.connect(self._remove_tile)
        self.dashboard_view.move_requested.connect(self._move_tile)
        self.dashboard_view.export_requested.connect(lambda: self.export(scope="dashboard"))
        self.dashboard_view.columns_changed.connect(lambda _: self._refresh_dashboard(force=True))
        self._dashboard_stale = False
        self._editing_tile: int | None = None

        self.central_tabs = QTabWidget(self)
        self.central_tabs.addTab(central, "Chart")
        self.central_tabs.addTab(self.dashboard_view, "Dashboard")
        self.central_tabs.currentChanged.connect(lambda _: self._refresh_dashboard())
        self.setCentralWidget(self.central_tabs)

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
        # Header clicks sort the whole frame (DataFrameModel.sort), not just the visible rows,
        # and show that column's statistics underneath.
        self.preview.setSortingEnabled(True)
        self.preview.horizontalHeader().setSortIndicatorShown(True)
        self.preview.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        self.preview.horizontalHeader().sectionClicked.connect(self._show_column_stats)
        self.preview_model.view_changed.connect(
            lambda: self.preview_status.setText(self.preview_model.status_text())
        )
        self.preview_runner = TaskRunner(self)
        self.preview_runner.idle.connect(self._preview_if_requested)
        self._preview_next: tuple[Any, Any] | None = None
        self.preview_model.scheduler = self._schedule_preview

        self.preview_search = QLineEdit(left)
        self.preview_search.setPlaceholderText("Search all rows…")
        self.preview_search.setClearButtonEnabled(True)
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(300)
        self._search_timer.timeout.connect(
            lambda: self.preview_model.set_search(self.preview_search.text())
        )
        self.preview_search.textChanged.connect(lambda _: self._search_timer.start())

        self.column_stats = QTextBrowser(left)
        self.column_stats.setMaximumHeight(170)
        self.column_stats.setPlaceholderText(
            "Click a column header to sort and see its statistics."
        )

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
        preview_layout.addWidget(self.preview_search)
        preview_layout.addWidget(self.preview, stretch=1)
        preview_layout.addWidget(self.preview_status)
        preview_layout.addWidget(self.column_stats)
        data_tabs.addTab(preview_page, "Preview")
        data_tabs.addTab(self.cleaning_report, "Cleaning report")

        left_layout.addWidget(data_tabs)

        self.data_dock = QDockWidget("Data", self)
        self.data_dock.setWidget(left)
        self.data_dock.setObjectName("data_dock")
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.data_dock)

        # --- right: recommendations, types, filters
        self.recommendations = QListWidget(self)
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

        self.customize_panel = CustomizePanel(recommend_page)
        self.customize_panel.customized.connect(self._on_customized)
        recommend_layout.addWidget(self.customize_panel)

        self.refine_button = QPushButton("Ask the model for a second opinion", recommend_page)
        self.refine_button.clicked.connect(self._on_refine)
        recommend_layout.addWidget(self.refine_button)

        pin_row = QHBoxLayout()
        self.pin_button = QPushButton("Pin to dashboard", recommend_page)
        self.pin_button.setToolTip("Keep this chart on the Dashboard tab (Ctrl+D)")
        self.pin_button.clicked.connect(self.pin_current)
        self.update_tile_button = QPushButton("Update tile", recommend_page)
        self.update_tile_button.clicked.connect(self._update_tile)
        self.update_tile_button.hide()
        pin_row.addWidget(self.pin_button)
        pin_row.addWidget(self.update_tile_button)
        recommend_layout.addLayout(pin_row)

        self.type_panel = TypeOverridePanel(self)
        self.type_panel.types_changed.connect(self._on_types_changed)

        self.filter_panel = FilterPanel(self)
        self.filter_panel.filters_changed.connect(self._on_filters_changed)

        self.analysis_tabs = QTabWidget(self)
        self.analysis_tabs.addTab(recommend_page, "Charts")
        self.analysis_tabs.addTab(self.type_panel, "Types")
        self.analysis_tabs.addTab(self.filter_panel, "Filters")

        self.cleaning_panel = CleaningPanel(self)
        self.cleaning_panel.cleaning_changed.connect(self._on_cleaning_changed)
        self.cleaning_panel.remove_requested.connect(self._on_remove_step)
        # Appended last so the tab indices stored in existing sessions keep their meaning.
        self.analysis_tabs.addTab(self.cleaning_panel, "Cleaning")

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

        open_options_action = QAction("Open with &options…", self)
        open_options_action.setShortcut("Ctrl+Shift+O")
        open_options_action.triggered.connect(self.choose_file_with_options)
        file_menu.addAction(open_options_action)

        self.recent_menu = file_menu.addMenu("Open &recent")
        self._rebuild_recent_menu()

        self.join_action = QAction("&Join another dataset…", self)
        self.join_action.triggered.connect(self.join_dataset)
        self.join_action.setEnabled(False)
        file_menu.addAction(self.join_action)

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

        self.report_action = QAction("Export &report…", self)
        self.report_action.triggered.connect(self.export_report)
        self.report_action.setEnabled(False)
        file_menu.addAction(self.report_action)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        edit_menu = self.menuBar().addMenu("&Edit")
        self.undo_action = QAction("&Undo", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.triggered.connect(self.undo)
        edit_menu.addAction(self.undo_action)
        self.redo_action = QAction("&Redo", self)
        self.redo_action.setShortcuts([QKeySequence.StandardKey.Redo, QKeySequence("Ctrl+Shift+Z")])
        self.redo_action.triggered.connect(self.redo)
        edit_menu.addAction(self.redo_action)
        self._sync_history_actions()

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.data_dock.toggleViewAction())
        view_menu.addAction(self.analysis_dock.toggleViewAction())
        view_menu.addSeparator()
        self.pin_action = QAction("&Pin chart to dashboard", self)
        self.pin_action.setShortcut("Ctrl+D")
        self.pin_action.triggered.connect(self.pin_current)
        view_menu.addAction(self.pin_action)
        view_menu.addSeparator()
        theme_menu = view_menu.addMenu("&Theme")
        self.theme_group = QActionGroup(self)
        self.theme_group.setExclusive(True)
        current = str(self.settings.value("appearance/theme", "system"))
        for value, label in (("system", "Follow the system"), ("light", "Light"), ("dark", "Dark")):
            action = QAction(label, self, checkable=True)
            action.setData(value)
            action.setChecked(value == current)
            action.triggered.connect(lambda _=False, v=value: self.set_theme(v))
            self.theme_group.addAction(action)
            theme_menu.addAction(action)
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

    def choose_file_with_options(self) -> None:
        """Pick a file, then set delimiter, encoding, number format, or sheet before opening."""
        start = str(self.settings.value("paths/last_dir", str(Path.home())))
        path, _ = QFileDialog.getOpenFileName(self, "Open dataset", start, _OPEN_FILTER)
        if not path:
            return
        dialog = ImportDialog(path, self)
        if dialog.exec() == ImportDialog.DialogCode.Accepted:
            self.open_path(path, options=dialog.options())

    def open_path(self, path: str | Path, *, options: dict[str, Any] | None = None) -> None:
        """Open a dataset or a session file, on a worker thread."""
        path = Path(path).expanduser()
        if not path.exists():
            self._error("File not found", f"There is no file at {path}.")
            return
        if not self._maybe_save_changes():
            return

        self.settings.setValue("paths/last_dir", str(path.parent))
        if path.suffix.lower() == SESSION_SUFFIX:
            try:
                session = Session.load(path)
            except PlotaVizError as exc:
                self._error("Could not open the session", str(exc))
                return
            if not Path(session.source_path).exists() and not self._locate_source(session):
                return
            if not self._confirm_session_queries(session):
                return
            self._session_view = dict(session.view)
            self._run(self._load_session_task, session, path, label=f"Restoring {path.name}…")
        else:
            self._run(self._load_dataset_task, path, options or {}, label=f"Loading {path.name}…")

    def _locate_source(self, session: Session) -> bool:
        """Ask where a session's data file went; True if the user pointed at one."""
        missing = Path(session.source_path or "?")
        answer = QMessageBox.question(
            self,
            "Data file not found",
            f"This session's data file is not at:\n{missing}\n\nLocate it? A file with "
            "different contents will still open, with a warning.",
            QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Open:
            return False
        chosen, _ = QFileDialog.getOpenFileName(
            self, f"Locate {missing.name}", str(Path.home()), _OPEN_FILTER
        )
        if not chosen:
            return False
        session.relocate(chosen)
        return True

    def _confirm_session_queries(self, session: Session) -> bool:
        """Ask before replaying query filters from a session file.

        A query filter is a pandas expression, and pandas' own documentation warns that
        ``query``/``eval`` can run arbitrary code. A ``.pviz`` file from someone else is untrusted
        input, so its expressions are shown and confirmed before anything evaluates them.

        Returns:
            False if the user cancelled. Choosing "Open without them" strips the query filters
            from ``session`` and returns True.
        """
        queries = [
            step.expression.strip()
            for step in session.pipeline
            if isinstance(step, QueryFilter) and step.expression.strip()
        ]
        if not queries:
            return True

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Session contains query filters")
        box.setText("This session applies the following query filters when it opens:")
        box.setInformativeText(
            "\n".join(f"• {q}" for q in queries)
            + "\n\nQuery filters are evaluated by pandas, which can run code. Only apply them "
            "if you trust where this file came from."
        )
        apply = box.addButton("Apply filters", QMessageBox.ButtonRole.AcceptRole)
        strip = box.addButton("Open without them", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(strip)
        box.exec()

        clicked = box.clickedButton()
        if clicked is apply:
            return True
        if clicked is strip:
            session.pipeline.steps = [
                s for s in session.pipeline.steps if not isinstance(s, QueryFilter)
            ]
            return True
        return False

    def _load_dataset_task(
        self, path: Path, options: dict[str, Any] | None = None, progress: Any = None
    ) -> tuple[Analysis, str | None, Path | None]:
        """Worker body: load and clean a dataset."""
        if progress:
            progress(15, "Reading the file…")
        analysis = Analysis.from_file(
            path,
            large_file_mb=float(self.settings.value("performance/large_file_mb", 500)),
            options=options or None,
        )
        if progress:
            progress(90, "Choosing a chart…")
        return analysis, None, None

    def _load_session_task(
        self, session: Session, path: Path, progress: Any = None
    ) -> tuple[Analysis, str | None, Path | None]:
        """Worker body: restore a session.

        The session path is handed back in the result rather than set here: the worker must not
        touch window state, and a cancelled restore must not leave Ctrl+S pointing at a file
        whose contents are not on screen.
        """
        if progress:
            progress(40, "Replaying the cleaning steps…")
        analysis, warning = Analysis.from_session(session)
        return analysis, warning, path

    def _on_loaded(self, result: tuple[Analysis, str | None, Path | None]) -> None:
        """Populate every panel once loading finishes."""
        analysis, warning, session_path = result
        self.analysis = analysis
        # Opening a plain dataset clears the session target; otherwise Ctrl+S would silently
        # overwrite the previously opened .pviz with an unrelated dataset.
        self.session_path = session_path

        self.preview_model.set_frame(analysis.df, analysis.profile)
        self.preview_status.setText(self.preview_model.status_text())
        self.cleaning_report.setPlainText(
            (analysis.result.cleaning_report() if analysis.result else "")
            + ("\n\n" + "\n".join(analysis.load_result.notes) if analysis.load_result.notes else "")
        )

        if analysis.profile:
            self.type_panel.set_profile(
                analysis.profile, analysis.type_overrides, inferred=analysis.inferred_types
            )
            self.filter_panel.set_profile(analysis.profile)
            self.customize_panel.set_facet_columns(
                [*analysis.profile.categorical, *analysis.profile.boolean]
            )
        self.nl_bar.set_columns(list(analysis.df.columns))

        self._populate_recommendations()
        self._render_current()

        self.filter_panel.set_filters(analysis.pipeline.filters)
        self._set_editing_tile(None)
        self._refresh_dashboard(force=True)
        self._sync_cleaning_panel()
        self._sync_history_actions()
        view, self._session_view = self._session_view, {}
        if session_path is not None:
            self._restore_view(view)
        self.statusBar().showMessage(analysis.load_result.summary())
        self._remember_recent(analysis.load_result.path)

        for action in (
            self.export_action,
            self.save_session_action,
            self.save_session_as_action,
            self.report_action,
            self.join_action,
        ):
            action.setEnabled(True)

        self._set_dirty(False)

        if warning:
            self._warn("Session restored with differences", warning)

    # ------------------------------------------------------------------ theme

    def set_theme(self, theme: str) -> None:
        """Switch between system, light, and dark, and redraw what is on screen."""
        self.settings.setValue("appearance/theme", theme)
        for action in self.theme_group.actions():
            action.setChecked(action.data() == theme)
        self._apply_theme()

    def _apply_theme(self) -> None:
        theme = str(self.settings.value("appearance/theme", "system"))
        self._dark = apply_theme(theme)
        self.preview_model.dark = self._dark
        self.preview_model.headerDataChanged.emit(
            Qt.Orientation.Horizontal, 0, max(0, self.preview_model.columnCount() - 1)
        )
        if self.analysis is not None:
            self._render_current(changed=False)
            self._refresh_dashboard(force=True, mark_dirty=False)

    def _on_system_scheme_changed(self, *_: object) -> None:
        if str(self.settings.value("appearance/theme", "system")) == "system":
            self._apply_theme()

    # ------------------------------------------------------------------ data preview

    def _schedule_preview(self, work: Any, done: Any) -> None:
        """Run a preview sort/search on its worker; only the newest request is applied."""
        if self.preview_runner.busy:
            self._preview_next = (work, done)
            return
        self.preview_runner.start(
            work,
            on_finished=lambda view: None if self._preview_next else done(view),
            on_failed=lambda message, _trace: self.preview_status.setText(message),
        )

    def _preview_if_requested(self) -> None:
        if self._preview_next is not None:
            work, done = self._preview_next
            self._preview_next = None
            self._schedule_preview(work, done)

    def _show_column_stats(self, section: int) -> None:
        """Show the clicked column's profile under the preview."""
        if self.analysis is None or self.analysis.profile is None:
            return
        columns = list(self.analysis.df.columns)
        if not 0 <= section < len(columns):
            return
        prof = self.analysis.profile.columns.get(str(columns[section]))
        if prof is not None:
            self.column_stats.setHtml(column_summary_html(prof))

    # ------------------------------------------------------------------ unsaved changes

    def _set_dirty(self, dirty: bool = True) -> None:
        """Record whether there are unsaved changes, and show it in the title (``*``)."""
        self._dirty = dirty and self.analysis is not None
        if self.analysis is None:
            self.setWindowTitle(__app_name__)
            return
        name = (self.session_path or self.analysis.load_result.path).name
        self.setWindowTitle(f"{name}{' *' if self._dirty else ''} — {__app_name__}")

    def _maybe_save_changes(self) -> bool:
        """Offer to save before the current work is replaced or the window closes.

        Returns:
            True to go ahead (saved, or the user chose to discard), False to stay.
        """
        if not self._dirty or self.analysis is None or not self.ask_before_discarding:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "This analysis has changes that are not saved in a session file.\n\n"
            "Save them before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self.save_session()
        return answer == QMessageBox.StandardButton.Discard

    # ------------------------------------------------------------------ charting

    def _populate_recommendations(self) -> None:
        """Fill the ranked list and highlight the chart that is actually active.

        Selection happens with signals blocked. Selecting row 0 with signals live used to fire
        ``currentItemChanged``, which re-chose the top recommendation — so every filter or type
        change replaced the user's chart, and every load rendered twice.
        """
        self.recommendations.blockSignals(True)
        self.recommendations.clear()

        active_row = -1
        if self.analysis:
            active = self.analysis.spec
            for row, spec in enumerate(self.analysis.recommendations):
                if active is not None and _same_chart(spec, active):
                    active_row = row
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
            if active_row >= 0:
                self.recommendations.setCurrentRow(active_row)

        self.recommendations.blockSignals(False)
        self._show_why(self.analysis.spec if self.analysis else None)

    def _on_recommendation_chosen(self, item: QListWidgetItem | None, *_: object) -> None:
        """Switch to a chart the user picked from the ranked list."""
        if item is None or self.analysis is None:
            return
        spec = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(spec, ChartSpec):
            return
        if self._defer_while_busy("choose", lambda: self._choose_and_render(spec)):
            return
        self._choose_and_render(spec)

    def _choose_and_render(self, spec: ChartSpec) -> None:
        """Make ``spec`` the active chart and draw it."""
        if self.analysis is None:
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
        if self._defer_while_busy("choose", lambda: self._on_spec_edited(spec)):
            return
        try:
            self.analysis.choose(spec)
        except PlotaVizError as exc:
            self.nl_bar.show_status(str(exc), error=True)
            return
        self._show_why(spec)
        self._render_current()

    def _on_customized(self, values: dict[str, Any]) -> None:
        """Apply title, labels, scales, and palette from the Customize panel."""
        if self.analysis is None or self.analysis.spec is None:
            return
        if self._defer_while_busy("customize", lambda: self._on_customized(values)):
            return
        try:
            self.analysis.choose(apply_customization(self.analysis.spec, values))
        except PlotaVizError as exc:
            self._error("That customization does not fit this chart", str(exc))
            return
        self._render_current()

    def _show_why(self, spec: ChartSpec | None) -> None:
        """Display the justification for the current chart."""
        self.why_label.setPlainText(spec.why if spec and spec.why else "")

    def _render_current(self, *, changed: bool = True) -> None:
        """Draw the active chart, surfacing any preparation notes.

        Args:
            changed: Whether this redraw follows a change worth saving. A redraw for a renderer
                switch is not one.
        """
        if changed:
            self._set_dirty()
        self._render_generation += 1
        if self.analysis is None or self.analysis.spec is None:
            self.chart_view.show_message("No chart is selected yet.")
            return
        # The builder row and the customize panel always mirror the chart on screen.
        self.nl_bar.show_spec(self.analysis.spec, announce=False)
        self.customize_panel.set_spec(self.analysis.spec)
        if self.render_runner.busy:
            self._render_again = True
            return

        # Snapshot what to draw: frames are never mutated in place, and the spec is copied, so
        # the worker cannot see a half-applied change.
        df, spec = self.analysis.df, self.analysis.spec.copy()
        interactive = self.chart_view.interactive
        max_points = int(self.settings.value("performance/max_points", MAX_POINTS))
        generation = self._render_generation
        dark = self._dark
        self.render_runner.start(
            lambda: build_render(
                df, spec, interactive=interactive, max_points=max_points, dark=dark
            ),
            on_finished=lambda result: self._install_render(result, generation),
            on_failed=lambda message, trace: self._render_failed(message, trace, generation),
        )

    def _render_if_requested(self) -> None:
        if self._render_again:
            self._render_again = False
            self._render_current(changed=False)

    def _install_render(self, result: RenderResult, generation: int) -> None:
        """Show a chart built on the render worker — unless it is already out of date."""
        if self._render_again or generation != self._render_generation:
            return
        notes = self.chart_view.install(result)
        if notes:
            self.chart_note.setText("  ".join(notes))
            self.chart_note.show()
        else:
            self.chart_note.hide()
        self.statusBar().showMessage(self.chart_view.renderer_note())

    def _render_failed(self, message: str, _trace: str, generation: int) -> None:
        if self._render_again or generation != self._render_generation:
            return
        self.chart_view.show_message(message)
        self.chart_note.hide()
        self.statusBar().showMessage("The chart could not be drawn.")

    # ------------------------------------------------------------------ panel callbacks

    def _on_types_changed(self, overrides: dict[str, str]) -> None:
        """Re-run the pipeline with corrected column types."""
        if self.analysis is None:
            return
        analysis, chosen = self.analysis, dict(overrides)
        self._replay("Applying types…", lambda: analysis.set_type_overrides(chosen))

    def _on_filters_changed(self, filters: list[Any]) -> None:
        """Re-run the pipeline with new filters, off the UI thread."""
        if self.analysis is None:
            return
        analysis, chosen = self.analysis, list(filters)
        self._replay(
            "Filtering…",
            lambda: analysis.set_filters(chosen),
            on_failed=lambda message, _trace: self.filter_panel.show_query_error(message),
        )

    def _on_cleaning_changed(self, settings: dict[str, Any]) -> None:
        """Swap the cleaning steps for the ones chosen in the Cleaning tab."""
        if self.analysis is None:
            return
        analysis, chosen = self.analysis, dict(settings)
        self._replay("Cleaning…", lambda: analysis.set_cleaning(**chosen))

    def _on_remove_step(self, index: int) -> None:
        """Delete one pipeline step."""
        if self.analysis is None:
            return
        analysis = self.analysis
        self._replay("Removing step…", lambda: analysis.remove_step(index), sync_panels=True)

    def undo(self) -> None:
        """Revert the last change to filters, types, or cleaning."""
        if self.analysis is not None and self.analysis.can_undo:
            analysis = self.analysis
            self._replay("Undoing…", analysis.undo, sync_panels=True)

    def redo(self) -> None:
        """Re-apply the last undone change."""
        if self.analysis is not None and self.analysis.can_redo:
            analysis = self.analysis
            self._replay("Redoing…", analysis.redo, sync_panels=True)

    def _sync_history_actions(self) -> None:
        """Enable Undo/Redo only when there is something to undo or redo."""
        self.undo_action.setEnabled(bool(self.analysis and self.analysis.can_undo))
        self.redo_action.setEnabled(bool(self.analysis and self.analysis.can_redo))

    def _sync_cleaning_panel(self) -> None:
        """Show the analysis's current cleaning settings and step list."""
        if self.analysis is None:
            return
        reports = self.analysis.result.reports if self.analysis.result else []
        self.cleaning_panel.set_state(
            self.analysis.cleaning(), list(self.analysis.pipeline), reports
        )

    def _replay(
        self,
        label: str,
        change: Callable[[], Any],
        *,
        on_failed: Callable[[str, str], None] | None = None,
        sync_panels: bool = False,
    ) -> None:
        """Apply a state change that replays the pipeline, on the worker thread.

        If another task is running the replay is queued, not dropped: dropping it used to leave
        the panels showing a type or filter the data did not reflect.

        Args:
            sync_panels: Rewrite the filter and type panels from the analysis afterwards. Needed
                when the change did not come from those panels (undo, redo, removing a step);
                skipped otherwise so edits made while the replay ran are not overwritten.
        """

        def refresh(*_: object) -> None:
            self._after_rerun(sync_panels=sync_panels)

        def task(progress: Any = None) -> None:
            if progress:
                progress(30, "Replaying the cleaning steps…")
            change()

        self._run(
            task,
            label=label,
            on_finished=refresh,
            on_failed=on_failed
            or (lambda message, trace: self._error("That change failed", message, detail=trace)),
            # The replay itself cannot be interrupted, and it has already changed the analysis by
            # the time it returns — so a cancelled replay still refreshes the panels.
            on_cancelled=refresh,
            queue=True,
            # Undo and redo stack: two presses mean two steps, so they are never merged.
            queue_key=None if label in {"Undoing…", "Redoing…"} else label,
        )

    # ------------------------------------------------------------------ dashboard

    def pin_current(self) -> None:
        """Add the chart on screen to the dashboard."""
        if self.analysis is None or self.analysis.spec is None:
            return
        try:
            index = self.analysis.pin()
        except PlotaVizError as exc:
            self._error("Could not pin this chart", str(exc))
            return
        self.statusBar().showMessage(f"Pinned as dashboard tile {index + 1}.")
        self._refresh_dashboard(force=True)

    def _open_tile(self, index: int) -> None:
        """Load a tile into the Chart tab so it can be edited, then offer "Update tile"."""
        if self.analysis is None or not 0 <= index < len(self.analysis.dashboard):
            return
        spec = self.analysis.dashboard[index].copy()
        if self._defer_while_busy("choose", lambda: self._open_tile(index)):
            return
        try:
            self.analysis.choose(spec)
        except PlotaVizError as exc:
            self._error("This tile does not fit the current data", str(exc))
            return
        self._set_editing_tile(index)
        self.central_tabs.setCurrentIndex(0)
        self.analysis_tabs.setCurrentIndex(0)
        self._populate_recommendations()
        self._render_current()

    def _update_tile(self) -> None:
        """Replace the tile being edited with the chart on screen."""
        if self.analysis is None or self._editing_tile is None:
            return
        try:
            self.analysis.replace_tile(self._editing_tile)
        except PlotaVizError as exc:
            self._error("Could not update the tile", str(exc))
            return
        self.statusBar().showMessage(f"Updated dashboard tile {self._editing_tile + 1}.")
        self._refresh_dashboard(force=True)

    def _remove_tile(self, index: int) -> None:
        if self.analysis is None:
            return
        self.analysis.unpin(index)
        if self._editing_tile is not None:
            if self._editing_tile == index:
                self._set_editing_tile(None)
            elif self._editing_tile > index:
                self._set_editing_tile(self._editing_tile - 1)
        self._refresh_dashboard(force=True)

    def _move_tile(self, index: int, to: int) -> None:
        if self.analysis is None:
            return
        self.analysis.move_tile(index, to)
        if self._editing_tile == index:
            self._set_editing_tile(max(0, min(to, len(self.analysis.dashboard) - 1)))
        elif self._editing_tile == to:
            self._set_editing_tile(index)
        self._refresh_dashboard(force=True)

    def _set_editing_tile(self, index: int | None) -> None:
        self._editing_tile = index
        self.update_tile_button.setVisible(index is not None)
        if index is not None:
            self.update_tile_button.setText(f"Update tile {index + 1}")

    def _refresh_dashboard(self, *, force: bool = False, mark_dirty: bool = True) -> None:
        """Redraw the tiles if the Dashboard tab is showing; otherwise mark them stale.

        ``force`` marks the dashboard stale first — used after pinning, removing, reordering,
        or any data change.
        """
        count = len(self.analysis.dashboard) if self.analysis else 0
        self.central_tabs.setTabText(1, f"Dashboard ({count})" if count else "Dashboard")
        if force:
            self._dashboard_stale = True
            self._tiles_generation += 1  # anything in flight now describes old tiles
            if mark_dirty:
                self._set_dirty()
        if not self._dashboard_stale or self.central_tabs.currentIndex() != 1:
            return
        self._dashboard_stale = False
        if self.analysis is None:
            self.dashboard_view.set_tiles(None, [], [])
            return
        if self.tile_runner.busy:
            self._tiles_again = True
            return
        df, specs = self.analysis.df, [tile.copy() for tile in self.analysis.dashboard]
        statuses = self.analysis.tile_status()
        self._tiles_generation += 1
        generation = self._tiles_generation

        def installed(built: list[Any]) -> None:
            if not self._tiles_again and generation == self._tiles_generation:
                self.dashboard_view.set_tiles(df, specs, statuses, built)

        dark = self._dark
        self.tile_runner.start(
            lambda: build_tiles(df, specs, statuses, dark=dark),
            on_finished=installed,
            on_failed=lambda message, _trace: self.statusBar().showMessage(message),
        )

    def _tiles_if_requested(self) -> None:
        if self._tiles_again:
            self._tiles_again = False
            self._dashboard_stale = True
            self._refresh_dashboard()

    def _after_rerun(self, *, sync_panels: bool = False) -> None:
        """Refresh every panel after a pipeline replay."""
        if self.analysis is None:
            return
        self._refresh_dashboard(force=True)
        self._sync_cleaning_panel()
        self._sync_history_actions()
        if self.analysis.profile is not None:
            # Types and cleaning change which columns are categories.
            self.customize_panel.set_facet_columns(
                [*self.analysis.profile.categorical, *self.analysis.profile.boolean]
            )
        if sync_panels:
            current = self.filter_panel._profile
            columns = set(self.analysis.df.columns)
            if current is None or set(current.columns) != columns:
                # Columns came or went (a join, an undo): rebuild the widgets, bounded by the
                # unfiltered data so a filter can always be widened again.
                self.filter_panel.set_profile(self.analysis.unfiltered_profile())
            self.filter_panel.set_filters(self.analysis.pipeline.filters)
            if self.analysis.profile:
                self.type_panel.set_profile(
                    self.analysis.profile,
                    self.analysis.type_overrides,
                    inferred=self.analysis.inferred_types,
                )
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
        # Consent is per provider: agreeing to send statistics to one service is not agreement
        # to send them to another.
        consent = self.settings.value(f"llm/consent/{name}", False, type=bool)
        return LLMAssistant(provider, consent=consent)

    def _refresh_llm_state(self) -> None:
        """Enable or disable the query bar to match the configured provider.

        Runs on the UI thread, so it does not probe the Ollama server — that network round trip
        froze startup for up to two seconds when Ollama was configured but not running. A dead
        server is reported clearly by the first request instead.
        """
        assistant = self._assistant()
        provider = assistant.provider
        usable = (
            provider is not None
            and (assistant.consent or not provider.sends_data_off_machine)
            and (isinstance(provider, OllamaProvider) or provider.is_configured())
        )
        self.nl_bar.set_enabled_for_provider(usable, provider.name if provider else "")
        self.refine_button.setEnabled(usable)

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

        def cancelled() -> None:
            self.nl_bar.set_busy(False)
            self.nl_bar.show_status("Cancelled.")

        if not self._run(
            task,
            label="Asking the model…",
            on_finished=done,
            on_failed=failed,
            on_cancelled=cancelled,
        ):
            self.nl_bar.set_busy(False)

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

    def export(self, *, scope: str = "chart") -> None:
        """Export the chart, or the whole dashboard, as an image, HTML page, or Python script."""
        if self.analysis is None or self.analysis.spec is None:
            return

        source = self.analysis.load_result.path
        tiles = len(self.analysis.dashboard)
        dialog = ExportDialog(
            self,
            default_name=(source.stem or "chart") + ("_dashboard" if scope == "dashboard" else ""),
            start_dir=source.parent,
            dashboard_tiles=tiles,
            dashboard_scope=scope == "dashboard",
        )
        if dialog.exec() != ExportDialog.DialogCode.Accepted:
            return

        target = dialog.path()
        mode = dialog.mode
        analysis = self.analysis
        specs = list(analysis.dashboard)
        columns = dialog.columns_spin.value()

        def task(progress: Any = None) -> Path:
            if progress:
                progress(40, f"Writing {target.name}…")
            if dialog.scope == "dashboard":
                from ..core import codegen, exporter

                if mode == "code":
                    return codegen.write(
                        target, analysis.generate_dashboard_code(flavour=dialog.flavour())
                    )
                if mode == "html":
                    return exporter.export_dashboard_html(
                        analysis.df, specs, target, columns=columns
                    )
                return exporter.export_dashboard_image(
                    analysis.df, specs, target, options=dialog.options(), columns=columns
                )
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

    def export_report(self) -> None:
        """Write the analysis report (HTML or PDF) for the current state."""
        if self.analysis is None:
            return
        source = self.analysis.load_result.path
        suggested = source.with_name(f"{source.stem}_report.html")
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Export report", str(suggested), "HTML report (*.html);;PDF report (*.pdf)"
        )
        if not chosen:
            return
        analysis, target = self.analysis, Path(chosen)

        def task(progress: Any = None) -> Path:
            if progress:
                progress(40, f"Writing {target.name}…")
            return export_report(analysis, target)

        self._run(
            task,
            label=f"Exporting {target.name}…",
            on_finished=lambda written: self.statusBar().showMessage(f"Saved {written}"),
        )

    def join_dataset(self) -> None:
        """Add columns from a second file, matched on key columns."""
        if self.analysis is None:
            return
        dialog = JoinDialog(
            self.analysis.join_base(),
            start_dir=self.analysis.load_result.path.parent,
            parent=self,
            normalize=self.analysis._renames_columns(),
        )
        if dialog.exec() != JoinDialog.DialogCode.Accepted:
            return
        values, analysis = dialog.result_values(), self.analysis
        self._replay(
            "Joining…",
            lambda: analysis.join_dataset(values["path"], values["on"], how=values["how"]),
            sync_panels=True,
        )

    def save_session(self, *, ask: bool = False) -> bool:
        """Save the current state to a ``.pviz`` file. Returns whether it was saved."""
        if self.analysis is None:
            return False

        target = self.session_path
        if target is None or ask:
            suggested = self.analysis.load_result.path.with_suffix(SESSION_SUFFIX)
            chosen, _ = QFileDialog.getSaveFileName(
                self, "Save session", str(suggested), f"PlotaViz sessions (*{SESSION_SUFFIX})"
            )
            if not chosen:
                return False
            target = Path(chosen)

        view = {
            "renderer": self.chart_view.renderer,
            "active_tab": self.analysis_tabs.currentIndex(),
            "central_tab": self.central_tabs.currentIndex(),
            "dashboard_columns": self.dashboard_view.columns,
            "geometry": bytes(self.saveGeometry().toBase64().data()).decode(),
        }
        try:
            written = self.analysis.to_session(view).save(target)
        except PlotaVizError as exc:
            self._error("Could not save the session", str(exc))
            return False

        self.session_path = written
        self._set_dirty(False)
        self.statusBar().showMessage(f"Session saved to {written}")
        return True

    # ------------------------------------------------------------------ settings / about

    def open_settings(self) -> None:
        """Show the settings dialog and apply anything that changed."""
        dialog = SettingsDialog(
            self,
            settings=self.settings,
            profile=self.analysis.profile if self.analysis else None,
            candidates=list(self.analysis.recommendations) if self.analysis else None,
        )
        if dialog.exec() == SettingsDialog.DialogCode.Accepted:
            self._refresh_llm_state()
            self.chart_view.set_prefer_static(
                self.settings.value("performance/prefer_static", False, type=bool)
            )
            self._render_current(changed=False)

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
        on_cancelled: Any = None,
        queue: bool = False,
        queue_key: str | None = None,
    ) -> bool:
        """Run a callable on the worker thread with progress and cancellation.

        Args:
            queue: When a task is already running, run this one after it instead of refusing.
            queue_key: Queued calls with the same key replace each other; ``None`` never does.

        Returns:
            Whether the task was started or queued. False means it was refused.
        """
        if self.runner.busy:
            if queue:
                self._enqueue(
                    queue_key,
                    lambda: self._run(
                        fn,
                        *args,
                        label=label,
                        on_finished=on_finished,
                        on_failed=on_failed,
                        on_cancelled=on_cancelled,
                        queue=True,
                        queue_key=queue_key,
                    ),
                )
                self.statusBar().showMessage(f"{label} (queued)")
                return True
            self.statusBar().showMessage("Still finishing the previous task…")
            return False

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

        def cancelled() -> None:
            self._end_task()
            self.statusBar().showMessage("Cancelled.")
            if on_cancelled:
                on_cancelled()

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
            on_cancelled=cancelled,
        )
        return True

    def _enqueue(self, key: str | None, action: Callable[[], None]) -> None:
        """Queue work for when the worker is free, replacing a queued entry with the same key."""
        if key is not None:
            for index, (existing, _) in enumerate(self._pending):
                if existing == key:
                    self._pending[index] = (key, action)
                    return
        self._pending.append((key, action))

    def _defer_while_busy(self, key: str, action: Callable[[], None]) -> bool:
        """Queue a UI-thread change to the analysis if a replay is running on the worker.

        The worker replaces the pipeline, result, and active chart while it replays. Choosing a
        chart or applying a customization at the same moment raced it — the choice could be
        rolled back or read half-updated state — so such changes wait their turn instead.

        Returns:
            True if ``action`` was queued; the caller should stop. False means run it now.
        """
        if not self.runner.busy:
            return False
        self._enqueue(key, action)
        self.statusBar().showMessage("Will apply when the current update finishes…")
        return True

    def _drain_pending(self) -> None:
        """Run queued work in order until something occupies the worker again."""
        while self._pending and not self.runner.busy:
            _, action = self._pending.pop(0)
            action()

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

    def _restore_view(self, view: dict[str, Any]) -> None:
        """Apply the window state a session saved: the active tab and the window geometry."""
        columns = view.get("dashboard_columns")
        if isinstance(columns, int):
            self.dashboard_view.set_columns(columns)
        central = view.get("central_tab")
        if isinstance(central, int) and 0 <= central < self.central_tabs.count():
            self.central_tabs.setCurrentIndex(central)
            self._refresh_dashboard(force=True)
        tab = view.get("active_tab")
        if isinstance(tab, int) and 0 <= tab < self.analysis_tabs.count():
            self.analysis_tabs.setCurrentIndex(tab)
        geometry = view.get("geometry")
        if isinstance(geometry, str) and geometry:
            self.restoreGeometry(QByteArray.fromBase64(geometry.encode()))

    def _restore_geometry(self) -> None:
        """Restore window size and dock layout from the last session."""
        geometry = self.settings.value("window/geometry")
        state = self.settings.value("window/state")
        if geometry:
            self.restoreGeometry(geometry)
        if state:
            self.restoreState(state)

    def closeEvent(self, event: Any) -> None:
        """Offer to save, then persist window state, stop background work, clean up."""
        if not self._maybe_save_changes():
            event.ignore()
            return
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState())
        self.settings.sync()
        self.runner.cancel()
        self.runner.wait()
        for worker in (self.render_runner, self.tile_runner, self.preview_runner):
            worker.cancel()
            worker.wait()
        self.chart_view.cleanup()
        super().closeEvent(event)


def _same_chart(a: ChartSpec, b: ChartSpec) -> bool:
    """Whether two specs draw the same chart, ignoring score and reasoning."""
    return (a.chart, a.x, a.y, a.color, a.agg) == (b.chart, b.x, b.y, b.color, b.agg)


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
