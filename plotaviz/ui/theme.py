"""Light, dark, or follow-the-system appearance.

The window theme is a Qt palette. Charts drawn **on screen** follow it — Plotly's dark template,
matplotlib figures recoloured for a dark background — so a dark window does not show glaring white
chart panels. **Exports stay light**: an image or script made for a report or a slide should not
depend on how the app happened to be themed when it was saved.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication

#: Values of the ``appearance/theme`` setting.
THEMES: tuple[str, ...] = ("system", "light", "dark")

#: Background used for dark on-screen charts, matching the dark palette's base.
DARK_BACKGROUND = "#1f1f22"
DARK_FOREGROUND = "#e6e6e6"

_original_style: str | None = None


def dark_palette() -> QPalette:
    """A neutral dark palette for the Fusion style."""
    palette = QPalette()
    window, base, text = QColor(43, 43, 46), QColor(31, 31, 34), QColor(230, 230, 230)
    palette.setColor(QPalette.ColorRole.Window, window)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, base)
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(38, 38, 41))
    palette.setColor(QPalette.ColorRole.ToolTipBase, window)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, window)
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 85, 85))
    palette.setColor(QPalette.ColorRole.Link, QColor(110, 170, 255))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(53, 110, 190))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Mid, QColor(140, 140, 145))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(140, 140, 145))
    for role in (
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
        QPalette.ColorRole.WindowText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(120, 120, 125))
    return palette


def system_prefers_dark() -> bool:
    """Whether the OS asks for a dark appearance (Qt 6.5+; False when unknown)."""
    hints = QGuiApplication.styleHints()
    scheme = getattr(hints, "colorScheme", None)
    return bool(scheme is not None and scheme() == Qt.ColorScheme.Dark)


def apply_theme(theme: str, app: Any = None) -> bool:
    """Apply a theme to the application.

    Args:
        theme: One of :data:`THEMES`. Unknown values mean ``"system"``.
        app: The application; defaults to the running one.

    Returns:
        Whether the result is dark — what on-screen charts should follow.
    """
    global _original_style
    app = app or QApplication.instance()
    if app is None:
        return False
    if _original_style is None:
        _original_style = app.style().name()

    dark = theme == "dark" or (theme not in ("light", "dark") and system_prefers_dark())
    if dark:
        # Fusion is the one built-in style that honours a custom palette on every platform.
        app.setStyle("Fusion")
        app.setPalette(dark_palette())
    else:
        app.setStyle(_original_style)
        app.setPalette(app.style().standardPalette())
    return dark


def darken_figure(figure: Any) -> None:
    """Recolour a matplotlib figure for a dark window (on-screen only; Qt-free)."""
    figure.patch.set_facecolor(DARK_BACKGROUND)
    for text in figure.texts:
        text.set_color(DARK_FOREGROUND)
    for axes in figure.axes:
        axes.set_facecolor(DARK_BACKGROUND)
        axes.title.set_color(DARK_FOREGROUND)
        axes.xaxis.label.set_color(DARK_FOREGROUND)
        axes.yaxis.label.set_color(DARK_FOREGROUND)
        axes.tick_params(colors=DARK_FOREGROUND, which="both")
        for spine in axes.spines.values():
            spine.set_edgecolor("#8a8a8f")
        legend = axes.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(DARK_BACKGROUND)
            legend.get_frame().set_edgecolor("#8a8a8f")
            for text in legend.get_texts():
                text.set_color(DARK_FOREGROUND)
            legend.get_title().set_color(DARK_FOREGROUND)
