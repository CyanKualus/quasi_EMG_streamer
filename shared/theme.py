"""Shared appearance for the EMG application and its trial browser.

Nothing in this module decides anything an analysis reads. It sets typefaces,
colours and spacing, and the same numbers come out of the pipelines whether it
is applied or not.
"""
from __future__ import annotations

import matplotlib
from PyQt6 import QtGui, QtWidgets

# ---------------------------------------------------------------------------
# Palette.
#
# A page is the soft grey a form or a set of cards sits on; a surface is the
# white of the things sitting on it, and of every matplotlib figure, so a plot
# and the panel holding it are the same white rather than two.
# ---------------------------------------------------------------------------
PALETTE = {
    "page": "#f1f4f7",
    "surface": "#ffffff",
    "border": "#e2e7ee",
    "line": "#ccd4de",
    "text": "#1f2933",
    "muted": "#66717f",
    "faint": "#9aa5b1",
    "accent": "#2f6f8f",
    "accent_hover": "#3a87aa",
    "accent_press": "#255a75",
    "accent_soft": "#e8f1f6",
    "warn": "#a04000",
    "good": "#2f8f5b",
    "mark": "#d1495b",
}

# Windows 11, Windows 10, then whatever a non-Windows machine can offer. The
# first family actually installed wins; an absent one costs nothing.
UI_FONT_CANDIDATES = ("Segoe UI Variable Text", "Segoe UI", "Inter",
                      "Noto Sans", "DejaVu Sans")
UI_FONT_POINT_SIZE = 10.0

# The gauge and time-frequency panels carry a lot of small print in a small
# box, so their figures run a size below the widgets around them.
PLOT_FONT_SIZE = 9.0

_STYLESHEET = """
QWidget { color: %(text)s; }
QMainWindow, QDialog { background: %(page)s; }

/* ---- tabs: a rule under the current one, not a box around the page ---- */
QTabWidget::pane { background: %(page)s; border: none; top: -1px; }
QTabBar { background: transparent; }
QTabBar::tab {
    background: transparent;
    border: none;
    border-bottom: 3px solid transparent;
    padding: 9px 20px 7px 20px;
    margin-right: 2px;
    color: %(muted)s;
    font-weight: 600;
}
QTabBar::tab:hover { color: %(text)s; }
QTabBar::tab:selected { color: %(accent)s; border-bottom: 3px solid %(accent)s; }

/* A page that is mostly figures takes the figures' own white, so a panel does
   not read as a bright rectangle pasted onto a grey page. */
QWidget#plotPage { background: %(surface)s; }
QFrame#card {
    background: %(surface)s;
    border: 1px solid %(border)s;
    border-radius: 12px;
}

/* ---- type ---- */
QLabel#appTitle { font-size: 25px; font-weight: 700; color: %(text)s; }
QLabel#appSubtitle { font-size: 12px; color: %(muted)s; }
QLabel#sectionHeader { font-size: 11px; font-weight: 700; color: %(accent)s; }
QLabel#fieldLabel { color: %(muted)s; font-weight: 600; }
QLabel#hint { color: %(muted)s; }
QLabel#warning { color: %(warn)s; }
QLabel#totalScore { font-size: 21px; font-weight: 700; color: %(text)s; }
QLabel#statusLine { color: %(muted)s; }

/* ---- fields ---- */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
    background: %(surface)s;
    border: 1px solid %(line)s;
    border-radius: 7px;
    padding: 5px 9px;
    selection-background-color: %(accent)s;
    selection-color: %(surface)s;
}
QComboBox { padding-right: 26px; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QPlainTextEdit:focus { border: 1px solid %(accent)s; }
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled,
QDoubleSpinBox:disabled {
    background: #f4f6f9; color: %(faint)s; border-color: %(border)s;
}
QComboBox::drop-down {
    subcontrol-origin: padding; subcontrol-position: center right;
    width: 22px; border: none;
}
QComboBox::down-arrow {
    image: none; width: 0; height: 0; margin-right: 8px;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid %(muted)s;
}
QComboBox::down-arrow:disabled { border-top-color: %(faint)s; }
QComboBox QAbstractItemView {
    background: %(surface)s;
    border: 1px solid %(line)s;
    border-radius: 7px;
    padding: 4px;
    outline: none;
    selection-background-color: %(accent_soft)s;
    selection-color: %(text)s;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border; subcontrol-position: top right;
    width: 18px; border: none;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border; subcontrol-position: bottom right;
    width: 18px; border: none;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: none; width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-bottom: 5px solid %(muted)s;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: none; width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid %(muted)s;
}
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {
    border-bottom-color: %(faint)s;
}
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {
    border-top-color: %(faint)s;
}

/* ---- buttons ---- */
QPushButton {
    background: %(surface)s;
    border: 1px solid %(line)s;
    border-radius: 7px;
    padding: 6px 16px;
    font-weight: 600;
    color: %(text)s;
}
QPushButton:hover { border-color: %(accent)s; color: %(accent_press)s; }
QPushButton:pressed { background: %(accent_soft)s; }
QPushButton:disabled {
    background: #f4f6f9; color: %(faint)s; border-color: %(border)s;
}
QPushButton#primaryButton {
    background: %(accent)s; color: %(surface)s; border: 1px solid %(accent)s;
}
QPushButton#primaryButton:hover {
    background: %(accent_hover)s; border-color: %(accent_hover)s;
    color: %(surface)s;
}
QPushButton#primaryButton:pressed {
    background: %(accent_press)s; border-color: %(accent_press)s;
}
QPushButton#primaryButton:disabled {
    background: #dfe5ec; color: %(faint)s; border-color: #dfe5ec;
}
QPushButton#startButton {
    background: %(accent)s; color: %(surface)s; border: none;
    border-radius: 10px; padding: 11px 32px;
    font-size: 16px; font-weight: 700;
}
QPushButton#startButton:hover { background: %(accent_hover)s; }
QPushButton#startButton:pressed { background: %(accent_press)s; }
QPushButton#startButton:disabled { background: #c8d2dc; color: #f4f6f9; }

/* ---- the EMG tab's one card per recording ----
   The title names the recording the card is about, so it belongs inside the
   card: positioned from the padding box rather than the margin box, which is
   what put it on the page above the border. The top padding is the room it
   needs, and is what keeps the first figure clear of it. */
QGroupBox {
    background: %(surface)s;
    border: 1px solid %(border)s;
    border-radius: 12px;
    margin-top: 0;
    padding: 34px 12px 12px 12px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: padding; subcontrol-position: top left;
    left: 6px; top: 10px; padding: 0;
    color: %(accent)s;
}

/* ---- the component selector ---- */
QSlider::groove:vertical {
    background: %(border)s; width: 6px; border-radius: 3px;
}
QSlider::handle:vertical {
    background: %(accent)s; border: none;
    width: 16px; height: 16px; margin: 0 -5px; border-radius: 8px;
}
QSlider::handle:vertical:hover { background: %(accent_hover)s; }
QSlider::handle:vertical:disabled { background: %(faint)s; }

/* ---- chrome ---- */
QScrollArea { border: none; background: transparent; }
QWidget#scrollBody { background: transparent; }
QScrollBar:vertical { background: transparent; width: 12px; margin: 0; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 0; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #c3ccd7; border-radius: 6px;
}
QScrollBar::handle:vertical { min-height: 32px; }
QScrollBar::handle:horizontal { min-width: 32px; }
QScrollBar::handle:hover { background: #a7b3c0; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QSplitter::handle { background: %(border)s; }
QToolTip {
    background: %(text)s; color: #eef2f6; border: none;
    padding: 6px 8px; border-radius: 6px;
}
"""


def ui_font_family():
    """The first candidate typeface this machine actually has."""
    installed = set(QtGui.QFontDatabase.families())
    for family in UI_FONT_CANDIDATES:
        if family in installed:
            return family
    return QtWidgets.QApplication.font().family()


def apply_plot_style(family):
    """Draw the matplotlib panels in the window's own type and colours.

    The figures are part of the page rather than pictures pasted onto it, so
    they take the same typeface, the same greys and the same white background.
    Only presentation is set here: no colour map, and no property cycle, so a
    panel that names its own colours keeps naming them.
    """
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [family, "Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": PLOT_FONT_SIZE,
        "axes.titlesize": PLOT_FONT_SIZE + 1,
        "axes.titleweight": "semibold",
        "axes.titlecolor": PALETTE["text"],
        "axes.labelsize": PLOT_FONT_SIZE,
        "axes.labelcolor": PALETTE["muted"],
        "axes.edgecolor": PALETTE["line"],
        "axes.linewidth": 0.8,
        "axes.facecolor": PALETTE["surface"],
        "figure.facecolor": PALETTE["surface"],
        "savefig.facecolor": PALETTE["surface"],
        "text.color": PALETTE["text"],
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "xtick.labelsize": PLOT_FONT_SIZE - 1,
        "ytick.labelsize": PLOT_FONT_SIZE - 1,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "grid.color": PALETTE["border"],
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": PLOT_FONT_SIZE - 1,
    })


def apply(widget):
    """Give *widget* and everything under it the application's look.

    The font and the style sheet are set on the window rather than on the
    application object, so a window built by a test or a screenshot script
    looks like the one an operator sees; dialogs parented to it inherit both.
    The widget style is the one thing that cannot be set per window, and is
    left alone if no application object exists yet.
    """
    application = QtWidgets.QApplication.instance()
    if application is not None:
        # Fusion draws the same on every Windows build and honours the whole
        # style sheet below. The native style ignores several of its rules,
        # which would leave one machine's window a different shape from the
        # next one's.
        application.setStyle("Fusion")
    family = ui_font_family()
    font = QtGui.QFont(family)
    font.setPointSizeF(UI_FONT_POINT_SIZE)
    widget.setFont(font)
    widget.setStyleSheet(_STYLESHEET % PALETTE)
    apply_plot_style(family)


def label_font(widget, *, point_size=None, weight=None, letter_spacing=None,
               uppercase=False):
    """Adjust one label's typeface where a stylesheet cannot reach it.

    Qt style sheets have no letter-spacing and no small-caps, and the section
    headings on the start form want both.
    """
    font = widget.font()
    if point_size is not None:
        font.setPointSizeF(point_size)
    if weight is not None:
        font.setWeight(weight)
    if letter_spacing is not None:
        font.setLetterSpacing(
            QtGui.QFont.SpacingType.PercentageSpacing, letter_spacing)
    font.setCapitalization(
        QtGui.QFont.Capitalization.AllUppercase if uppercase
        else QtGui.QFont.Capitalization.MixedCase)
    widget.setFont(font)
