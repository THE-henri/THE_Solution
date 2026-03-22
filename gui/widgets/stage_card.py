from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
)
from PyQt6.QtCore import Qt
from gui.widgets.info_button import InfoButton

# ── Status definitions ────────────────────────────────────────────────────

WAITING = "waiting"
READY   = "ready"
DONE    = "done"
STALE   = "stale"
ERROR   = "error"

_BADGE = {
    WAITING: ("● Waiting", "#555",    "#888"),
    READY:   ("● Ready",   "#1a3a6e", "#5b8dee"),
    DONE:    ("✓ Done",    "#1a4a30", "#3cb371"),
    STALE:   ("⚠ Stale",   "#4a3a10", "#e8a020"),
    ERROR:   ("✗ Error",   "#4a1a1a", "#e84d4d"),
}


class StageCard(QWidget):
    """
    Collapsible card that wraps one pipeline stage.

    Usage
    -----
    card = StageCard("Stage 1 — Load data")
    card.add_widget(my_widget)
    card.set_status(READY)
    card.set_card_enabled(True)
    """

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title     = title
        self._status    = WAITING
        self._collapsed = False
        self._build_ui()

    # ── Build ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 8)
        outer.setSpacing(0)

        self._frame = QFrame()
        self._frame.setObjectName("stage_card")
        frame_layout = QVBoxLayout(self._frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)
        outer.addWidget(self._frame)

        # ── Header ────────────────────────────────────────────────────────
        self._header = QWidget()
        self._header.setObjectName("stage_header")
        self._header.setFixedHeight(38)
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(14, 0, 14, 0)
        header_layout.setSpacing(6)

        self._arrow = QLabel("▼")
        self._arrow.setFixedWidth(14)
        self._arrow.setStyleSheet("color:#5b8dee; font-size:9pt;")
        header_layout.addWidget(self._arrow)

        self._title_lbl = QLabel(self._title)
        self._title_lbl.setObjectName("stage_title")
        header_layout.addWidget(self._title_lbl)
        header_layout.addStretch()

        self._badge = QLabel()
        self._badge.setFixedWidth(88)
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(self._badge)

        frame_layout.addWidget(self._header)

        # ── Separator ─────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("card_separator")
        frame_layout.addWidget(sep)

        # ── Content ───────────────────────────────────────────────────────
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(16, 12, 16, 16)
        self._content_layout.setSpacing(8)
        frame_layout.addWidget(self._content)

        # Wire header click
        self._header.mousePressEvent = lambda _: self._toggle()

        self._refresh_badge()

    # ── Public API ────────────────────────────────────────────────────────

    def set_status(self, status: str):
        self._status = status
        self._refresh_badge()

    def set_card_enabled(self, enabled: bool):
        self._content.setEnabled(enabled)
        alpha = "1.0" if enabled else "0.35"
        self._title_lbl.setStyleSheet(
            f"color: #c8c8e0; font-weight: bold; font-size: 10pt; opacity: {alpha};")

    def add_info_button(self, title: str, text: str):
        """Insert an ℹ button into the stage header, just before the badge."""
        btn = InfoButton(title, text, self._header)
        layout = self._header.layout()
        # Insert before the last item (the badge)
        layout.insertWidget(layout.count() - 1, btn)

    def add_widget(self, widget: QWidget):
        self._content_layout.addWidget(widget)

    def add_layout(self, layout):
        self._content_layout.addLayout(layout)

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    def expand(self):
        self._collapsed = False
        self._content.setVisible(True)
        self._arrow.setText("▼")

    def collapse(self):
        self._collapsed = True
        self._content.setVisible(False)
        self._arrow.setText("▶")

    # ── Internal ─────────────────────────────────────────────────────────

    def _toggle(self):
        if self._collapsed:
            self.expand()
        else:
            self.collapse()

    def _refresh_badge(self):
        label, bg, fg = _BADGE.get(self._status, _BADGE[WAITING])
        self._badge.setText(label)
        self._badge.setStyleSheet(
            f"background:{bg}; color:{fg}; border-radius:3px; "
            f"padding:2px 6px; font-size:9pt; font-weight:bold;")
