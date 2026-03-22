from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class SplashScreen(QDialog):
    """
    Welcome / disclaimer screen shown once at startup.
    Clicking "I Agree" accepts and opens the main window.
    The window close button (X) exits the application.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("QY Tool")
        self.setMinimumSize(500, 400)
        self.setMaximumSize(620, 500)
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.WindowCloseButtonHint
        )
        self._agreed = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 44, 48, 32)
        layout.setSpacing(16)

        # Title
        title = QLabel("QY Tool")
        f = QFont()
        f.setPointSize(26)
        f.setBold(True)
        title.setFont(f)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setObjectName("splash_title")
        layout.addWidget(title)

        subtitle = QLabel("Quantum Yield & Photochemistry Analysis")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setObjectName("splash_subtitle")
        layout.addWidget(subtitle)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("card_separator")
        layout.addWidget(sep)

        body = QLabel(
            "This tool assists in the analysis of photoswitchable compounds, "
            "including half-life determination, quantum yield calculation, "
            "actinometry, extinction coefficient fitting, and thermal analysis "
            "(Arrhenius / Eyring).\n\n"
            "Raw data folders are read-only — the tool never modifies source files. "
            "All results are written to the separately configured output folder.\n\n"
            "Please set both folder paths before running any workflow."
        )
        body.setWordWrap(True)
        body.setAlignment(Qt.AlignmentFlag.AlignJustify | Qt.AlignmentFlag.AlignTop)
        body.setObjectName("splash_body")
        layout.addWidget(body)

        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn = QPushButton("I Agree  →")
        btn.setObjectName("accent")
        btn.setFixedSize(140, 38)
        btn.clicked.connect(self._on_agree)
        btn_row.addWidget(btn)
        layout.addLayout(btn_row)

        note = QLabel(
            "By continuing you confirm that this tool is used for research purposes "
            "and is provided without warranty of any kind."
        )
        note.setWordWrap(True)
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setObjectName("splash_note")
        layout.addWidget(note)

    def _on_agree(self):
        self._agreed = True
        self.accept()

    @property
    def agreed(self) -> bool:
        return self._agreed
