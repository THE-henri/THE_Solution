from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from app_info import APP_NAME, APP_SUBTITLE, __version__


class SplashScreen(QDialog):
    """
    Welcome / disclaimer screen shown once at startup.
    Clicking "I Agree" accepts and opens the main window.
    The window close button (X) exits the application.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(560, 560)
        self.setMaximumSize(680, 720)
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
        title = QLabel(APP_NAME)
        f = QFont()
        f.setPointSize(26)
        f.setBold(True)
        title.setFont(f)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setObjectName("splash_title")
        layout.addWidget(title)

        subtitle = QLabel(f"{APP_SUBTITLE}  ·  v{__version__}")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setObjectName("splash_subtitle")
        layout.addWidget(subtitle)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("card_separator")
        layout.addWidget(sep)

        body = QLabel(
            f"<p>{APP_NAME} was developed in the <b>Hecht Lab</b> for the analysis of "
            "UV/Vis spectroscopy data of photoswitchable compounds: extinction "
            "coefficients, spectra, half-lives, Arrhenius / Eyring analysis, "
            "actinometry and quantum yields.</p>"
            "<p><b>Research use.</b> The program is intended for academic research. "
            "It is licensed under the MIT License and provided \"as is\", without "
            "warranty of any kind.</p>"
            "<p><b>Your responsibility.</b> You are responsible for the data you "
            "generate with this program. Critically check every fit, plot and "
            "number before you use or publish it.</p>"
            "<p>Raw data folders are read-only — the program never modifies source "
            "files. All results are written to the output folder you select.</p>"
            "<p>More information, including how to cite the program and the methods "
            "it uses, can be found in the <b>Handbook → GENERAL</b> section.</p>"
        )
        body.setTextFormat(Qt.TextFormat.RichText)
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
            "By clicking \"I Agree\" you confirm that you have read and accept "
            "these conditions."
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
