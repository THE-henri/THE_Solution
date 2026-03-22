from PyQt6.QtWidgets import (
    QPushButton, QDialog, QVBoxLayout, QLabel,
    QDialogButtonBox, QFrame,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class InfoDialog(QDialog):
    def __init__(self, title: str, text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.WindowCloseButtonHint
        )
        self.setMinimumWidth(320)
        self.setMaximumWidth(500)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 14)
        layout.setSpacing(10)

        title_lbl = QLabel(title)
        f = QFont()
        f.setBold(True)
        f.setPointSize(10)
        title_lbl.setFont(f)
        title_lbl.setStyleSheet("color: #5b8dee;")
        layout.addWidget(title_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("card_separator")
        layout.addWidget(sep)

        body = QLabel(text)
        body.setWordWrap(True)
        body.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        body.setStyleSheet("color: #c0c0d0; line-height: 1.4;")
        layout.addWidget(body)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)


class InfoButton(QPushButton):
    """Small circular ℹ button that opens a descriptive dialog when clicked."""

    def __init__(self, title: str, text: str, parent=None):
        super().__init__("ℹ", parent)
        self._title = title
        self._text  = text
        self.setObjectName("info_btn")
        self.setFixedSize(20, 20)
        self.setToolTip(f"Info: {title}")
        self.clicked.connect(self._show)

    def _show(self):
        dlg = InfoDialog(self._title, self._text, self)
        dlg.exec()
