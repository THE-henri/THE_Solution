from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt


class PlaceholderTab(QWidget):
    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl = QLabel(f"{name}\n\nComing soon")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setObjectName("placeholder_label")
        layout.addWidget(lbl)
