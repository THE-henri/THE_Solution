from pathlib import Path
from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QInputDialog,
)
from PyQt6.QtCore import pyqtSignal


class FolderHeader(QGroupBox):
    """
    Header bar at the top of the main window.

    Rows
    ----
    1. Raw data folder  (read-only source, browse start for data files)
    2. Output folder    (results are written here)
    3. Compound name    (used in plot titles and output filenames)
    4. Preferences      (save / load project preferences)

    Signals
    -------
    raw_folder_changed(Path)    – emitted when raw data folder is set
    output_folder_changed(Path) – emitted when output folder is set
    compound_name_changed(str)  – emitted when compound name is edited
    prefs_save_requested()      – emitted when Save Preferences is clicked
    """

    raw_folder_changed    = pyqtSignal(Path)
    output_folder_changed = pyqtSignal(Path)
    compound_name_changed = pyqtSignal(str)
    prefs_save_requested  = pyqtSignal()
    prefs_load_requested  = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Project", parent)
        self.setObjectName("folder_header")
        self._raw_path:    Path | None = None
        self._output_path: Path | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)

        layout.addLayout(self._make_folder_row(
            "Raw data folder:", "raw",
            "Read-only source data. The tool never writes here.",
        ))
        layout.addLayout(self._make_folder_row(
            "Output folder:", "out",
            "Results, plots, and master CSVs are written here.",
            show_new_btn=True,
        ))
        layout.addLayout(self._make_compound_row())
        layout.addLayout(self._make_prefs_row())

    def _make_folder_row(self, label_text: str, key: str, tooltip: str,
                         show_new_btn: bool = False) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        lbl = QLabel(label_text)
        lbl.setFixedWidth(130)
        row.addWidget(lbl)

        edit = QLineEdit()
        edit.setPlaceholderText("Select folder…")
        edit.setToolTip(tooltip)
        edit.setReadOnly(True)
        row.addWidget(edit)

        btn_browse = QPushButton("Browse")
        btn_browse.setFixedWidth(72)
        row.addWidget(btn_browse)

        if show_new_btn:
            btn_new = QPushButton("New…")
            btn_new.setFixedWidth(56)
            btn_new.setToolTip("Create a new output folder")
            btn_new.clicked.connect(self._create_output_folder)
            row.addWidget(btn_new)

        if key == "raw":
            self._raw_edit = edit
            btn_browse.clicked.connect(lambda: self._browse("raw"))
        else:
            self._out_edit = edit
            btn_browse.clicked.connect(lambda: self._browse("out"))

        return row

    def _make_compound_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        lbl = QLabel("Compound name:")
        lbl.setFixedWidth(130)
        row.addWidget(lbl)

        self._compound_edit = QLineEdit()
        self._compound_edit.setPlaceholderText(
            "e.g. AZA-SO2Me  —  used in plot titles and output filenames")
        self._compound_edit.setMaximumWidth(360)
        self._compound_edit.textChanged.connect(self.compound_name_changed)
        row.addWidget(self._compound_edit)
        row.addStretch()

        return row

    def _make_prefs_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        lbl = QLabel("Preferences:")
        lbl.setFixedWidth(130)
        row.addWidget(lbl)

        self._prefs_status_lbl = QLabel("No preferences loaded.")
        self._prefs_status_lbl.setObjectName("warning_label")
        row.addWidget(self._prefs_status_lbl)
        row.addStretch()

        btn_load = QPushButton("Load preferences…")
        btn_load.setFixedWidth(140)
        btn_load.setToolTip("Load a preferences file (.json) from any location")
        btn_load.clicked.connect(self.prefs_load_requested)
        row.addWidget(btn_load)

        btn_save = QPushButton("Save preferences…")
        btn_save.setFixedWidth(140)
        btn_save.setToolTip("Save current panel settings to a preferences file (.json)")
        btn_save.clicked.connect(self.prefs_save_requested)
        row.addWidget(btn_save)

        return row

    def set_prefs_status(self, text: str, ok: bool = True):
        """Update the small status label next to the Preferences row."""
        self._prefs_status_lbl.setText(text)
        self._prefs_status_lbl.setObjectName(
            "info_label" if ok else "warning_label")
        # Re-apply style so the object-name colour takes effect
        self._prefs_status_lbl.style().unpolish(self._prefs_status_lbl)
        self._prefs_status_lbl.style().polish(self._prefs_status_lbl)

    def _browse(self, key: str):
        folder = QFileDialog.getExistingDirectory(self, "Select folder", "")
        if not folder:
            return
        path = Path(folder)
        if key == "raw":
            self._raw_path = path
            self._raw_edit.setText(str(path))
            self.raw_folder_changed.emit(path)
        else:
            self._output_path = path
            self._out_edit.setText(str(path))
            self.output_folder_changed.emit(path)

    def _create_output_folder(self):
        parent_dir = QFileDialog.getExistingDirectory(
            self, "Select parent directory for new output folder", "")
        if not parent_dir:
            return
        name, ok = QInputDialog.getText(self, "New output folder", "Folder name:")
        if not ok or not name.strip():
            return
        path = Path(parent_dir) / name.strip()
        path.mkdir(parents=True, exist_ok=True)
        self._output_path = path
        self._out_edit.setText(str(path))
        self.output_folder_changed.emit(path)

    @property
    def raw_path(self) -> Path | None:
        return self._raw_path

    @property
    def output_path(self) -> Path | None:
        return self._output_path

    @property
    def compound_name(self) -> str:
        return self._compound_edit.text().strip()
