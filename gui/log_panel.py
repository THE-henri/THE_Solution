import sys
from datetime import datetime
from pathlib import Path
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel
from PyQt6.QtGui import QTextCursor
from PyQt6.QtCore import pyqtSlot

_LEVEL_COLORS = {
    "INFO":    "#b0b0cc",
    "WARNING": "#e8a020",
    "ERROR":   "#e84d4d",
    "SUCCESS": "#3cb371",
}


class LogPanel(QWidget):
    """
    Scrollable log at the bottom of the main window.

    Captures main-thread print() output via stdout redirect.
    Worker threads emit log_signal(str, str) which should be
    connected to self.append().

    When set_log_dir() is called, a session log file is opened there and
    every subsequent line is written to it immediately (line-buffered) so
    that crash recovery is possible.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._old_stdout = None
        self._log_file   = None   # open file handle, or None
        self._log_path:  Path | None = None
        self._build_ui()
        self._install_redirect()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(4, 2, 4, 2)
        lbl = QLabel("  Log")
        lbl.setObjectName("log_header")
        header.addWidget(lbl)
        header.addStretch()
        self._log_path_lbl = QLabel("")
        self._log_path_lbl.setObjectName("detected_label")
        self._log_path_lbl.setStyleSheet("font-size:8pt; color:#555577;")
        header.addWidget(self._log_path_lbl)
        btn_clear = QPushButton("Clear")
        btn_clear.setObjectName("plot_tool_btn")
        btn_clear.setFixedHeight(22)
        btn_clear.clicked.connect(self.clear)
        header.addWidget(btn_clear)
        layout.addLayout(header)

        self._edit = QTextEdit()
        self._edit.setReadOnly(True)
        self._edit.setObjectName("log_edit")
        layout.addWidget(self._edit)

    def _install_redirect(self):
        self._old_stdout = sys.stdout
        sys.stdout = _LogStream(self)

    # ── File logging ──────────────────────────────────────────────────────

    def set_log_dir(self, output_path: Path):
        """
        Open a new session log file in <output_path>/logs/.

        Called whenever the output folder is changed.  Any previously open
        log file is closed first.  The filename embeds the session start
        timestamp so each run produces a distinct file.
        """
        self._close_log_file()

        log_dir = output_path / "logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_path = log_dir / f"session_{ts}.log"
            # buffering=1 → line-buffered: every newline flushes to disk
            self._log_file  = open(log_path, "w", encoding="utf-8", buffering=1)
            self._log_path  = log_path
            self._log_path_lbl.setText(f"Logging → {log_path}")
            # Write a header line so the file is non-empty from the start
            self._log_file.write(
                f"# QY Tool session log  —  started {datetime.now().isoformat(timespec='seconds')}\n"
                f"# Output folder: {output_path}\n"
                "# ─────────────────────────────────────────────────────────\n"
            )
        except Exception as exc:
            print(f"WARNING: could not open log file in {log_dir}: {exc}")

    def _close_log_file(self):
        if self._log_file is not None:
            try:
                self._log_file.write(
                    f"\n# Session ended {datetime.now().isoformat(timespec='seconds')}\n")
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None
            self._log_path = None
            self._log_path_lbl.setText("")

    def _write_to_file(self, message: str, level: str):
        if self._log_file is None:
            return
        try:
            ts  = datetime.now().strftime("%H:%M:%S")
            tag = f"[{level:<7}]"
            self._log_file.write(f"{ts}  {tag}  {message}\n")
            # buffering=1 handles flush, but be explicit for safety
            self._log_file.flush()
        except Exception:
            pass

    # ── GUI log ───────────────────────────────────────────────────────────

    @pyqtSlot(str, str)
    def append(self, message: str, level: str = "INFO"):
        color = _LEVEL_COLORS.get(level, _LEVEL_COLORS["INFO"])
        ts  = datetime.now().strftime("%H:%M:%S")
        tag = f"[{level:<7}]"
        # escape HTML special chars
        msg_esc = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = (
            f'<span style="color:#444466">{ts}</span>&nbsp;'
            f'<span style="color:{color}">{tag}&nbsp;{msg_esc}</span>'
        )
        self._edit.append(html)
        self._edit.moveCursor(QTextCursor.MoveOperation.End)
        self._write_to_file(message, level)

    def clear(self):
        self._edit.clear()

    def restore_stdout(self):
        if self._old_stdout is not None:
            sys.stdout = self._old_stdout
        self._close_log_file()


class _LogStream:
    """Minimal file-like object that forwards to LogPanel."""

    def __init__(self, panel: LogPanel):
        self._panel = panel
        self._buf   = ""

    def write(self, text: str):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line:
                self._panel.append(line, "INFO")

    def flush(self):
        if self._buf:
            self._panel.append(self._buf, "INFO")
            self._buf = ""
