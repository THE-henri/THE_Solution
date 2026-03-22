import sys
import traceback
from PyQt6.QtCore import QThread, pyqtSignal


class _StreamCapture:
    """Redirect print() / stderr to a Qt signal during worker execution."""

    def __init__(self, signal, level: str):
        self._signal = signal
        self._level = level
        self._buf = ""

    def write(self, text: str):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line:
                self._signal.emit(line, self._level)

    def flush(self):
        if self._buf:
            self._signal.emit(self._buf, self._level)
            self._buf = ""


class Worker(QThread):
    """
    Run any callable in a background thread.

    Signals
    -------
    log_signal(message, level)   – captured stdout / stderr lines
    result_signal(object)        – return value of the callable
    error_signal(str)            – exception message on failure
    finished_signal()            – always emitted when thread exits
    """

    log_signal      = pyqtSignal(str, str)
    result_signal   = pyqtSignal(object)
    error_signal    = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, func, **kwargs):
        super().__init__()
        self._func   = func
        self._kwargs = kwargs

    def run(self):
        old_out = sys.stdout
        old_err = sys.stderr
        sys.stdout = _StreamCapture(self.log_signal, "INFO")
        sys.stderr = _StreamCapture(self.log_signal, "ERROR")
        try:
            result = self._func(**self._kwargs)
            self.result_signal.emit(result)
        except Exception as exc:
            sys.stdout.flush()
            sys.stderr.flush()
            self.error_signal.emit(str(exc))
            self.log_signal.emit(traceback.format_exc(), "ERROR")
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = old_out
            sys.stderr = old_err
            self.finished_signal.emit()
