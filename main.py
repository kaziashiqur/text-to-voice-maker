import sys

from app import config  # noqa: F401  (creates the app folders before anything else loads)

# pythonw.exe has no console; voice libraries print progress, so send it to a log file.
if sys.stdout is None or sys.stderr is None:
    _log = open(config.CACHE / "app.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or _log
    sys.stderr = sys.stderr or _log
from PySide6.QtWidgets import QApplication

from app.ui import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("Text to Voice Maker")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
