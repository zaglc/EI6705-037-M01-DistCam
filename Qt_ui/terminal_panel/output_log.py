from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QLabel, QTextBrowser, QHeaderView, QVBoxLayout, QTabWidget, 
    QToolBar, QWidget,
)

from Qt_ui.utils import gpc_stream
from Qt_ui.threads import QThread4stdout


class terminal(QToolBar):
    
    def __init__(self,
                 parent: QWidget,
    ):
        super().__init__("terminal", parent)
        self.output_text = QTextBrowser(parent)

        # redirect stdout
        self.std_thread = QThread4stdout()
        self.std_thread.redirect_signal.connect(self.redirect_stdout_slot)
        self.std_thread.start()

        outer = QTabWidget(self)
        outer.setMinimumSize(800, 100)
        outer.addTab(self.output_text, "OUTPUT")
        self.addWidget(outer)

    
    def redirect_stdout_slot(self, suffix: str):
        """
        redirect terminal output to Qt widget
        """

        with gpc_stream.log_buffer.get_lock():
            with gpc_stream.offset.get_lock():
                ofs = gpc_stream.offset.value
            if ofs == 0: return

            text = gpc_stream.log_buffer[:ofs].decode()
            gpc_stream.offset.value = 0

        cursor = self.output_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text+suffix)
        self.output_text.setTextCursor(cursor)
        self.output_text.ensureCursorVisible()