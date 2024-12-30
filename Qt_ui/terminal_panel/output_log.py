import datetime
import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTabWidget,
    QTextBrowser,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from Qt_ui.threads import QThread4stdout


class terminal(QToolBar):
    def __init__(
        self,
        parent: QWidget,
        curtime: datetime.datetime,
        need_log: bool = True,
    ):
        super().__init__("terminal", parent)
        self.output_text = QTextBrowser(parent)

        # redirect stdout
        self.std_thread = QThread4stdout()
        self.std_thread.redirect_signal.connect(self.redirect_stdout_slot)
        self.std_thread.start()
        self.start_time = curtime
        self.log_dir = os.path.join("data", curtime.strftime("%Y-%m-%d_%H-%M-%S"))
        self.log_fn = "output.log"
        self.need_log = need_log

        if self.need_log and not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        outer = QTabWidget(self)
        outer.setMinimumSize(800, 100)
        outer.addTab(self.output_text, "OUTPUT")
        self.addWidget(outer)

        # color for process label: e.g. [FRAME 1] in blue
        self.colors = {"FRAME": "blue", "MAIN": "red", "MODEL": "green"}

    def redirect_stdout_slot(self, text: str):
        """
        redirect terminal output to Qt widget
        [time] [process] content, separate by '\t'
        """

        def _add_html_color_tag(text: str, color: str) -> str:

            return f"<font color={color}>{text}</font>"

        if text != "" and text != "\n":
            cost = datetime.datetime.now() - self.start_time
            cost = cost.total_seconds()
            text = f"[{cost:.3f}]@{text}"

            cursor = self.output_text.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.EndOfLine)
            for i, t in enumerate(text.split("@")):
                if i == 1:
                    cursor.insertHtml(_add_html_color_tag(f"[{t}]", self.colors[t.split(" ")[0]]))
                    cursor.insertText("\t")
                elif i == 0:
                    cursor.insertHtml(_add_html_color_tag(t, "black"))
                    cursor.insertText("\t")
                else:
                    rows = t.split("\n")
                    for idx, tt in enumerate(rows):
                        if idx != 0:
                            cursor.insertText("\t" * 2)
                        cursor.insertHtml(_add_html_color_tag(tt, "black"))
                        cursor.insertText("\n")

            self.output_text.setTextCursor(cursor)
            self.output_text.ensureCursorVisible()

            if self.need_log:
                with open(os.path.join(self.log_dir, self.log_fn), "a") as f:
                    f.write(text.replace("@", "\t") + "\n")
