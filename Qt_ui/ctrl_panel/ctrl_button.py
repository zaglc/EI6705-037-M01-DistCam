from PyQt6.QtWidgets import QPushButton, QWidget
from PyQt6.QtGui import QFont, QIcon


class ctrl_btn(QPushButton):
    """
    ctrl_btn in ctrl_panel
    """

    def __init__(self,
                 parent: QWidget,
                 icon: str,
                 cmd: int,
                 text: str | None = None,
    ) -> None:
        super().__init__(parent=parent)
        font = QFont()
        font.setPointSize(9)
        self.setFont(font)
        self.setText(text if text is not None else "")
        self.setIcon(QIcon(icon))
        self.command = cmd if cmd > 0 else None
