from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QPushButton
from typing import List


class ctrl_btn(QPushButton):
    def __init__(self,
                 parent: QtWidgets.QWidget,
                 icon: str,
                 name: str,
                 pos: tuple,
                 cmd: int,
    ):
        super().__init__(parent=parent)
        self.setGeometry(QtCore.QRect(*pos))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.setFont(font)
        self.setText("")
        self.setIcon(QtGui.QIcon(icon))
        self.setObjectName(name)
        self.command = cmd

