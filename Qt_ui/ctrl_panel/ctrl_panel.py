from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QPushButton
from typing import List
from functools import partial
from multiprocessing.connection import Connection

from Qt_ui.ctrl_panel.ctrl_button import ctrl_btn
from central_monitor.HCNetSDK import (
    TILT_UP, TILT_DOWN, PAN_LEFT, PAN_RIGHT, 
    UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT, 
)

class ctrl_panel(QtWidgets.QWidget):
    def __init__(self, parent, conn):
        super().__init__(parent=parent)
        self.ctrl_btn_lst : List[ctrl_btn] = []
        self.conn : Connection = conn
        self.if_pressed = False
        pass

    def init_btns(self, parent: QtWidgets.QWidget, pos_2: tuple):
        # UP_LEFT
        self.ctrl_btn_lst.append(
            ctrl_btn(
                parent, "Qt_ui/ctrl_panel/icon/TILT_UP.png", "UP_LEFT", (pos_2[0]+20, pos_2[1]+20, 40, 40), UP_LEFT)
        )
        # TILT_UP
        self.ctrl_btn_lst.append(
            ctrl_btn(
                parent, "Qt_ui/ctrl_panel/icon/TILT_UP.png", "TILT_UP", (pos_2[0]+70, pos_2[1]+20, 40, 40), TILT_UP)
        )
        # UP_RIGHT
        self.ctrl_btn_lst.append(
            ctrl_btn(
                parent, "Qt_ui/ctrl_panel/icon/TILT_UP.png", "UP_RIGHT", (pos_2[0]+120, pos_2[1]+20, 40, 40), UP_RIGHT)
        )
        # PAN_LEFT
        self.ctrl_btn_lst.append(
            ctrl_btn(
                parent, "Qt_ui/ctrl_panel/icon/TILT_UP.png", "PAN_LEFT", (pos_2[0]+20, pos_2[1]+70, 40, 40), PAN_LEFT)
        )
        # PAN_RIGHT
        self.ctrl_btn_lst.append(
            ctrl_btn(
                parent, "Qt_ui/ctrl_panel/icon/TILT_UP.png", "PAN_RIGHT", (pos_2[0]+120, pos_2[1]+70, 40, 40), PAN_RIGHT)
        )
        # DOWN_LEFT
        self.ctrl_btn_lst.append(
            ctrl_btn(
                parent, "Qt_ui/ctrl_panel/icon/TILT_UP.png", "DOWN_LEFT", (pos_2[0]+20, pos_2[1]+120, 40, 40), DOWN_LEFT)
        )
        # TILT_DOWN
        self.ctrl_btn_lst.append(
            ctrl_btn(
                parent, "Qt_ui/ctrl_panel/icon/TILT_UP.png", "TILT_DOWN", (pos_2[0]+70, pos_2[1]+120, 40, 40), TILT_DOWN)
        )
        # DOWN_RIGHT
        self.ctrl_btn_lst.append(
            ctrl_btn(
                parent, "Qt_ui/ctrl_panel/icon/TILT_UP.png", "DOWN_RIGHT", (pos_2[0]+120, pos_2[1]+120, 40, 40), DOWN_RIGHT)
        )

        for btn in self.ctrl_btn_lst:
            func = partial(self.slot_fun, cmd=btn.command)
            btn.pressed.connect(func)
            btn.released.connect(func)

    def slot_fun(self, cmd):
        if self.if_pressed:
            self.conn.send((cmd, 1))
            self.if_pressed = False
        else:
            self.conn.send((cmd, 0))
            self.if_pressed = True

    def setupUi(self, MainWindow, pos_2):
        centralwidget = MainWindow.centralWidget()
        self.init_btns(centralwidget, pos_2)
        MainWindow.setCentralWidget(centralwidget)



