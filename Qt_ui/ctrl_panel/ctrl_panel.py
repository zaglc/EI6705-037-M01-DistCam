import datetime
import os
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.sharedctypes import SynchronizedBase
from multiprocessing.synchronize import Condition
from typing import Dict

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextBrowser,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from central_monitor.HCNetSDK import (
    DOWN_LEFT,
    DOWN_RIGHT,
    FOCUS_FAR,
    FOCUS_NEAR,
    IRIS_CLOSE,
    IRIS_OPEN,
    PAN_LEFT,
    PAN_RIGHT,
    TILT_DOWN,
    TILT_UP,
    UP_LEFT,
    UP_RIGHT,
    ZOOM_IN,
    ZOOM_OUT,
)
from Qt_ui.ctrl_panel.ctrl_button import ctrl_btn


class ctrl_panel(QToolBar):
    """
    ctrl panel in QT mainwindow
    """

    # signal to be triggered when pushing CAPTURE btn
    camera_btn_signal = pyqtSignal()

    # signal to be triggered when pushing RECORD btn
    record_btn_signal = pyqtSignal()

    # signal to be triggered when pushing other PTZ btns
    ptz_btn_signal = pyqtSignal(int, int, int)

    def __init__(
        self,
        parent: QWidget,
        num_cam: int,
    ) -> None:
        """
        Args:
            parent: main window
            num_cam: total camera number
            conn: one end of Pipe, transmiting command tuple
            cond: condition of multiprocessing, let the process sleep
                  when no command occurs
            exec_seq: like that in frame process
        """

        super().__init__("Ctrl Panel", parent=parent)
        self.num_cam = num_cam

        # internal params
        self.if_pressed = False
        self.selected_cam = -1
        self.ctrl_btn_lst: Dict[int:ctrl_btn] = {}
        self.camera_btn: QPushButton
        self.record_btn: QPushButton
        self.camera_btn_defalut_path: str
        self.is_recording = False
        self.support_PTZ = {
            UP_LEFT: (0, 0, 1, 1),
            TILT_UP: (0, 1, 1, 1),
            UP_RIGHT: (0, 2, 1, 1),
            PAN_LEFT: (1, 0, 1, 1),
            PAN_RIGHT: (1, 2, 1, 1),
            DOWN_LEFT: (2, 0, 1, 1),
            TILT_DOWN: (2, 1, 1, 1),
            DOWN_RIGHT: (2, 2, 1, 1),
            ZOOM_OUT: (0, 4, 1, 1),
            ZOOM_IN: (0, 5, 1, 1),
            FOCUS_FAR: (1, 4, 1, 1),
            FOCUS_NEAR: (1, 5, 1, 1),
            IRIS_CLOSE: (2, 4, 1, 1),
            IRIS_OPEN: (2, 5, 1, 1),
        }

    def init_PTZ_btns(self, parent: QWidget):
        """
        initialize control button zone
        """

        for cmd in self.support_PTZ:
            self.ctrl_btn_lst[cmd] = ctrl_btn(parent, f"Qt_ui/ctrl_panel/icon/icon_PTZ_{cmd}.png", cmd)

        for btn in self.ctrl_btn_lst.values():
            btn: ctrl_btn
            func = partial(self.PTZ_control_slot, cmd=btn.command)
            btn.pressed.connect(func)
            btn.released.connect(func)

        grid = QGridLayout()
        grid.setSpacing(5)
        for pos, btn in zip(list(self.support_PTZ.values()), list(self.ctrl_btn_lst.values())):
            grid.addWidget(btn, *pos)
        space = QLabel(parent)
        grid.addWidget(space, 0, 3, 3, 1)

        groupbox = QGroupBox("PTZ CONTROL", parent=parent)
        groupbox.setLayout(grid)
        groupbox.setStyleSheet("QGroupBox{font:30;}")
        grid2 = QVBoxLayout()
        grid2.addWidget(groupbox)

        self.outer = QWidget(self)
        self.outer.setMinimumSize(150, 300)
        self.outer.setLayout(grid2)
        self.addWidget(self.outer)

    def init_frame_btn(self, parent: QWidget):
        """
        initialize record/capture zone
        """

        self.camera_btn = ctrl_btn(parent, f"Qt_ui/ctrl_panel/icon/icon_CAPTURE.png", -1, "CAPTURE")
        self.camera_btn.clicked.connect(self.camera_btn_slot)
        self.record_btn = ctrl_btn(parent, f"Qt_ui/ctrl_panel/icon/icon_RECORD.png", -1, "RECORD")
        self.record_btn.clicked.connect(self.record_btn_slot)

        grid = QGridLayout()
        grid.setSpacing(5)
        grid.addWidget(self.camera_btn, 0, 0, 1, 1)
        grid.addWidget(self.record_btn, 1, 0, 1, 1)
        groupbox = QGroupBox("DATA-SAVING", parent=parent)
        groupbox.setStyleSheet("QGroupBox{font:30;}")
        groupbox.setLayout(grid)

        grid2 = self.outer.layout()
        grid2.addWidget(groupbox)
        grid2.addStretch(1)
        self.outer.setLayout(grid2)

    def PTZ_control_slot(self, cmd):
        """
        slot func invoke after pushing PTZ ctrl button
        """

        if self.if_pressed:
            self.ptz_btn_signal.emit(self.selected_cam, cmd, 1)
            self.if_pressed = False
        else:
            self.ptz_btn_signal.emit(self.selected_cam, cmd, 0)
            self.if_pressed = True

    def camera_btn_slot(self):
        """
        slot func invoke after pushing CAPTURE button
        """

        self.camera_btn_signal.emit()

    def record_btn_slot(self):
        """
        slot func invoke after pushing RECORD button
        """

        self.record_btn_signal.emit()

    def setupUi(self, MainWindow: QMainWindow):
        """
        set all ui to mainwindow
        """

        self.init_PTZ_btns(self)
        self.init_frame_btn(self)

    def set_selected_cam(self, cam_id: int):
        """
        set camera selected for ctrl-buttons
        """

        self.selected_cam = cam_id
