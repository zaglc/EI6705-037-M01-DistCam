import os, sys
from PyQt6.QtWidgets import (
    QPushButton, QVBoxLayout, QGridLayout, QGroupBox, QFileDialog, 
    QTextBrowser, QLabel, QWidget, QMainWindow, QToolBar, QApplication
)
from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtGui import QTextCursor
from typing import Dict
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.sharedctypes import SynchronizedBase
from multiprocessing.synchronize import Condition

from central_monitor.HCNetSDK import (
    TILT_UP, TILT_DOWN, PAN_LEFT, PAN_RIGHT, 
    UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT, 
    ZOOM_IN, ZOOM_OUT, FOCUS_NEAR, FOCUS_FAR, 
    IRIS_OPEN, IRIS_CLOSE,
)
from Qt_ui.ctrl_panel.ctrl_button import ctrl_btn
from Qt_ui.utils import gpc_stream

class ctrl_panel(QToolBar):
    """
    ctrl panel in QT mainwindow
    """

    # signal to be triggered when pushing CAPTURE btn
    camera_btn_signal = pyqtSignal(str)

    # singal to be triggered when pushing RECORD btn
    record_btn_signal = pyqtSignal(tuple)

    def __init__(
            self, 
            parent: QWidget, 
            num_cam: int,
            conn: Connection,
            cond: Condition,
            exec_seq: SynchronizedBase,
        ) -> None:
        """
        Args:
            parent: minwin.centralwidgt
            num_cam: total camera number
            conn: one end of Pipe, transmiting command tuple
            cond: condition of multiprocessing, let the process sleep
                  when no command occurs
            exec_seq: like that in frame process
        """

        super().__init__("Ctrl Panel", parent=parent)
        self.num_cam = num_cam
        self.conn = conn
        self.cond = cond
        self.exec_seq = exec_seq

        # internal params
        self.if_pressed = False
        self.selected_cam = -1
        self.ctrl_btn_lst : Dict[int: ctrl_btn] = {}
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
        grid2.addStretch(1)

        self.outer = QWidget(self)
        self.outer.setMinimumSize(150, 750)
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
        self.path_browser_lab = QLabel(parent)
        self.path_browser_lab.setText("Current Active Path")
        self.path_browser_lab.setStyleSheet("QLabel{font:20}")
        self.path_browser = QTextBrowser(parent)
        self.path_browser.setFixedHeight(int(self.path_browser.document().size().height())+5)
        
        self.select_pth_btn = ctrl_btn(parent, f"Qt_ui/ctrl_panel/icon/icon_PTZ_21.png", -1, "SEL_PATH")
        self.select_pth_btn.clicked.connect(self.select_path_btn_slot)
        self.camera_btn_defalut_path = os.getcwd()

        self.output_text = QTextBrowser(parent)
        self.output_text_lab = QLabel(parent)
        self.output_text_lab.setText("Output")
        sys.stdout = gpc_stream
        sys.stdout.newText_signal.connect(self.redirect_stdout_slot)

        grid = QGridLayout()
        grid.setSpacing(5)
        grid.addWidget(self.path_browser_lab, 0, 0, 1, 1)
        grid.addWidget(self.path_browser, 1, 0, 1, 1)
        grid.addWidget(self.select_pth_btn ,2, 0, 1, 1)
        grid.addWidget(self.camera_btn, 3, 0, 1, 1)
        grid.addWidget(self.record_btn, 4, 0, 1, 1)
        grid.addWidget(self.output_text_lab, 5, 0, 1, 1)
        grid.addWidget(self.output_text, 6, 0, 1, 1)
        groupbox = QGroupBox("CAMERA", parent=parent)
        groupbox.setStyleSheet("QGroupBox{font:30;}")
        groupbox.setLayout(grid)

        grid2 = self.outer.layout()
        grid2.addWidget(groupbox)
        grid2.addStretch(3)
        self.outer.setLayout(grid2)


    def PTZ_control_slot(self, cmd):
        """
        slot func invoke after pushing PTZ ctrl button
        """

        while True:
            with self.exec_seq.get_lock():
                if self.exec_seq.value == 1:
                    self.exec_seq.value = 0
                    break
        
        # when there's a ctrl signal, awake the subprocess in charge of ctrl 
        with self.cond:
            self.cond.notify_all()
        
        if self.if_pressed:
            self.conn.send((self.selected_cam, cmd, 1))
            self.if_pressed = False
        else:
            self.conn.send((self.selected_cam, cmd, 0))
            self.if_pressed = True


    def select_path_btn_slot(self):
        """
        slot func invoke after pushing SEL_PATH button
        """

        folder = QFileDialog.getExistingDirectory(self, 'Choose folder to save', directory=self.camera_btn_defalut_path)
        if folder:
            self.path_browser.setText(folder)
            self.path_browser.setFixedHeight(int(self.path_browser.document().size().height())+5)
            self.camera_btn_defalut_path = folder


    def camera_btn_slot(self):   
        """
        slot func invoke after pushing CAPTURE button
        """

        self.camera_btn_signal.emit(self.camera_btn_defalut_path)


    def record_btn_slot(self):
        """
        slot func invoke after pushing RECORD button
        """

        if not self.is_recording:
            self.record_btn_signal.emit((self.camera_btn_defalut_path, 1))
        else:
            self.record_btn_signal.emit(("", 0))


    def setupUi(self, MainWindow: QMainWindow):
        """
        set all ui to mainwindow
        """

        self.init_PTZ_btns(self)
        self.init_frame_btn(self)


    def set_selected_cam(self, val: int):
        self.selected_cam = val


    def redirect_stdout_slot(self, text: str):
        cursor = self.output_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.output_text.setTextCursor(cursor)
        self.output_text.ensureCursorVisible()