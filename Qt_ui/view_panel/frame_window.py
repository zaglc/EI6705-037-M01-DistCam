from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QLabel, QPushButton, QGridLayout, QGroupBox, QSizePolicy
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QPixmap
from typing import List
import numpy as np, ctypes

from Qt_ui.threads import QThread4VideoDisplay


class frame_win(QtWidgets.QWidget):

    ctrl_select_btn_signal = pyqtSignal(tuple)
    def __init__(self,
                 parent: QtWidgets.QWidget,
                 chan_num: int,
                 id: int,
                 frame_buffer,
                 thread_param: list,
    ):
        # resize the figure size according to ratio(camera)
        super().__init__(parent=parent)
        self.fm_qss = {
            "common": "QLabel{border-style:solid;border-width: 2px;border-color:rgba(0,0,0,150)}",
            "cap": "QLabel{border-style:solid;border-width: 2px;border-color:rgba(255,0,0,150)}",
            "ctrl": "QLabel{border-style:solid;border-width: 2px;border-color:rgba(0,0,255,150)}",
            "double": "QLabel{border-style:solid;border-width: 2px;border-color:rgba(255,0,255,150)}",
        }

        self.id = id
        self.frame_buffer = frame_buffer
        self.is_selected_cap = False
        self.is_selected_ctrl = False
        self.streaming = False
        self.switching = False
        self.switch_cam_lock = thread_param[-1]
        self.frame = QLabel(parent=parent)

        self.frame.setScaledContents(True)
        self.frame.setStyleSheet("QLabel{border-style: solid;border-width: 2px;border-color: rgba(0, 0, 0, 150)}")
        
        # button for capture vedio/image
        self.capture_btn = QPushButton(parent=parent)
        self.capture_btn.setText("Select")
        self.capture_btn.clicked.connect(self.selected_cap_slot)

        # button for switch channel
        self.switch_cha = QPushButton(parent=parent)
        self.switch_cha.setText("Switch")
        self.switch_cha.clicked.connect(self.switch_cam_slot)
        if chan_num == 1:
            self.switch_cha.setEnabled(False)

        # button for PTZ control
        self.control_btn = QPushButton(parent=parent)
        self.control_btn.setText("Control")
        self.control_btn.clicked.connect(self.selected_ctrl_slot)

        # button for switching to single view
        self.single_view_btn = QPushButton(parent=parent)
        self.single_view_btn.setText("View")
        
        grid = QGridLayout()
        grid.setSpacing(5)
        grid.addWidget(self.frame, 0, 0, 3, 0)
        grid.addWidget(self.single_view_btn, 4, 0)
        grid.addWidget(self.capture_btn, 4, 1)
        grid.addWidget(self.control_btn, 4, 2)
        grid.addWidget(self.switch_cha, 4, 3)

        posfix = "" if chan_num == 1 else "+THER"
        groupbox = QGroupBox(f"CAM {id+1}: RGB{posfix}", parent=parent)
        groupbox.setLayout(grid)
        groupbox.setStyleSheet("QGroupBox{font:30;}")
        grid2 = QGridLayout()
        grid2.addWidget(groupbox, 0, 0)
        self.setLayout(grid2)

        self.frame_thread = QThread4VideoDisplay(*thread_param)
        self.frame_thread.switch_btn_recover_signal.connect(self.recover_switch_cam_slot)

    # slot functions
    def selected_cap_slot(self):
        if not self.is_selected_cap:
            self.capture_btn.setStyleSheet("QPushButton{color:rgb(255,0,0)}")
            self.frame.setStyleSheet(self.fm_qss["double"] if self.is_selected_ctrl else self.fm_qss["cap"])
            self.is_selected_cap = True
        else:
            self.capture_btn.setStyleSheet("QPushButton{color:rgb(0,0,0)}")
            self.frame.setStyleSheet(self.fm_qss["ctrl"] if self.is_selected_ctrl else self.fm_qss["common"])
            self.is_selected_cap = False     

    def selected_ctrl_slot(self):
        if not self.is_selected_ctrl:
            self.control_btn.setStyleSheet("QPushButton{color:rgb(0,0,255)}")
            self.frame.setStyleSheet(self.fm_qss["double"] if self.is_selected_cap else self.fm_qss["ctrl"])
            self.is_selected_ctrl = True
            self.ctrl_select_btn_signal.emit((self.id, 1))
        else:
            self.control_btn.setStyleSheet("QPushButton{color:rgb(0,0,0)}")
            self.frame.setStyleSheet(self.fm_qss["cap"] if self.is_selected_cap else self.fm_qss["common"])
            self.is_selected_ctrl = False
            self.ctrl_select_btn_signal.emit((self.id, 0))

    def switch_cam_slot(self):
        self.switch_cha.setEnabled(False)
        self.switch_cam_lock.lock()
        self.frame_thread.switch_flag = True
        self.switch_cam_lock.unlock()

    def switch_frame_slot(self, shape):
        with self.frame_buffer.get_lock():
            pics = np.frombuffer(self.frame_buffer.get_obj(), dtype=ctypes.c_uint8)
        nbytes = shape[0] * shape[1] * shape[2]
        image = QtGui.QImage(pics[:nbytes].reshape(shape),shape[1],shape[0],QtGui.QImage.Format.Format_RGB888)
        # w, h = self.frame.width(), self.frame.height()
        # self.frame.setPixmap(QPixmap.fromImage(image).scaled(w, h,))
        self.frame.setPixmap(QPixmap.fromImage(image))

    def recover_switch_cam_slot(self):
        self.switch_cha.setEnabled(True)


    # TODO: may be deleted
    def response_main_click_slot(self):
        if not self.streaming:
            self.frame_thread.send_signal.connect(self.switch_frame_slot)
            self.frame_thread.start()
            self.streaming = True
        else:
            self.frame_thread.send_signal.disconnect(self.switch_frame_slot)
            self.streaming = False