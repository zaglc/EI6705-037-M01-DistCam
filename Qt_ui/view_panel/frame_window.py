import time
import json
from multiprocessing import Queue
from queue import Queue as TQueue

import numpy as np
from PyQt6.QtCore import QMutex, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QGridLayout, QGroupBox, QLabel, QPushButton, QWidget

from Qt_ui.childwins.save_prefer import ResCropWidget
from Qt_ui.threads import QThread4VideoDisplay
from Qt_ui.utils import BOX_JSON_PATH


class frame_win(QWidget):
    """
    small window for single camera in view panel
    """

    ctrl_select_btn_signal = pyqtSignal(tuple)

    def __init__(
        self,
        id: int,
        parent: QWidget,
        frame_queue: Queue,
        command_queue: Queue,
        name: str,
        min_size: tuple,
    ):
        """
        Args:
            parent: main window
            chan_num: channel number for this camera
            id: id for this window, identical to camera id
            frame_buffer: shared memory storing single frame
            thread_param: parameters for starting internal Qthread
            min_size: minimal size for frame label widget
        """

        # resize the figure size according to ratio(camera)
        super().__init__(parent=parent)
        self.fm_qss = {
            "common": "QLabel{border-style:solid;border-width: 2px;border-color:rgba(0,0,0,150)}",
            "cap": "QLabel{border-style:solid;border-width: 2px;border-color:rgba(255,0,0,150)}",
            "ctrl": "QLabel{border-style:solid;border-width: 2px;border-color:rgba(0,0,255,150)}",
            "double": "QLabel{border-style:solid;border-width: 2px;border-color:rgba(255,0,255,150)}",
        }

        self.id = id
        self.frame_queue = frame_queue
        self.command_queue = command_queue

        self.is_selected_cap = False
        self.is_selected_ctrl = False
        self.switching = False
        self._loc_frame_queue = TQueue()

        # save preview fig
        self._lock = QMutex()
        self._preview_image: QImage | None = None
        self.save_freq = [0, 50]

        self.frame = QLabel(parent=parent)
        self.frame.setMinimumSize(*min_size)
        self.frame.setScaledContents(True)
        self.frame.setStyleSheet("QLabel{border-style: solid;border-width: 2px;border-color: rgba(0, 0, 0, 150)}")

        # button for capture vedio/image
        self.capture_btn = QPushButton(parent=parent)
        self.capture_btn.setText("Select")
        self.capture_btn.clicked.connect(self.selected_cap_slot)

        # button for switch channel
        self.switch_cha = QPushButton(parent=parent)
        self.switch_cha.setText("Switch")

        # button for PTZ control
        self.control_btn = QPushButton(parent=parent)
        self.control_btn.setText("Ctrl")
        self.control_btn.clicked.connect(self.selected_ctrl_slot)

        # button for switching to single view
        self.single_view_btn = QPushButton(parent=parent)
        self.single_view_btn.setText("View")

        align = Qt.AlignmentFlag.AlignCenter
        grid = QGridLayout()
        grid.setSpacing(5)
        grid.addWidget(self.frame, 0, 0, 3, 0, align)
        grid.addWidget(self.single_view_btn, 4, 0)
        grid.addWidget(self.capture_btn, 4, 1)
        grid.addWidget(self.control_btn, 4, 2)
        grid.addWidget(self.switch_cha, 4, 3)

        self.groupbox = QGroupBox(name, parent=parent)
        self.groupbox.setLayout(grid)
        self.groupbox.setStyleSheet("QGroupBox{font:30;}")
        grid2 = QGridLayout()
        grid2.addWidget(self.groupbox, 0, 0)
        self.setLayout(grid2)

        self.frame_thread = QThread4VideoDisplay(self.id, self.frame_queue, self.command_queue, self._loc_frame_queue)
        self.frame_thread.send_signal.connect(self.switch_frame_slot)
        self.frame_thread.switch_btn_recover_signal.connect(self.recover_switch_cam_slot)
        self.switch_cam_lock = self.frame_thread.switch_cam_lock
        self.frame_thread.start()

    def selected_cap_slot(self):
        """
        slot function for select button, for capture and record
        """

        if not self.is_selected_cap:
            self.capture_btn.setStyleSheet("QPushButton{color:rgb(255,0,0)}")
            self.frame.setStyleSheet(self.fm_qss["double"] if self.is_selected_ctrl else self.fm_qss["cap"])
            self.is_selected_cap = True
        else:
            self.capture_btn.setStyleSheet("QPushButton{color:rgb(0,0,0)}")
            self.frame.setStyleSheet(self.fm_qss["ctrl"] if self.is_selected_ctrl else self.fm_qss["common"])
            self.is_selected_cap = False

    def selected_ctrl_slot(self):
        """
        slot function for ctrl button, for PTZ control
        """

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

    # slot for switch button
    def switch_cam_slot(self, source_type, source_info):
        """
        slot function for switch button,
        changing channel between RGB and THER if possible

        Args:
            source_type (str): local-vid, ip-camera, hikvision
            source_info (dict): info of source
        """

        self.switch_cam_lock.lock()
        self.frame_thread.switch_flag = True
        self.frame_thread.vid_src_info_tuple = (source_type, source_info)
        self.switch_cam_lock.unlock()

    def switch_frame_slot(self):
        """
        slot function for thread Qsingal to change displaying image
        """

        frame: np.ndarray = self._loc_frame_queue.get()
        if len(frame.shape) < 3:
            return

        time0 = time.time()
        shape = frame.shape
        w, h = shape[1], shape[0]
        image = QImage(frame, w, h, QImage.Format.Format_BGR888)
        time2 = time.time()
        self.frame.setPixmap(QPixmap.fromImage(image))
        time3 = time.time()
        # if self.save_freq[0] % 10 == 0:
        #     print(f"{self.id}---resize: {round(1000*(time2 - time0), 3)} ms, set frame: {round(1000*(time3 - time2), 3)} ms")

        if self.save_freq[0] == 0:
            self._lock.lock()
            self._preview_image = [image, [w, h]]
            self._lock.unlock()

        self.save_freq[0] += 1
        if self.save_freq[0] == self.save_freq[1]:
            self.save_freq[0] = 0

    def recover_switch_cam_slot(self, resolution: tuple):
        """
        slot function for recover switch button
        """

        name = self.groupbox.title()
        box_obj = ResCropWidget(None, self.id, list(resolution), name, None)
        dicts = box_obj.gather_infos()
        with open(BOX_JSON_PATH, "r") as f:
            box_infos = json.load(f)

        box_infos["box_list"].update({name: dicts})
        with open(BOX_JSON_PATH, "w") as f:
            json.dump(box_infos, f, indent=4)

        self.switch_cha.setEnabled(True)

    # TODO: 初始高度比比例更大怎么办
    def resizeEvent(self, event):
        """
        resizeEvent overload
        when size of frame_win changes, auto adjust widget label showing image
        """

        # o_w = self.control_btn.width() + self.capture_btn.width() + self.switch_cha.width() + self.single_view_btn.width() + 15
        # o_h = self.frame.height()
        # print(f"{self.groupbox.width()}-{self.groupbox.height()}-{o_w}-{o_h}--{(self.control_btn.width()+5)*4-5}")
        # if o_w > 1.8*o_h:
        #     w = int(o_h * 1.8)
        #     self.frame.setFixedWidth(w)
        # else:
        #     h = int(o_w / 1.8)
        #     self.frame.setFixedWidth(o_w)
        #     self.frame.setFixedHeight(h)

        w = int(self.frame.height() * 1.8)
        self.frame.setFixedWidth(w)
        super().resizeEvent(event)
