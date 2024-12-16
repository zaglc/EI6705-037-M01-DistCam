import time
from multiprocessing import Queue
from queue import Queue as TQueue

import numpy as np
from PyQt6.QtCore import (
    QMutex,
    QMutexLocker,
    QObject,
    QThread,
    QWaitCondition,
    pyqtSignal,
)

from Qt_ui.utils import (
    FV_CAPTURE_IMAGE_Q,
    FV_FLIP_MODEL_ENABLE_Q,
    FV_FLIP_SIMU_STREAM_Q,
    FV_FRAME_PROC_READY_F,
    FV_PKGLOSS_OCCUR_F,
    FV_PTZ_CTRL_Q,
    FV_QTHREAD_PAUSE_Q,
    FV_QTHREAD_READY_Q,
    FV_RECORD_VIDEO_Q,
    FV_RUNNING,
    FV_STOP,
    FV_SWITCH_CHANNEL_Q,
    gpc_stream,
)


class QThread4VideoDisplay(QThread):
    """
    QThread possessed by each frame window
    in charge of sending msg such as capture img, updating frame window
    """

    send_signal = pyqtSignal()
    switch_btn_recover_signal = pyqtSignal()
    realtime_tab_singal = pyqtSignal(tuple)
    camera_capture_recover_signal = pyqtSignal()

    def __init__(
        self,
        thread_id: int,
        frame_queue: Queue,
        command_queue: Queue,
        loc_frame_queue: TQueue,
        parent: QObject = None,
    ) -> None:
        """
        Args: the same as args in `display.py`
        """

        super().__init__(parent)
        self.name = f"QThread4VideoDisplay-{thread_id}"
        self.id = thread_id
        self.frame_queue = frame_queue
        self.command_queue = command_queue
        self.loc_frame_queue = loc_frame_queue

        # ctrl info, when no signal received, it is None
        self.ctrl_info = None

        # cv2 videocapture has read buffer of size 3
        self.frame_cache_len = 3

        # QLock protecting internal, infrequently called variables
        self.switch_cam_lock = QMutex()

        # switch camera source
        self.switch_flag = False
        self.vid_src_info_tuple = None

        # internal variable
        self.need_refresh_cam_flag = False
        self.model_flag = False

        self.display_real_flag = 0
        self.camera_active = 0
        self.record_active = 0

        self.pause_flag = False
        self.is_paused = True
        self.is_running = True

    def run(self):
        """
        Main loop for QThread4VideoDisplay
        """

        print(f"{self.name} launched")
        time0 = time.time()
        while True:

            # fetch context and destory when necessary
            self.switch_cam_lock.lock()
            # action: simultaneous streaming
            refresh_cam_flag = self.need_refresh_cam_flag
            self.need_refresh_cam_flag = False
            # buttons: up, down, left, right...
            ctrl_info = self.ctrl_info
            ctrl_flag = ctrl_info is not None
            self.ctrl_info = None
            # button: video source switching
            switch_flag = self.switch_flag
            self.switch_flag = False
            vid_tuple = self.vid_src_info_tuple
            self.vid_src_info_tuple = None
            # action: model inference on or off
            model_flag = self.model_flag
            self.model_flag = False
            # button: camera capture
            cam_f = self.camera_active
            self.camera_active = 0
            # button: camera record
            rec_f = self.record_active
            self.record_active = 0
            # action: real-time video preview on or off
            pause_flag = self.pause_flag
            self.pause_flag = False
            self.switch_cam_lock.unlock()

            # get frame from frame_queue
            frame, (ret_status, ret_val) = self.frame_queue.get()
            drop_flag = ret_status == FV_PKGLOSS_OCCUR_F

            # trigger frame window image updating
            if not self.is_paused:
                self.loc_frame_queue.put(frame)
                self.send_signal.emit()
                # if len(frame.shape) == 3:
                #     self.is_paused = True

            # prepare command for frame_process, each period <--> one command
            ret_cmd = FV_QTHREAD_READY_Q
            command_msg = None
            if pause_flag:
                self.is_paused = not self.is_paused
                ret_cmd = FV_QTHREAD_PAUSE_Q
            elif switch_flag:
                command_msg = vid_tuple
                self.switch_btn_recover_signal.emit()
                ret_cmd = FV_SWITCH_CHANNEL_Q
            elif refresh_cam_flag:
                ret_cmd = FV_FLIP_SIMU_STREAM_Q
            elif model_flag:
                ret_cmd = FV_FLIP_MODEL_ENABLE_Q
            elif ctrl_flag:
                command_msg = ctrl_info
                ret_cmd = FV_PTZ_CTRL_Q
            elif cam_f:
                self.camera_capture_recover_signal.emit()
                ret_cmd = FV_CAPTURE_IMAGE_Q
            elif rec_f:
                ret_cmd = FV_RECORD_VIDEO_Q

            time4 = time.time()
            # print(time4-time0)

            # trigger data table updating
            self.realtime_tab_singal.emit(
                (self.id, max(time4 - time0, 0.02), ret_val if drop_flag else 0, self.display_real_flag == 0)
            )
            self.display_real_flag = (self.display_real_flag + 1) % 5
            time0 = time4

            # write command to frame_main
            self.command_queue.put((FV_RUNNING if self.is_running else FV_STOP, ret_cmd, command_msg))
            if not self.is_running:
                break
        print(f"{self.name} normally quit")


class QThread4stdout(QThread):
    """
    QThread for output redirection: from terminal to text widget
    """

    redirect_signal = pyqtSignal(str)

    def __init__(
        self,
        parent: QObject = None,
    ) -> None:

        super().__init__(parent)
        self.is_running = True
        self.name = "QThread4stdout"

    def run(self):
        """
        Main loop for QThread4stdout
        """

        print(f"{self.name} launched")
        while True:
            text = gpc_stream.log_buffer.get()
            self.redirect_signal.emit(text)
            if not self.is_running:
                break
        print(f"{self.name} normally quit")
