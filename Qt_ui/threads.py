from PyQt6.QtCore import (
    QThread, pyqtSignal, QObject, QMutex, QWaitCondition, QMutexLocker,
)
import time
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Condition
from multiprocessing.sharedctypes import (
    SynchronizedBase,
)

from Qt_ui.utils import (
    FV_QTHREAD_READY_Q, FV_FRAME_PROC_READY_F, 
    FV_SWITCH_CHANNEL_Q, FV_PKGLOSS_OCCUR_F, 
    FV_CAPTURE_IMAGE_Q, FV_RECORD_VIDEO_Q, 
    FV_FLIP_SIMU_STREAM_Q, FV_FLIP_MODEL_ENABLE_Q,
    FV_QTHREAD_PAUSE_Q,
)
from Qt_ui.utils import gpc_stream


class QThread4VideoDisplay(QThread):
    """
    QThread possessed by each frame window
    in charge of sending msg such as capture img, updating frame window
    """

    send_signal = pyqtSignal(tuple)
    switch_btn_recover_signal = pyqtSignal()
    realtime_tab_singal = pyqtSignal(tuple)
    camera_capture_recover_signal = pyqtSignal()
    camera_record_recover_signal = pyqtSignal()
    
    def __init__(
            self, 
            thread_id: int,
            conn: Connection,
            cond: Condition,
            exec_seq: SynchronizedBase,
            parent: QObject = None,
        ) -> None:
        """
        Args: the same as args in `display.py`
        """

        super().__init__(parent)
        self.id = thread_id
        self.conn = conn
        self.cond = cond
        self.exec_seq = exec_seq
        
        # cv2 videocapture has read buffer of size 3
        self.frame_cache_len = 3
        
        # QLock protecting internal, infrequently called variables
        self.switch_cam_lock = QMutex()
        self.Qconds = QWaitCondition()
        
        # internal variable
        self.switch_flag = False
        self.need_refresh_cam_flag = False
        self.model_flag = False
        self.display_real_flag = 0
        self.camera_active = 0
        self.record_active = 0
        self.is_recording = False

        self.pause_flag = False
    

    def run(self):
        """
        Main loop for QThread4VideoDisplay
        """

        while True:
            time0 = time.time()
            
            # fetch context and destory when necessary
            self.switch_cam_lock.lock()
            refresh_cam_flag = self.need_refresh_cam_flag
            self.need_refresh_cam_flag = False
            switch_flag = self.switch_flag
            self.switch_flag = False
            model_flag = self.model_flag
            self.model_flag = False
            cam_f = self.camera_active
            self.camera_active = 0
            rec_f = self.record_active
            pause_flag = self.pause_flag
            self.switch_cam_lock.unlock()

            # keep rolling when frame process is ready
            # and get command for current frame loop
            while True:
                with self.exec_seq.get_lock():
                    if (self.exec_seq.value == FV_FRAME_PROC_READY_F 
                        or self.exec_seq.value == FV_PKGLOSS_OCCUR_F):
                        
                        drop_flag = self.exec_seq.value == FV_PKGLOSS_OCCUR_F
                        self.exec_seq.value = FV_QTHREAD_READY_Q
                        if pause_flag:
                            self.exec_seq.value = FV_QTHREAD_PAUSE_Q
                            break

                        if switch_flag:
                            self.exec_seq.value = FV_SWITCH_CHANNEL_Q
                        elif refresh_cam_flag:
                            self.exec_seq.value = FV_FLIP_SIMU_STREAM_Q
                        elif model_flag:
                            self.exec_seq.value = FV_FLIP_MODEL_ENABLE_Q

                        if cam_f: 
                            self.exec_seq.value = FV_CAPTURE_IMAGE_Q
                        elif rec_f: 
                            self.exec_seq.value = FV_RECORD_VIDEO_Q
                            if not self.is_recording:
                                self.is_recording = True
                        break
            
            # wake up awaiting frame process
            if pause_flag:
                with QMutexLocker(self.switch_cam_lock):
                    self.pause_flag = False
                    self.Qconds.wait(self.switch_cam_lock)

            with self.cond:
                self.cond.notify_all()

            # waiting for image shape, acting as conditional variable
            try:
                shape = self.conn.recv()
            except EOFError:
                break

            # trigger frame window image updating
            self.send_signal.emit(shape)

            # trigger channel switching
            if switch_flag:
                self.switch_btn_recover_signal.emit()
            time4 = time.time()

            # trigger data table updating
            self.realtime_tab_singal.emit((self.id, time4-time0, self.frame_cache_len if drop_flag else 0, self.display_real_flag == 0))
            self.display_real_flag = (self.display_real_flag + 1) % 5

            # trigger capture/record button recovery and print prompt info
            if cam_f:
                self.camera_capture_recover_signal.emit()
            elif not rec_f and self.is_recording:
                self.is_recording = False
                self.camera_record_recover_signal.emit()



class QThread4stdout(QThread):
    """
    QThread for output redirection: from terminal to text widget
    """

    redirect_signal = pyqtSignal(str)

    def __init__(
            self, 
            parent: QObject = None,
            camera_fps: int = 25,
        ) -> None:
        
        super().__init__(parent)
        self.wait_time = 1/camera_fps
        self.min_wait_time = 1/camera_fps
        self.max_wait_time = 1

    def run(self):
        """
        Main loop for QThread4stdout
        keep rolling with auto-adjustable interval
        """
        while True:
            with gpc_stream.offset.get_lock():
                ofs = gpc_stream.offset.value
            
            if ofs == 0:
                time.sleep(self.wait_time)
                self.wait_time = min(self.max_wait_time, self.wait_time*2)
            else:
                self.wait_time = self.min_wait_time
                self.redirect_signal.emit("" if self.wait_time <= 2*self.min_wait_time else "\n")

