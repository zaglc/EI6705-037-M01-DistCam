from PyQt6 import QtCore
import time
import numpy as np
from multiprocessing.connection import Connection
from multiprocessing import Process
from multiprocessing.synchronize import Condition

class QThread4VideoDisplay(QtCore.QThread):

    send_signal = QtCore.pyqtSignal(tuple)
    switch_btn_recover_signal = QtCore.pyqtSignal()
    realtime_tab_singal = QtCore.pyqtSignal(tuple)
    camera_capture_recover_signal = QtCore.pyqtSignal()
    camera_record_recover_signal = QtCore.pyqtSignal()
    
    def __init__(
            self, 
            thread_id: int,
            conn: Connection,
            cond: Condition,
            exec_seq,
            save_pth,
            lock_switch,
            parent: QtCore.QObject = None,
        ) -> None:

        super().__init__(parent)
        self.id = thread_id
        self.conn = conn
        self.cond = cond
        self.exec_seq = exec_seq
        self.save_pth = save_pth
        self.switch_flag = False
        self.switch_cam_lock = lock_switch
        self.frame_cache_len = 3
        self.display_real_flag = 0
        self.camera_btn_path = None
        self.record_btn_path = None
        self.is_recording = False

        self.time = 0
        self.cnt = 0
    
    def run(self):
        # 发射后开始定时，并需求下一张，定时结束后再发射
        while True:
            # if self.id == 0: print("here1")
            time0 = time.time()
            self.switch_cam_lock.lock()
            # 拿信息后销毁
            flag = self.switch_flag
            self.switch_flag = False
            cam_path = self.camera_btn_path
            self.camera_btn_path = None
            rec_path = self.record_btn_path

            self.switch_cam_lock.unlock()
            # if self.id == 0: print("here2")
            while True:
                with self.exec_seq.get_lock():
                    if self.exec_seq.value == 1 or self.exec_seq.value == 3:
                        drop_flag = self.exec_seq.value == 3
                        
                        self.exec_seq.value = 2 if flag else 0
                        if cam_path is not None: 
                            self.exec_seq.value = 4
                            with self.save_pth.get_lock():
                                self.save_pth.value = cam_path.encode()
                        elif rec_path is not None: 
                            self.exec_seq.value = 5
                            if not self.is_recording:
                                self.is_recording = True
                                with self.save_pth.get_lock():
                                    self.save_pth.value = rec_path.encode()
                        break
            time1 = time.time()
            self.cond.acquire(True)
            self.cond.notify_all()
            self.cond.release()
            time2 = time.time()
            # if self.id == 0: print("here3")
            try:
                shape = self.conn.recv()
            except EOFError:
                break
            time3 = time.time()

            self.send_signal.emit(shape)

            if flag:
                self.switch_btn_recover_signal.emit()
            time4 = time.time()

            self.realtime_tab_singal.emit((self.id, time4-time0, self.frame_cache_len if drop_flag else 0, self.display_real_flag == 0))
            self.display_real_flag = (self.display_real_flag + 1) % 5

            if cam_path is not None:
                self.camera_capture_recover_signal.emit()
                print(f"finish capture of camera {self.id}")
            elif rec_path is None and self.is_recording:
                self.is_recording = False
                self.camera_record_recover_signal.emit()
                print(f"finish record of camera {self.id}")

            # print(time4-time3)

            
            # if self.id == 0:
            #     print(self.id, round(time1-time0, 6), round(time2-time1, 6), round(time3-time2, 6), round(time4-time3, 6))
            #     if self.cnt >= 5:
            #         self.time += time4-time1
            #     if self.cnt >= 20:
            #         print(self.time/(self.cnt-5))
            #     self.cnt += 1
