from PyQt6 import QtCore
import time
import numpy as np
from multiprocessing.connection import Connection
from multiprocessing import Process
from multiprocessing.synchronize import Condition

class QThread4VideoDisplay(QtCore.QThread):

    send_signal = QtCore.pyqtSignal(np.ndarray, int)
    def __init__(
            self, 
            thread_id: int,
            cam_proc: Process,
            conn: Connection,
            cond: Condition,
            parent: QtCore.QObject = None,
        ) -> None:

        super().__init__(parent)
        self.id = thread_id
        self.conn = conn
        self.cond = cond
        
        self.time = 0
        self.cnt = 0
    
    def run(self):
        # 发射后开始定时，并需求下一张，定时结束后再发射
        while True:
            time1 = time.time()
            self.cond.acquire(True)
            self.cond.notify_all()
            self.cond.release()
            time2 = time.time()

            try:
                frame = self.conn.recv()
            except EOFError:
                break
            
            time3 = time.time()

            self.send_signal.emit(frame, self.id)
            time.sleep(1/30)
            time4 = time.time()
            # if self.id == 0:
            #     print(self.id, round(time2-time1, 6), round(time3-time2, 6), round(time4-time3, 6))
            #     if self.cnt >= 5:
            #         self.time += time4-time1-0.03333
            #     if self.cnt >= 20:
            #         print(self.time/(self.cnt-5))
            #     self.cnt += 1

