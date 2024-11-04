import sys, os, numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from Qt_ui.py.display import Ui_MainWindow as dis_win
from Qt_ui.threads import QThread4VideoDisplay
from Qt_ui.ctrl_panel.ctrl_panel import ctrl_panel

from typing import List

class custom_window(QtWidgets.QMainWindow):
    
    sum = 0
    def __init__(self, gpc: dict):
        QtWidgets.QMainWindow.__init__(self)
        self.num_cam = gpc["num_cam"]
        dis = dis_win()
        dis.setupUi(self)
        self.ctrl_panel = ctrl_panel(self, gpc["ctrl_pa_conn"][0])
        self.ctrl_panel.setupUi(self, (1200, 100))
        self.total_signal(gpc)
        self.run_flag = gpc["run_flag"]
        self.pool = gpc["pool"]
        # 退出按钮
        self.exit_btn: QtWidgets.QPushButton
        self.exit_btn.clicked.connect(self.close_fun)

    def total_signal(self, gpc: dict):
        # 实例化线程类
        self.myThread : List[QThread4VideoDisplay] = []
        for i in range(self.num_cam):
            self.myThread.append(
                QThread4VideoDisplay(
                    thread_id=i,
                    cam_proc=gpc["pool"][i],
                    conn=gpc["frame_pa_conn"][i],
                    cond=gpc["frame_flag"][i],
                )
            )
        # 单击按钮, 以单击为发送信号
        self.switch_btn.clicked.connect(self.on_clicked)
        
    def on_clicked(self):
        self.sum+=1
        if self.sum % 2 == 0:
            for thread in self.myThread:
                thread.send_signal.disconnect(self.switch_slot)
            self.switch_btn.setText("Start")
        else:
            for thread in self.myThread:
                thread.send_signal.connect(self.switch_slot)
                thread.start()
            self.switch_btn.setText("Stop")
            
    def switch_slot(self, pics: np.ndarray, id: int):
        image = QtGui.QImage(pics,pics.shape[1],pics.shape[0],QtGui.QImage.Format.Format_RGB888)
        s = self.centralWidget()
        # return

        for name, child in zip([ch.objectName() for ch in s.children()], s.children()):
            if name == f"videoWin_{id}" and isinstance(child, QtWidgets.QLabel):
                child.setPixmap(QtGui.QPixmap.fromImage(image))

    def close_fun(self):
        self.run_flag.acquire(True)
        self.run_flag.notify_all()
        self.run_flag.release()
        for child in self.pool:
            child.join()
        self.close()

    


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = custom_window()


    MainWindow.resize(800, 480)
    MainWindow.setWindowTitle("Monitor")

    MainWindow.show()
    sys.exit(app.exec_())