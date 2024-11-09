import sys, os, numpy as np, sys
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import (
    QGridLayout, QMenu, QStatusBar, QLayout
)
from PyQt6.QtGui import (
    QAction
)
from Qt_ui.view_panel.display import Ui_MainWindow as dis_win
from Qt_ui.ctrl_panel.ctrl_panel import ctrl_panel
from Qt_ui.data_panel.data_table import Realtime_Datatab

from typing import List

class custom_window(QtWidgets.QMainWindow):
    
    def __init__(self, gpc: dict):
        QtWidgets.QMainWindow.__init__(self)
        self.resize(1500, 900)
        # self.setMaximumSize(1500, 600)
        self.setWindowTitle("Monitor")
        ctw = QtWidgets.QWidget(self)
        self.setCentralWidget(ctw)
        self.num_cam = gpc["num_cam"]

        # initializing view panel
        self.dis = dis_win(ctw, self.num_cam)
        self.dis.setupUi(self, gpc)
        for win in self.dis.videoWin:
            win.ctrl_select_btn_signal.connect(self.ctrl_select_btn_slot)
            win.frame_thread.camera_capture_recover_signal.connect(self.camera_cap2rec_btn_recover_slot)
            win.frame_thread.camera_record_recover_signal.connect(self.camera_cap2rec_btn_recover_slot)

        # initializing ctrl panel
        self.ctrl_panel = ctrl_panel(
            parent=self, 
            num_cam=gpc["num_cam"],
            conn=gpc["ctrl_pa_conn"],
            cond=gpc["ctrl_flag"],
            exec_seq=gpc["ctrl_val4exec_seq"],
        )
        self.ctrl_panel.setupUi(self)
        self.ctrl_panel.camera_btn_signal.connect(self.camera_capture_btn_slot)
        self.ctrl_panel.record_btn_signal.connect(self.camera_record_btn_slot)

        # initializing data panel
        self.data_panel = Realtime_Datatab(self, self.num_cam)
        self.setCentralWidget(ctw)
        for win in self.dis.videoWin:
            win.frame_thread.realtime_tab_singal.connect(self.update_realtime_tab_slot)

        self.run_flag = gpc["run_flag"]
        self.pool = gpc["pool"]
        self.conds = [*gpc["frame_flag"], gpc["ctrl_flag"]]
        # 退出按钮
        self.dis.exitbtn.clicked.connect(self.close)

        # 修改grid，以后还要改
        ctw = self.centralWidget()
        ctw.setMaximumSize(1400, 800)
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        ctw.setLayout(grid)

        grid.addWidget(self.dis, 0, 0, 7, 7)
        self.data_panel.grid.addWidget(self.dis.switch_btn, 0, 14)
        self.data_panel.grid.addWidget(self.dis.exitbtn, 1, 14)
        # grid.addWidget(self.data_panel, 8, 0, 1, 7)
        # grid.addWidget(self.ctrl_panel, 0, 8, 7, 2)
        self._init_meunbar()


    def _init_meunbar(self):
        ctrl_act = QAction("Ctrl Panel", self)
        ctrl_act.setStatusTip("open/close ctrl panel")
        ctrl_act.setCheckable(True)

        data_act = QAction("Data Panel", self)
        data_act.setStatusTip("open/close data panel")
        data_act.setCheckable(True)

        self.menubar = self.menuBar()
        self.menubar.addSeparator()
        windows_menu = self.menubar.addMenu("Windows(&F)")
        windows_menu.addAction(ctrl_act)
        windows_menu.addSeparator()
        windows_menu.addAction(data_act)

        self.setStatusBar(QStatusBar(self))

        self.addToolBar(QtCore.Qt.ToolBarArea.RightToolBarArea, self.ctrl_panel)
        self.addToolBar(QtCore.Qt.ToolBarArea.BottomToolBarArea, self.data_panel)


    def ctrl_select_btn_slot(self, info):
        id, flag = info
        if flag:
            self.ctrl_panel.set_selected_cam(id)
            for idx, win in enumerate(self.dis.videoWin):
                if idx != id:
                    win.control_btn.setEnabled(False)
                    win.control_btn.setStyleSheet("QPushButton{color:rgb(128,128,128)}")
        else:
            self.ctrl_panel.set_selected_cam(-1)
            for idx, win in enumerate(self.dis.videoWin):
                if idx != id:
                    win.control_btn.setEnabled(True)
                    win.control_btn.setStyleSheet("QPushButton{color:rgb(0,0,0)}")


    def update_realtime_tab_slot(self, info):
        id ,interval, drop, display_flag = info
        self.data_panel._compute_slide_exp_average((interval, drop), id)
        if display_flag:
            self.data_panel._updateTabItem(id)


    def camera_capture_btn_slot(self, info):
        folder = info
        if all([not win.is_selected_cap for win in self.dis.videoWin]): 
            return
        
        self.ctrl_panel.camera_btn.setEnabled(False)
        self.ctrl_panel.camera_btn.setStyleSheet("ctrl_btn{color:rgb(127,127,127)}")
        self.ctrl_panel.record_btn.setEnabled(False)
        self.ctrl_panel.record_btn.setStyleSheet("ctrl_btn{color:rgb(127,127,127)}")
        self.ctrl_panel.select_pth_btn.setEnabled(False)
        self.ctrl_panel.select_pth_btn.setStyleSheet("ctrl_btn{color:rgb(127,127,127)}")

        for idx, win in enumerate(self.dis.videoWin):
            win.capture_btn.setEnabled(False)
            style = win.capture_btn.styleSheet().replace("0","127")
            win.capture_btn.setStyleSheet(style)

        for idx, win in enumerate(self.dis.videoWin):
            if not win.is_selected_cap: continue
            win.frame_thread.switch_cam_lock.lock()
            win.frame_thread.camera_btn_path = folder
            win.frame_thread.switch_cam_lock.unlock()


    def camera_cap2rec_btn_recover_slot(self):
        for idx, win in enumerate(self.dis.videoWin):
            win.capture_btn.setEnabled(True)
            style = win.capture_btn.styleSheet().replace("127","0")
            win.capture_btn.setStyleSheet(style)

        self.ctrl_panel.camera_btn.setEnabled(True)
        self.ctrl_panel.camera_btn.setStyleSheet("ctrl_btn{color:rgb(0,0,0)}")
        self.ctrl_panel.record_btn.setEnabled(True)
        self.ctrl_panel.record_btn.setStyleSheet("ctrl_btn{color:rgb(0,0,0)}")
        self.ctrl_panel.select_pth_btn.setEnabled(True)
        self.ctrl_panel.select_pth_btn.setStyleSheet("ctrl_btn{color:rgb(0,0,0)}")

    def camera_record_btn_slot(self, info):
        folder, is_start = info
        if all([not win.is_selected_cap for win in self.dis.videoWin]): 
            return

        self.ctrl_panel.is_recording = True
        self.ctrl_panel.camera_btn.setEnabled(False)
        self.ctrl_panel.camera_btn.setStyleSheet("ctrl_btn{color:rgb(127,127,127)}")
        self.ctrl_panel.select_pth_btn.setEnabled(False)
        self.ctrl_panel.select_pth_btn.setStyleSheet("ctrl_btn{color:rgb(127,127,127)}")
        self.ctrl_panel.record_btn.setStyleSheet("ctrl_btn{color:rgb(255,0,0)}")
        
        for idx, win in enumerate(self.dis.videoWin):
            win.capture_btn.setEnabled(False)
            style = win.capture_btn.styleSheet().replace("0","127")
            win.capture_btn.setStyleSheet(style)

        for idx, win in enumerate(self.dis.videoWin):
            if not win.is_selected_cap: continue
            win.frame_thread.switch_cam_lock.lock()
            win.frame_thread.record_btn_path = folder if is_start else None
            win.frame_thread.switch_cam_lock.unlock()


    def closeEvent(self, event):
        # 先把发送图片的子进程关闭，管道发送端关闭，接收端报错，q线程也会退出

        with self.run_flag.get_lock():
            self.run_flag.value = 0
        for idx, win in enumerate(self.dis.videoWin):
            win.frame_thread.quit()
            print(f"frame Qthread {idx} quit")
        for idx, child in enumerate(self.pool):
            with self.conds[idx]:
                self.conds[idx].notify_all()
            child.join()
            print(f"subprocess {idx} quit, pid is {child.pid}")
        sys.stdout = sys.__stdout__
        super().closeEvent(event)