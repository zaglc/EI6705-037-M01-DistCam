import sys, time, os, json, datetime
from functools import partial
from PyQt6.QtWidgets import (
    QStatusBar, QScrollArea, QVBoxLayout, QMainWindow, QWidget, QSizePolicy
)
from typing import List
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt

from Qt_ui.view_panel.display import Ui_MainWindow as dis_win
from Qt_ui.ctrl_panel.ctrl_panel import ctrl_panel
from Qt_ui.data_panel.data_table import Realtime_Datatab
from Qt_ui.terminal_panel.output_log import terminal
from Qt_ui.childwins.save_prefer import childWindow


class custom_window(QMainWindow):
    """
    Main window, including:
        view panel
        data panel
        ctrl panel
    """

    def __init__(self, gpc: dict):
        """
        Args:
            gpc: global process context such as frame_buffer, condition vars
        """

        QMainWindow.__init__(self)
        self.resize(1500, 900)
        self.setWindowTitle("Monitor")
        ctw = QWidget(self)
        self.setCentralWidget(ctw)
        self.num_cam = gpc["num_cam"]
        self.switch_btn_label = False
        self.curtime = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        # initializing view panel
        self.dis = dis_win(
            parent=self, 
            num_cam=self.num_cam,
            frame_pa_conn=gpc["frame_pa_conn"],
            frame_flag=gpc["frame_flag"],
            frame_val4exec_seq=gpc["frame_val4exec_seq"],
            num_channel=gpc["num_channel"],
            frame_buffer=gpc["frame_buffer"],
        )
        for win in self.dis.videoWin:
            win.ctrl_select_btn_signal.connect(self.ctrl_select_btn_slot)
            win.frame_thread.camera_capture_recover_signal.connect(self.camera_cap2rec_btn_recover_slot)
            win.frame_thread.camera_record_recover_signal.connect(self.camera_cap2rec_btn_recover_slot)
        self.scroll_area = QScrollArea(ctw)
        self.scroll_area.setWidget(self.dis)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

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
        self.addToolBar(Qt.ToolBarArea.RightToolBarArea, self.ctrl_panel)

        # initializing data panel
        self.data_panel = Realtime_Datatab(self, self.num_cam)
        for win in self.dis.videoWin:
            win.frame_thread.realtime_tab_singal.connect(self.update_realtime_tab_slot)
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, self.data_panel)

        self.terminal_panel = terminal(self.centralWidget())
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, self.terminal_panel)

        # initializing global context
        # TODO: need optimize
        self.run_flag = gpc["run_flag"]
        self.pool = gpc["pool"]
        self.conds = [*gpc["inference_fg2"], *gpc["frame_flag"], gpc["ctrl_flag"]]
        self.other_conds = [*gpc["inference_fg"]]
        self.resolution: List[list] = gpc["resolution"]
        self.proc_num = [len(self.pool)-1-self.num_cam, len(self.pool)-1]

        # 修改grid，以后还要改
        ctw = self.centralWidget()
        grid = QVBoxLayout()
        grid.addWidget(self.scroll_area)
        ctw.setLayout(grid)

        # intitializing menubar
        self._init_meunbar()
        self.terminal_panel.redirect_stdout_slot("")


    def _init_meunbar(self):
        self.setStatusBar(QStatusBar(self))

        # "File" menu
        start_stop_act = QAction("Start", self)
        start_stop_act.setStatusTip("Start camera streaming...")
        start_stop_act.setCheckable(True)
        start_stop_act.triggered.connect(self.start_stop_slot)
        start_stop_act.setShortcut("ctrl+l")
        exit_act = QAction("Quit", self)
        exit_act.setStatusTip("Close the window...")
        exit_act.triggered.connect(self.close)
        exit_act.setShortcut("ctrl+q")
        
        self.menubar = self.menuBar()
        file_menu = self.menubar.addMenu("File(&F)")
        file_menu.addAction(start_stop_act)
        file_menu.addSeparator()
        file_menu.addAction(exit_act)
        self.start_stop_action = start_stop_act

        # "Windows" menu
        ctrl_act = QAction("Ctrl Panel", self)
        ctrl_act.setStatusTip("open/close ctrl panel")
        ctrl_act.setCheckable(True)
        data_act = QAction("Data Panel", self)
        data_act.setStatusTip("open/close data panel")
        data_act.setCheckable(True)

        self.menubar.addSeparator()
        windows_menu = self.menubar.addMenu("Windows(&W)")
        windows_menu.addAction(ctrl_act)
        windows_menu.addSeparator()
        windows_menu.addAction(data_act)

        # "View" menu
        zoom_in_act = QAction("Zoom in", self)
        zoom_in_act.setStatusTip("View panel zoom in")
        func1 = partial(self.dis.view_panel_zoom_in_slot, self.statusBar())
        zoom_in_act.triggered.connect(func1)
        zoom_in_act.setShortcut("ctrl+]")
        zoom_out_act = QAction("Zoom out", self)
        zoom_out_act.setStatusTip("View panel zoom out")
        func2 = partial(self.dis.view_panel_zoom_out_slot, self.statusBar())
        zoom_out_act.triggered.connect(func2)
        zoom_out_act.setShortcut("ctrl+[")

        self.menubar.addSeparator()
        windows_menu = self.menubar.addMenu("View(&V)")
        windows_menu.addAction(zoom_in_act)
        windows_menu.addSeparator()
        windows_menu.addAction(zoom_out_act)

        # "Setting" menu
        simu_stream_act = QAction("Simultaneous streaming", self)
        simu_stream_act.setStatusTip("Open/Close simu streaming of cams")
        simu_stream_act.setCheckable(True)
        simu_stream_act.triggered.connect(self.refresh_active_cam_slot)
        model_inference_act = QAction("Model inference", self)
        model_inference_act.setStatusTip("Open/Close model inference")
        model_inference_act.setCheckable(True)
        model_inference_act.setShortcut("ctrl+m")
        model_inference_act.triggered.connect(self.enable_model_infer_slot)
        save_preference_act = QAction("Preference setting...", self)
        save_preference_act.setStatusTip("Saving preference setting")
        save_preference_act.triggered.connect(self.save_prefer_set_slot)
        save_preference_act.setShortcut("ctrl+shift+p")
        self._config_bak()

        setting_menu = self.menubar.addMenu("Setting(&T)")
        setting_menu.addAction(simu_stream_act)
        setting_menu.addAction(model_inference_act)
        setting_menu.addSeparator()
        setting_menu.addAction(save_preference_act)


    def start_stop_slot(self):
        """
        slot function for start/stop displaying
        """

        if self.switch_btn_label:
            for win in self.dis.videoWin:
                win.response_main_click_slot()
            self.start_stop_action.setText("Start")
            self.switch_btn_label = False
        else:
            for win in self.dis.videoWin:
                win.response_main_click_slot()
            self.start_stop_action.setText("Stop")
            self.switch_btn_label = True


    def ctrl_select_btn_slot(self, info: tuple):
        """
        slot function for changing current active selected camera
        communicating between ctrl panel and view panel
        """

        id, flag = info
        if flag:
            self.ctrl_panel.set_selected_cam(id)
            for idx, win in enumerate(self.dis.videoWin):
                if idx != id and win.is_selected_ctrl:
                    win.control_btn.setStyleSheet("QPushButton{color:rgb(0,0,0)}")
                    win.frame.setStyleSheet(win.fm_qss["cap"] if win.is_selected_cap else win.fm_qss["common"])
                    win.is_selected_ctrl = False
        else:
            self.ctrl_panel.set_selected_cam(-1)


    def update_realtime_tab_slot(self, info: tuple):
        """
        slot function for updating data panel
        """

        id ,interval, drop, display_flag = info
        self.data_panel._compute_slide_exp_average((interval, drop), id)
        if display_flag:
            self.data_panel._updateTabItem(id)


    def camera_capture_btn_slot(self):
        """
        slot function for clicking capture button
        setting img saving path
        """

        if all([not win.is_selected_cap for win in self.dis.videoWin]): 
            return
        
        self.ctrl_panel.camera_btn.setEnabled(False)
        self.ctrl_panel.camera_btn.setStyleSheet("ctrl_btn{color:rgb(127,127,127)}")
        self.ctrl_panel.record_btn.setEnabled(False)
        self.ctrl_panel.record_btn.setStyleSheet("ctrl_btn{color:rgb(127,127,127)}")

        for _, win in enumerate(self.dis.videoWin):
            win.capture_btn.setEnabled(False)
            style = win.capture_btn.styleSheet().replace("0","127")
            win.capture_btn.setStyleSheet(style)

        for _, win in enumerate(self.dis.videoWin):
            if not win.is_selected_cap: continue
            win.frame_thread.switch_cam_lock.lock()
            win.frame_thread.camera_active = 1
            win.frame_thread.switch_cam_lock.unlock()


    def camera_cap2rec_btn_recover_slot(self):
        """
        slot function for recover capture and record button
        when workload finished
        """

        for _, win in enumerate(self.dis.videoWin):
            win.capture_btn.setEnabled(True)
            style = win.capture_btn.styleSheet().replace("127","0")
            win.capture_btn.setStyleSheet(style)

        self.ctrl_panel.camera_btn.setEnabled(True)
        self.ctrl_panel.camera_btn.setStyleSheet("ctrl_btn{color:rgb(0,0,0)}")
        self.ctrl_panel.record_btn.setEnabled(True)
        self.ctrl_panel.record_btn.setStyleSheet("ctrl_btn{color:rgb(0,0,0)}")

        if self.ctrl_panel.is_recording:
            self.ctrl_panel.is_recording = False


    def camera_record_btn_slot(self, info: int):
        """
        slot function for record button
        """

        is_start = info
        if all([not win.is_selected_cap for win in self.dis.videoWin]): 
            return

        self.ctrl_panel.is_recording = True
        self.ctrl_panel.camera_btn.setEnabled(False)
        self.ctrl_panel.camera_btn.setStyleSheet("ctrl_btn{color:rgb(127,127,127)}")
        self.ctrl_panel.record_btn.setStyleSheet("ctrl_btn{color:rgb(255,0,0)}")
        
        for _, win in enumerate(self.dis.videoWin):
            win.capture_btn.setEnabled(False)
            style = win.capture_btn.styleSheet().replace("0","127")
            win.capture_btn.setStyleSheet(style)

        for _, win in enumerate(self.dis.videoWin):
            if not win.is_selected_cap: continue
            win.frame_thread.switch_cam_lock.lock()
            win.frame_thread.record_active = 1 if is_start else 0
            win.frame_thread.switch_cam_lock.unlock()


    def refresh_active_cam_slot(self):
        """
        slot function for flipping flag 'simo_streaming'
        which indicating whether multiple channels in one camera
        can be simultaneously streamed or not
        """

        for win in self.dis.videoWin:
            win.switch_cam_lock.lock()
            win.frame_thread.need_refresh_cam_flag = True
            win.switch_cam_lock.unlock()


    def enable_model_infer_slot(self):
        """
        slot function for flipping model_flag
        which indicating whether model inference should be activated or not
        """

        for win in self.dis.videoWin:
            win.switch_cam_lock.lock()
            win.frame_thread.model_flag = True
            win.switch_cam_lock.unlock()


    def save_prefer_set_slot(self):
        img_lst = []
        for win in self.dis.videoWin:
            win._lock.lock()
            img_lst.append(win._preview_image)
            win._lock.unlock()

        self._config_bak(img_lst)


    def _config_bak(self, img_lst = [None]):
        if img_lst[0] is None and os.path.exists("data/temp/box_config.json"):
            with open("data/temp/box_config.json", 'r') as f:
                dicts = json.load(f)
                dicts["select_path"] = os.getcwd()+f"/data/{self.curtime}"
            with open("data/temp/box_config.json", 'w') as f:
                json.dump(dicts, f)
            return

        chan_ids = [bag.index(img_lst[i][1]) for i, bag in enumerate(self.resolution)]
        reso_info = [[*bag[idx], idx] for bag, idx in zip(self.resolution, chan_ids)]
        img_lst_temp = [img[0] for img in img_lst]
        dialog = childWindow(self, self.num_cam, reso_info, img_lst_temp, "data/temp/box_config.json", self.curtime)
        dialog.preview_move_slot(dialog.preview_id)
        ret = dialog.exec() if img_lst_temp[0] is not None else 0
        if ret or img_lst_temp[0] is None:
            dicts = dialog.gather_infos()
            if not os.path.exists("data/temp/"):
                os.makedirs("data/temp/")
            with open("data/temp/box_config.json", 'r') as f:
                old_dicts = json.load(f)
            with open("data/temp/box_config.json", 'w') as f:
                old_dicts.update(dicts)
                json.dump(old_dicts, f)
        dialog.destroy()


    def resizeEvent(self, event):
        """
        show size message
        """

        w, h = self.size().width(), self.size().height()
        cw, ch = self.centralWidget().size().width(), self.centralWidget().size().height()
        self.statusBar().showMessage(f"total:{w}, {h}; central:{cw}, {ch}", 30)
        return super().resizeEvent(event)


    def closeEvent(self, event):
        """
        overload closeEvent, new sequences are as follow:
            1. stop QThread of each frame
            2. stop subprocess: model*ddp(2), frame*cam_num(6), ctrl(1)
            3. stop main process
        """

        with self.run_flag.get_lock():
            self.run_flag.value = 0
        for idx, win in enumerate(self.dis.videoWin):
            win.frame_thread.quit()
            print(f"frame Qthread {idx} quit")

        for idx, child in enumerate(self.pool):
            if idx >= self.proc_num[0] and idx < self.proc_num[1]:
                m_id = (idx-self.proc_num[0])//(self.num_cam//self.proc_num[0])
                with self.other_conds[m_id]:
                    self.other_conds[m_id].notify_all()
            with self.conds[idx]:
                self.conds[idx].notify_all()
            child.join()
            print(f"subprocess {idx} quit, pid is {child.pid}")
        
        self.terminal_panel.std_thread.quit()
        sys.stdout = sys.__stdout__
        print("Std Qthread quit")
        super().closeEvent(event)