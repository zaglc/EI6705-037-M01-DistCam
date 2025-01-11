import datetime
import json
import os
import sys
from functools import partial
from typing import List

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QDialog,
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QToolBar,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from Qt_ui.childwins.model_selection import ModelSelectionWindow
from Qt_ui.childwins.save_prefer import childWindow
from Qt_ui.ctrl_panel.ctrl_panel import ctrl_panel
from Qt_ui.data_panel.data_table import Realtime_Datatab
from Qt_ui.data_panel.model_stats_visual_panel import Stats_Visualize
from Qt_ui.terminal_panel.output_log import terminal
from Qt_ui.utils import (
    BOX_JSON_PATH,
    MODEL_JSON_PATH,
    RS_RUNNING,
    RS_STOP,
    RS_WAITING,
    VIDEO_SOURCE_POOL_PATH,
    compute_best_size4view_panel,
)
from Qt_ui.view_panel.display import Ui_MainWindow as dis_win
from running_models.extern_model import YOLOV3_DETECT

WAIT_CHILD_PROC_TIME = 5


class custom_window(QMainWindow):
    """
    Main window, including:
        view panel
        data panel
        ctrl panel
    """

    def __init__(self, gpc: dict, curtime: datetime.datetime, selected_classes = {0: "person", 2: "car"}):
        """
        Args:
            gpc: global process context such as frame_buffer, condition vars
        """

        QMainWindow.__init__(self)
        self.resize(1500, 900)
        self.setWindowTitle("Monitor")
        ctw = QWidget(self)
        ctw.setMinimumSize(900, 540)
        self.setCentralWidget(ctw)
        self.num_cam = gpc["num_cam"]

        # temp var for model selection
        self.model_status = RS_WAITING
        self.model_type = YOLOV3_DETECT
        self.selected_classes = selected_classes

        # only for start and stop model process
        self.data_queues = gpc["data_queues"]
        # final image to be shown in view panel
        self.frame_write_queues = gpc["frame_write_queues"]
        # video command and ctrl command, maintained by custom_window
        self.command_queues = gpc["command_queues"]

        self.model_config = gpc["model_config"]

        self.pool = gpc["pool"]
        self.streaming = False
        self.curtime = curtime.strftime("%Y-%m-%d_%H-%M-%S")

        # initializing other content
        self.resolution: List[list] = []
        self.names: List[str] = []
        for queue in self.frame_write_queues:
            reso, n = queue.get()
            self.resolution.append(reso)
            self.names.append(n)

        # initializing view panel
        self.dis = dis_win(
            parent=self,
            num_cam=self.num_cam,
            frame_queues=self.frame_write_queues,
            command_queues=self.command_queues,
            names=self.names,
            video_source_choice=gpc["current_chosen_video_source"],
            video_source_info=gpc["video_source_info_lst"],
        )
        for idx, win in enumerate(self.dis.videoWin):
            win.ctrl_select_btn_signal.connect(self.ctrl_select_btn_slot)
            win.frame_thread.camera_capture_recover_signal.connect(self.camera_cap2rec_btn_recover_slot)
            win.switch_cha.clicked.connect(partial(self.switch_video_source_outer_slot, idx))
            win.frame_thread.switch_btn_recover_signal.connect(partial(self.update_resolution_after_switch_slot, id=idx))

        # ctw has only children: scroll area; scroll area has no margin with dis
        self.scroll_area = QScrollArea(ctw)
        self.scroll_area.setWidget(self.dis)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.dis.parent_scroll_area = self.scroll_area

        # initializing ctrl panel
        self.ctrl_panel = ctrl_panel(
            parent=self,
            num_cam=self.num_cam,
        )
        self.ctrl_panel.setupUi(self)
        self.ctrl_panel.camera_btn_signal.connect(self.camera_capture_btn_slot)
        self.ctrl_panel.record_btn_signal.connect(self.camera_record_btn_slot)
        self.ctrl_panel.ptz_btn_signal.connect(self.camera_ctrl_signal_slot)
        self.addToolBar(Qt.ToolBarArea.RightToolBarArea, self.ctrl_panel)

        # initializing data panel
        self.data_panel = Realtime_Datatab(self, self.num_cam)
        for win in self.dis.videoWin:
            win.frame_thread.realtime_tab_singal.connect(self.update_realtime_tab_slot)
        self.addToolBar(Qt.ToolBarArea.RightToolBarArea, self.data_panel)

        # initializing terminal panel
        self.terminal_panel = terminal(self, curtime=curtime, need_log=gpc["log"])
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, self.terminal_panel)

        # initializing model stats figure panel
        self.stats_panel = Stats_Visualize(self, self.names, classes_of_interest=self.selected_classes)
        for win in self.dis.videoWin:
            win.frame_thread.model_stats_update_signal.connect(self.stats_panel.update_data_slot)
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, self.stats_panel)

        # 修改grid，以后还要改
        ctw = self.centralWidget()
        grid = QVBoxLayout()
        grid.addWidget(self.scroll_area)
        ctw.setLayout(grid)

        # intitializing menubar
        self._init_meunbar()
        self.terminal_panel.redirect_stdout_slot("")
        self.start_stop_slot()

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
        data_act = QAction("Data Panel", self)
        data_act.setStatusTip("open/close data panel")
        terminal_act = QAction("Terminal Panel", self)
        terminal_act.setStatusTip("open/close terminal panel")
        model_act = QAction("Model Figure Panel", self)
        model_act.setStatusTip("open/close model figure panel")
        self.windows_act_lst = [ctrl_act, data_act, terminal_act, model_act]
        self.toolbar_name_lst = ["ctrl_panel", "data_panel", "terminal_panel", "stats_panel"]
        for idx, (n, act) in enumerate(zip(self.toolbar_name_lst, self.windows_act_lst)):
            act.setCheckable(True)
            act.setChecked(True)
            act.triggered.connect(partial(self.set_toolbar_visible_slot, idx, n))

        self.menubar.addSeparator()
        windows_menu = self.menubar.addMenu("Windows(&W)")
        windows_menu.addAction(ctrl_act)
        windows_menu.addSeparator()
        windows_menu.addAction(data_act)
        windows_menu.addSeparator()
        windows_menu.addAction(terminal_act)
        windows_menu.addSeparator()
        windows_menu.addAction(model_act)

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
        # simu_stream_act = QAction("Simultaneous streaming", self)
        # simu_stream_act.setStatusTip("Open/Close simu streaming of cams")
        # simu_stream_act.setCheckable(True)
        # simu_stream_act.triggered.connect(self.refresh_active_cam_slot)
        model_inference_act = QAction("Model inference", self)
        model_inference_act.setStatusTip("Open/Close model inference")
        model_inference_act.setShortcut("ctrl+m")
        model_inference_act.triggered.connect(self.model_inference_set_slot)
        save_preference_act = QAction("Preference setting...", self)
        save_preference_act.setStatusTip("Saving preference setting")
        save_preference_act.triggered.connect(self.save_prefer_set_slot)
        save_preference_act.setShortcut("ctrl+shift+p")
        self._config_bak()

        setting_menu = self.menubar.addMenu("Setting(&T)")
        # setting_menu.addAction(simu_stream_act)
        setting_menu.addAction(model_inference_act)
        setting_menu.addSeparator()
        setting_menu.addAction(save_preference_act)

    def start_stop_slot(self):
        """
        slot function for start/stop displaying
        """

        for win in self.dis.videoWin:
            win.frame_thread.switch_cam_lock.lock()
            win.frame_thread.pause_flag = True
            win.frame_thread.switch_cam_lock.unlock()
        self.start_stop_action.setText("Start" if self.streaming else "Stop")
        self.streaming = not self.streaming

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

        id, interval, drop, infer_cost = info
        self.data_panel._compute_slide_exp_average((interval, drop, infer_cost), id)
        self.data_panel._updateTabItem(id)

    def camera_ctrl_signal_slot(self, selected_cam: int, cmd: int, on_off: int):
        """
        slot function for camera control signal
        """

        if selected_cam != -1:
            win = self.dis.videoWin[selected_cam]
            win.frame_thread.switch_cam_lock.lock()
            win.frame_thread.ctrl_info = (cmd, on_off)
            win.frame_thread.switch_cam_lock.unlock()

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
            win.switch_cha.setEnabled(False)
            style = win.capture_btn.styleSheet().replace("0", "127")
            win.capture_btn.setStyleSheet(style)
            style = win.switch_cha.styleSheet()
            win.switch_cha.setStyleSheet(style)

        for _, win in enumerate(self.dis.videoWin):
            if not win.is_selected_cap:
                continue
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
            win.switch_cha.setEnabled(True)
            style = win.capture_btn.styleSheet().replace("127", "0")
            win.capture_btn.setStyleSheet(style)
            style = win.switch_cha.styleSheet()
            win.switch_cha.setStyleSheet(style)

        self.ctrl_panel.camera_btn.setEnabled(True)
        self.ctrl_panel.camera_btn.setStyleSheet("ctrl_btn{color:rgb(0,0,0)}")
        self.ctrl_panel.record_btn.setEnabled(True)
        self.ctrl_panel.record_btn.setStyleSheet("ctrl_btn{color:rgb(0,0,0)}")

        if self.ctrl_panel.is_recording:
            self.ctrl_panel.is_recording = False

    def camera_record_btn_slot(self):
        """
        slot function for record button
        """

        if all([not win.is_selected_cap for win in self.dis.videoWin]):
            return

        if not self.ctrl_panel.is_recording:
            self.ctrl_panel.is_recording = True
            self.ctrl_panel.camera_btn.setEnabled(False)
            self.ctrl_panel.camera_btn.setStyleSheet("ctrl_btn{color:rgb(127,127,127)}")
            self.ctrl_panel.record_btn.setStyleSheet("ctrl_btn{color:rgb(255,0,0)}")
            for _, win in enumerate(self.dis.videoWin):
                win.capture_btn.setEnabled(False)
                win.switch_cha.setEnabled(False)
                style = win.capture_btn.styleSheet().replace("0", "127")
                win.capture_btn.setStyleSheet(style)
                style = win.switch_cha.styleSheet()
                win.switch_cha.setStyleSheet(style)

        else:
            self.camera_cap2rec_btn_recover_slot()

        for _, win in enumerate(self.dis.videoWin):
            if not win.is_selected_cap:
                continue
            win.frame_thread.switch_cam_lock.lock()
            win.frame_thread.record_active = 1
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

    def model_inference_set_slot(self):
        """
        slot function for switch model and set hyper params
        """

        dialog = ModelSelectionWindow(
            self,
            self.num_cam,
            self.names,
            self.model_config,
            self.model_type,
            self.model_status == RS_RUNNING,
            self.selected_classes,
        )

        ret = dialog.exec()
        if ret[0] == QDialog.DialogCode.Accepted:
            current_model, is_active, model_config, selected_class, polygons = ret[1:]
            self.model_config.update(model_config)
            img_size = model_config[current_model]["img_size"]
            conf_thre = model_config[current_model]["conf_thre"]
            iou_thre = model_config[current_model]["iou_thre"]
            print(f"Select model: {current_model}")

            # if len(polygons) > 0:
            #     polygons = [[(round(r[0] * pp[0]), round(r[1] * pp[1])) for pp in p] for r, p in zip(self.resolution, polygons)]

            self.model_status = RS_RUNNING if is_active else RS_WAITING
            if self.model_type != current_model or selected_class != self.selected_classes:
                self.stats_panel.reset_figure(classes_of_interest=selected_class)
            self.model_type = current_model
            self.selected_classes = selected_class

            for queue in self.data_queues:
                queue.put((0, self.model_status, None, (current_model, polygons, img_size, list(selected_class.keys()), conf_thre, iou_thre)))
            for win in self.dis.videoWin:
                win.switch_cam_lock.lock()
                win.frame_thread.model_flag = True
                win.frame_thread.model_tuple = (current_model, is_active, polygons, img_size, list(selected_class.keys()), conf_thre, iou_thre)
                win.switch_cam_lock.unlock()

        dialog.destroy()

    def save_prefer_set_slot(self):
        """
        slot function for clicking save preference button
        """

        img_lst = []
        for win in self.dis.videoWin:
            win._lock.lock()
            img_lst.append(win._preview_image)
            win._lock.unlock()

        self._config_bak(img_lst)

    def _config_bak(self, img_lst=[None], save_temp_file=True):

        load_pth, save_pth = BOX_JSON_PATH, BOX_JSON_PATH
        if save_temp_file and not BOX_JSON_PATH.endswith("_temp.json"):
            save_pth = save_pth.replace(".json", "_temp.json")
            if os.path.exists(save_pth):
                load_pth = save_pth

        if img_lst[0] is None and os.path.exists(load_pth):
            with open(load_pth, "r") as f:
                dicts = json.load(f)
                dicts["select_path"] = os.path.join("data", self.curtime)
            with open(save_pth, "w") as f:
                json.dump(dicts, f, indent=4)
            return

        reso_info = [self.resolution, self.names]
        img_lst_temp = [img[0] for img in img_lst]
        dialog = childWindow(self, self.num_cam, reso_info, img_lst_temp, BOX_JSON_PATH, self.curtime)
        dialog.preview_move_slot(dialog.preview_id)

        # TODO: MAY NOT NEED EXECUTE
        ret = dialog.exec() if img_lst_temp[0] is not None else 0
        if ret == QDialog.DialogCode.Accepted or img_lst_temp[0] is None:
            dicts = dialog.gather_infos()
            if not os.path.exists(os.path.dirname(load_pth)):
                os.makedirs(os.path.dirname(load_pth))
            with open(load_pth, "r") as f:
                old_dicts = json.load(f)
            with open(save_pth, "w") as f:
                old_dicts.update(dicts)
                json.dump(old_dicts, f, indent=4)

        dialog.destroy()

    def switch_video_source_outer_slot(self, cam_id: int):
        """
        slot function for switching camera
        """

        if self.dis.src_changed:
            self.dis.src_changed = False
            self.names[cam_id] = self.dis.names[cam_id]
            self.stats_panel.reset_figure(camera_names=self.names, cam_id=cam_id)

    def set_toolbar_visible_slot(self, idx: int, obj_name: str):
        """
        slot function for setting toolbar visible
        """

        visible = self.windows_act_lst[idx].isChecked()
        obj: QToolBar = getattr(self, obj_name)
        obj.setVisible(visible)
        print(f"set {obj_name} visible: {visible}")

    def update_resolution_after_switch_slot(self, resolution: tuple, id: int):
        """
        slot function for updating resolution after switching camera
        """

        self.resolution[id] = resolution
        # print(resolution)


    def resizeEvent(self, event):
        """
        show size message
        """

        w, h = self.size().width(), self.size().height()
        cw, ch = self.centralWidget().size().width(), self.centralWidget().size().height()
        if self.dis.single_view_id != -1:
            dis_target_size = compute_best_size4view_panel(
                self.dis.videoWin[self.dis.single_view_id],
                self.dis.parent_ctw,
                self.dis.parent_ctw.layout(),
                self.dis.layout(),
            )
            self.dis.resize(*dis_target_size)
        self.statusBar().showMessage(f"total:{w}, {h}; central:{cw}, {ch}", 30)

        return super().resizeEvent(event)

    def closeEvent(self, event):
        """
        overload closeEvent, new sequences are as follow:
            0. save model_cfg and video_src_cfg
            1. stop QThread of each frame
            2. stop subprocess: model*ddp(2), frame*cam_num(6), ctrl(1)
            3. stop main process
        """

        # save configs
        model_cfg_save_pth = MODEL_JSON_PATH
        if not MODEL_JSON_PATH.endswith("_temp.json"):
            model_cfg_save_pth = model_cfg_save_pth.replace(".json", "_temp.json")
        with open(model_cfg_save_pth, "w") as f:
            json.dump(self.model_config, f, indent=4)

        video_src_pool_pth = VIDEO_SOURCE_POOL_PATH
        if not VIDEO_SOURCE_POOL_PATH.endswith("_temp.json"):
            video_src_pool_pth = video_src_pool_pth.replace(".json", "_temp.json")
        infos = [self.dis.video_source_choice, self.dis.video_source_info]
        infos[0] = [[ii[0], infos[1][ii[0]][ii[1]]["NICKNAME"]] if ii[0] == "local-vid" else ["local-vid", infos[1]["local-vid"]["NICKNAME"]] for ii in infos[0]]
        with open(video_src_pool_pth, "w") as f:
            json.dump({"choices": infos[0], "sources": infos[1]}, f, indent=4)

        # pass quit flag to Qthreads
        for idx, win in enumerate(self.dis.videoWin):
            win.frame_thread.is_running = False

        # pass quit flag to sub-processes
        for queue in self.data_queues:
            queue.put((0, RS_STOP, None, (None, None, None, None, None, None)))
        for queue in self.command_queues:
            queue.put((RS_STOP, None, None))

        for idx, child in enumerate(self.pool):
            child.join(WAIT_CHILD_PROC_TIME)
            if child.is_alive():
                print(f"{child.name} is still alive after {WAIT_CHILD_PROC_TIME}s, kill it")
                child.terminate()

        self.terminal_panel.std_thread.is_running = False
        sys.stdout = sys.__stdout__
        super().closeEvent(event)
