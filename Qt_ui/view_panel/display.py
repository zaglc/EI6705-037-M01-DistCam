import os
from functools import partial
from multiprocessing import Queue
from typing import Dict, List

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QGridLayout, QMainWindow, QStatusBar, QWidget

from Qt_ui.childwins.vid_src_config import vid_src_config_window
from Qt_ui.utils import FRAME_ZOOM_LEVEL, compute_best_size4view_panel
from Qt_ui.view_panel.frame_window import frame_win


class Ui_MainWindow(QWidget):
    """
    view panel that unify all camera frame window
    define frame_win and corresponding QThread
    """

    def __init__(
        self,
        parent: QMainWindow,
        num_cam: int,
        frame_queues: List[Queue],
        command_queues: List[Queue],
        names: List[str],
        video_source_choice: List[list],
        video_source_info: Dict[str, List[dict]],
    ) -> None:
        """
        define layout for six cameras
        current layout: 2 * 3

        Args:
            num_cam: number of cameras,
            video_source_choices:[
                [<VIDEO_TYPE>, index], e.g: ['ip-cam', 1]
            ]
            video_source_info: 'sources' in pool.json: {
                "ip-cam":[
                    {
                        "NICKNAME": xxx,
                    }
                ],
                "local_vid":[
                    {
                        "NICKNAME": xxx,
                    }
                ]
            }
        """

        super().__init__(parent=parent)
        self.num_cam = num_cam
        # current choice for video source, from `video_source_pool.json`
        self.video_source_choice = video_source_choice
        # overall video source pool
        self.video_source_info = video_source_info

        self.videoWin: List[frame_win] = []
        self.switch_btn_label = False
        self.single_view = False
        self.single_view_id = -1
        self.fmin_size = (320, 180)
        self.zoom_level = 0

        self.parent_ctw = parent.centralWidget()
        self.parent_scroll_area = None
        grid = QGridLayout()
        mid = (self.num_cam + 1) // 2
        row_expand = 1 if self.num_cam == 1 else 2
        self._mid, self._row_expand = mid, row_expand

        for i in range(self.num_cam):
            win = frame_win(i, self, frame_queues[i], command_queues[i], names[i], self.fmin_size)
            switch_func = partial(self.switch_video_source_slot, i)
            win.switch_cha.clicked.connect(switch_func)
            view_func = partial(self.single_view_btn_slot, i)
            win.single_view_btn.clicked.connect(view_func)
            self.videoWin.append(win)
            grid.addWidget(win, i >= mid, i % mid)

        for i in range(row_expand):
            grid.setRowStretch(i, 1)
        for i in range(mid):
            grid.setColumnStretch(i, 1)
        self.setLayout(grid)
        self.zoom_delta = (self.fmin_size[0] // 10 * mid, self.fmin_size[1] // 10 * row_expand)
        self.size_buffer = (self.width(), self.height())

    def single_view_btn_slot(self, id: int):
        """
        slot function for switching between single-view and multi-view
        """

        for i, win in enumerate(self.videoWin):
            if i == id:
                win.single_view_btn.setEnabled(False)
                continue
            win.single_view_btn.setEnabled(self.single_view)
            win.setVisible(self.single_view)

        self.videoWin[id].single_view_btn.setEnabled(True)
        grid_out: QGridLayout = self.layout()

        mid, row_expand = self._mid, self._row_expand
        for i in range(row_expand):
            if i != id // mid:
                grid_out.setRowStretch(i, int(self.single_view))
        for i in range(mid):
            if i != id % mid:
                grid_out.setColumnStretch(i, int(self.single_view))

        self.videoWin[id].is_selected_single = not self.single_view
        if self.single_view:
            print("Switched to multi-view")
            self.single_view_id = -1
            self.parent_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.parent_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.parent_scroll_area.setAlignment(Qt.AlignmentFlag.AlignLeft)
            self.resize(*self.size_buffer)
        else:
            print(f"Switched to single-view of {id}")
            self.single_view_id = id
            self.parent_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.parent_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.parent_scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)

            self.size_buffer = (self.width(), self.height())
            # compute best size for Ui_MainWindow
            target_size = compute_best_size4view_panel(
                self.videoWin[id], self.parent_ctw, self.parent_ctw.layout(), grid_out
            )
            self.resize(*target_size)

        self.single_view = not self.single_view

    def view_panel_zoom_in_slot(self, status: QStatusBar):
        """
        slot function for zoom in view panel
        """

        if self.single_view:
            status.showMessage(f"Single view mode, can't zoom in", msecs=500)
        elif self.zoom_level < FRAME_ZOOM_LEVEL - 1:
            w, h = self.width(), self.height()
            self.resize(w + self.zoom_delta[0], h + self.zoom_delta[1])
            self.zoom_level += 1
            status.showMessage(f"Current size level: {self.zoom_level}", msecs=500)
        else:
            status.showMessage(f"Maximal size level", msecs=500)

    def view_panel_zoom_out_slot(self, status: QStatusBar):
        """
        slot function for zoom out view panel
        """

        if self.single_view:
            status.showMessage(f"Single view mode, can't zoom out", msecs=500)
        elif self.zoom_level > 0:
            w, h = self.width(), self.height()
            self.resize(w - self.zoom_delta[0], h - self.zoom_delta[1])
            self.zoom_level -= 1
            status.showMessage(f"Current size level: {self.zoom_level}", msecs=500)
        else:
            status.showMessage(f"Minimal size level", msecs=500)

    def switch_video_source_slot(self, id: int):
        """
        slot function for switching video source
        outside each video frame: frame_win

        if video source change, the button will be re-activate by signal 'switch_btn_recover_signal'
        in frame_thread
        """

        self.videoWin[id].switch_cha.setEnabled(False)
        window = vid_src_config_window(
            self,
            self.video_source_info,
            self.video_source_choice[id][0],
            self.video_source_choice[id][1],
            os.path.join("data", "src"),
            [t for idx, t in enumerate(self.video_source_choice) if idx != id],
        )
        ret, source_type, index, source_info = window.exec()
        self.video_source_info = source_info
        if ret == QDialog.DialogCode.Accepted:
            nickname = source_info[source_type][index]["NICKNAME"]
            self.videoWin[id].groupbox.setTitle(nickname)
            self.video_source_choice[id] = [source_type, index]
            self.videoWin[id].switch_cam_slot(source_type, source_info[source_type][index])
            print(f"Video source of {id} changed to {source_type}: {nickname}")
        else:
            self.videoWin[id].switch_cha.setEnabled(True)
