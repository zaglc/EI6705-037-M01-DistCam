import functools
from PyQt6.QtWidgets import (
    QGridLayout, QStatusBar, QWidget, QMainWindow
)
from typing import List

from Qt_ui.view_panel.frame_window import frame_win


class Ui_MainWindow(QWidget):
    """
    view panel that unify all camera frame window
    define frame_win and corresponding QThread
    """

    def __init__(
            self, 
            parent: QMainWindow, 
            num_cam, 
            frame_pa_conn,
            frame_flag,
            frame_val4exec_seq,
            num_channel,
            frame_buffer,
        ) -> None:
        """
        define layout for six cameras
        current layout: 2 * 3

        Args:
            num_cam: number of cameras, 
            frame_pa_conn: one end of pipes between main window and QThread,
            frame_flag: conditional var for sync between main and frame_main,
            frame_val4exec_seq: command signal,
            num_channel: number of channel of current camera,
            frame_buffer: output_buffer storing results with bbx,
        """
        
        super().__init__(parent=parent)
        self.num_cam = num_cam
        self.videoWin : List[frame_win] = []
        self.switch_btn_label = False
        self.single_view = False
        self.fmin_size = (320, 180)
        self.zoom_level = 0
        
        ctw = parent.centralWidget()
        grid = QGridLayout()
        mid = (self.num_cam + 1) // 2

        for i in range(self.num_cam):
            thread_params = [i, 
                frame_pa_conn[i], 
                frame_flag[i], 
                frame_val4exec_seq[i], 
            ]
            
            win = frame_win(i, 
                ctw, 
                num_channel[i], 
                frame_buffer[i], 
                thread_params, 
                self.fmin_size
            )
            func = functools.partial(self.single_view_btn_slot, i)
            win.single_view_btn.clicked.connect(func)
            self.videoWin.append(win)
            grid.addWidget(win, i >= mid, i % mid)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        for i in range(mid): grid.setColumnStretch(i, 1)
        self.setLayout(grid)
        self.zoom_delta = (self.fmin_size[0] // 10 * mid, self.fmin_size[1] // 10 * 2)


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

            # TODO: this btn doesn't need, but may need later
            # if self.single_view == False:
            #     win.frame_thread.switch_cam_lock.lock()
            #     win.frame_thread.pause_flag = True
            #     win.frame_thread.switch_cam_lock.unlock()
            #     while win.frame_thread.pause_flag == True: pass
            # else:
            #     win.frame_thread.switch_cam_lock.lock()
            #     win.frame_thread.Qconds.wakeOne()
            #     win.frame_thread.switch_cam_lock.unlock()
        
        self.videoWin[id].single_view_btn.setEnabled(True)
        grid_out: QGridLayout = self.layout()
        mid = (self.num_cam + 1) // 2
        grid_out.setRowStretch(1-(id//mid), int(self.single_view))
        for i in range(mid):
            if i != id%mid:
                grid_out.setColumnStretch(i, int(self.single_view))

        if self.single_view:
            print("Switched to multi-view")
        else:
            print(f"Switched to single-view of {id}")
        self.single_view = not self.single_view

        
    def view_panel_zoom_in_slot(self, status: QStatusBar):
        """
        slot function for zoom in view panel
        """

        if self.zoom_level < 9:
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
        
        if self.zoom_level > 0:
            w, h = self.width(), self.height()
            self.resize(w - self.zoom_delta[0], h - self.zoom_delta[1])
            self.zoom_level -= 1
            status.showMessage(f"Current size level: {self.zoom_level}", msecs=500)
        else:
            status.showMessage(f"Minimal size level", msecs=500)