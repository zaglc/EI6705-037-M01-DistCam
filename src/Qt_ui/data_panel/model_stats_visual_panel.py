import sys
import time
from functools import partial
from threading import Thread
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QMutex, Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsScene,
    QGraphicsView,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

__tabs__ = ["Current Frame Obj", "Cumulative New Obj"]
# number of frames to show, from current frame
SHOWING_FRAME_NUMBER = 20
BORDER_NUMBER = 4
# interval between each shown frame
FRAME_INTERVAL = 5
XTICKS_NUMBER = 5
MAX_CLASSES = 5


class _Fixed_QGraphicsView(QGraphicsView):
    """
    for adaptive size
    """

    def __init__(self, parent=None):
        super().__init__(parent)

    def resizeEvent(self, event):
        # print("adjust", self.width(), self.height())
        return super().resizeEvent(event)


class Stats_Visualize(QToolBar):
    def __init__(
        self,
        parent: QWidget,
        camera_names: List[str],
        unittest=False,
        classes_of_interest: Dict[int, str] = {0: "person", 2: "car"},
    ):
        super().__init__(parent)
        self.classes_of_interest = {k: classes_of_interest[k] for k in list(classes_of_interest.keys())[:MAX_CLASSES]}
        self.names = camera_names
        self.num_cam = len(camera_names)
        self.active_cam = 0
        assert FRAME_INTERVAL >= 1

        # 创建布局
        outer = QTabWidget(self)
        self.outer_tab = outer
        self.addWidget(outer)
        self.setMinimumSize(600, 150)

        # data update flag
        self.update_lock = QMutex()
        self.update_count = [0 for _ in range(self.num_cam)]
        self.last_updated_frame = [0 for _ in range(self.num_cam)]

        # create multiple QGraphicsView
        self.graphics_view_lst: List[_Fixed_QGraphicsView] = []
        self.scene_lst: List[QGraphicsScene] = []
        self.figure_lst: List[Figure] = []
        self.canvas_lst: List[FigureCanvas] = []
        self.axs_lst: List[Dict[str, Axes]] = []

        # data part init
        self.data_lst: List[Dict[str, Dict[str, List[int]]]] = [None for _ in range(self.num_cam)]
        self.data_cache_lst: List[Dict[str, Dict[str, List[int]]]] = [None for _ in range(self.num_cam)]
        self.plt_objs_lst: List[Dict[str, Dict[str, plt.Line2D]]] = [None for _ in range(self.num_cam)]
        for i, n in enumerate(self.names):
            graphics_view = _Fixed_QGraphicsView(self.outer_tab)
            self.outer_tab.addTab(graphics_view, n)

            # set graphics scene
            scene = QGraphicsScene(self.outer_tab)
            graphics_view.setScene(scene)
            self.scene_lst.append(scene)
            self.graphics_view_lst.append(graphics_view)

            # matplotlib axs
            figure, axs = plt.subplots(1, 2, figsize=(9, 2.7))
            canvas = FigureCanvas(figure)
            scene.addWidget(canvas)
            self.figure_lst.append(figure)
            self.canvas_lst.append(canvas)
            self.axs_lst.append({k: v for k, v in zip(__tabs__, axs)})

            # init data
            self._init_data_and_draw_fig(i, figure)

        outer.currentChanged.connect(self.tab_changed_slot)
        outer.setCurrentIndex(0)

        # set timer for unit test
        if unittest:
            self.current_cache_frame = [0 for _ in range(self.num_cam)]
            self.temp_data_packets_lst = [
                {k: {c: 0 for c in self.classes_of_interest.values()} for k in __tabs__}
                for _ in range(len(camera_names))
            ]
            self.timer = QTimer(self)
            self.timer.timeout.connect(partial(self.update_data_slot))
            self.timer.start(100)
            self.last_update = time.time()

    def _init_data_and_draw_fig(self, cam_id: int, figure: Figure):

        # data for different axs: number of objs per class in class of interest
        data: Dict[str, Dict[str, List[int]]] = {
            k: {c: [0 for _ in range(SHOWING_FRAME_NUMBER + BORDER_NUMBER)] for c in self.classes_of_interest.values()}
            for k in __tabs__
        }
        # cache: per FRAME_INTERVAL an update
        data_cache: Dict[str, Dict[str, List[int]]] = {
            k: {c: [] for c in self.classes_of_interest.values()} for k in __tabs__
        }
        plt_objs: Dict[str, Dict[str, plt.Line2D]] = {k: {} for k in __tabs__}

        # initialize figure
        hd = []
        for idx, ((title, ax), sub_data) in enumerate(zip(self.axs_lst[cam_id].items(), data.values())):
            ax.cla()
            x_data = list(range(-SHOWING_FRAME_NUMBER - BORDER_NUMBER + 1, 1, 1))
            for label, y_data in sub_data.items():
                (plt_obj,) = ax.plot(x_data, y_data, label=label, linewidth=3)
                plt_objs[title].update({label: plt_obj})
                if idx == 0:
                    hd.append(plt_obj)
            ax.set_xlim(-SHOWING_FRAME_NUMBER, 1)
            x_tick_pos = list(range(0, -SHOWING_FRAME_NUMBER, -SHOWING_FRAME_NUMBER // XTICKS_NUMBER))
            x_tick_labels = [str(-x) if x != 0 else "Current" for x in x_tick_pos]
            ax.set_xticks(x_tick_pos, x_tick_labels)
            ax.set_xlabel("Previous Frames Count")
            ax.yaxis.tick_right()
            ax.set_ylabel("Number of Objects")
            ax.set_title(title)
            ax.axvline(x=0, color="r", linestyle="--")
            ax.grid("on")
        figure.subplots_adjust(wspace=0.3, bottom=0.17, left=0.13)
        figure.legends.clear()
        figure.legend(
            handles=hd,
            bbox_to_anchor=(0.11, 0.9),
            ncol=1,
            handlelength=1,
            handletextpad=0.3,
            labelspacing=0.1,
            columnspacing=0.7,
        )

        self.data_lst[cam_id] = data
        self.data_cache_lst[cam_id] = data_cache
        self.plt_objs_lst[cam_id] = plt_objs

    def update_data_slot(self, modify_idx=None, current_frame=None, count_packet=None):
        """
        when new detection data arrived, update the data and plot
        """

        assert (modify_idx is None and current_frame is None and count_packet is None) or (
            modify_idx is not None and current_frame is not None and count_packet is not None
        )

        for i, n in enumerate(self.names):
            # they must be modified in atomic manner
            if modify_idx is None:
                self.update_lock.lock()
                current_cache_frame = self.current_cache_frame[i]
                temp_data_packets = self.temp_data_packets_lst[i]
                self.update_lock.unlock()
            else:
                if i != modify_idx:
                    continue
                current_cache_frame = current_frame
                temp_data_packets = {k: v for k, v in zip(__tabs__, count_packet)}

            last_update_frame = self.last_updated_frame[i]
            if last_update_frame == current_cache_frame:
                continue
            else:
                self.last_updated_frame[i] = current_cache_frame
            data_cache = self.data_cache_lst[i]
            data = self.data_lst[i]
            axs = self.axs_lst[i]
            plt_objs = self.plt_objs_lst[i]
            canvas = self.canvas_lst[i]

            data_packets = temp_data_packets
            for title, new_data in data_packets.items():
                if new_data is None:
                    continue
                for label, count in new_data.items():
                    if label in data_cache[title]:
                        data_cache[title][label].append(count)
            self.update_count[i] += 1

            if self.update_count[i] % FRAME_INTERVAL == 0:
                self.update_count[i] = 0
                # update data storage
                for title, new_data in data_cache.items():
                    for label, count_lst in new_data.items():
                        d = (
                            0
                            if not len(count_lst)
                            else count_lst[-1]
                            if title == "Cumulative New Obj"
                            else sum(count_lst) / FRAME_INTERVAL
                        )
                        # print(f"update {title} {label} with {d}")
                        # print(self.data_lst)
                        self.data_lst[i][title][label].pop(0)
                        self.data_lst[i][title][label].append(d)
                        count_lst.clear()
                if self.active_cam != i:
                    continue

                # update plot
                for (title, ax), idata in zip(axs.items(), data.values()):
                    new_ylim = []
                    for label, y_data in idata.items():
                        plt_objs[title][label].set_ydata(y_data)
                        if new_ylim == []:
                            new_ylim = [np.floor(min(y_data)), np.ceil(max(y_data))]
                        else:
                            new_ylim[0] = min(new_ylim[0], np.floor(min(y_data)))
                            new_ylim[1] = max(new_ylim[1], np.ceil(max(y_data)))
                    delta = max(2, (new_ylim[1] - new_ylim[0]) * 0.1)
                    new_ylim = [new_ylim[0] - delta, new_ylim[1] + delta]
                    old_ylim = [int(a) for a in ax.get_ylim()]
                    if old_ylim != new_ylim:
                        ax.set_ylim(new_ylim)

                # refresh the canvas
                self.figure_lst[i]
                canvas.draw()
                break

    def tab_changed_slot(self):
        """
        slot invoked when tab changed
        """

        self.active_cam = self.outer_tab.currentIndex()

    def reset_figure(
        self, classes_of_interest: Dict[int, str] = None, camera_names: List[str] = None, cam_id: int = None
    ):
        """
        reset figure if video source config changes, incoked when:
        1. model type change
        2. camera source change
        """

        if classes_of_interest is not None:
            self.classes_of_interest = classes_of_interest
        if camera_names is not None:
            self.names = camera_names
        for i, n in enumerate(self.names):
            if cam_id is not None and cam_id != i:
                continue
            self.data_lst[i] = None
            self.data_cache_lst[i] = None
            self.plt_objs_lst[i] = None
            self.update_count[i] = 0
            if n != self.outer_tab.tabText(i):
                self.outer_tab.setTabText(i, n)
            figure = self.figure_lst[i]
            self._init_data_and_draw_fig(i, figure)

    def resizeEvent(self, event):

        # for i in range(len(self.names)):
        #     figure = self.figure_lst[i]
        #     graphics = self.graphics_view_lst[i]
        #     if i == 0:
        #         print("outer: ", self.width(), self.height(), self.outer_tab.width(), self.outer_tab.height(), [self.outer_tab.currentWidget().width()], graphics.width(), graphics.height(), self.canvas_lst[i].get_width_height())
        #         print("figure: ", figure.get_size_inches(), figure.get_dpi())
        #     figure.set_size_inches(1200 / figure.get_dpi(), (500 - 10) / figure.get_dpi())
        #     canvas = FigureCanvas(figure)
        #     scene = QGraphicsScene(self.outer_tab)
        #     scene.addWidget(canvas)
        #     graphics.setScene(scene)
        #     self.scene_lst[i] = scene
        #     self.canvas_lst[i] = canvas

        #     canvas.draw()

        return super().resizeEvent(event)


if __name__ == "__main__":

    classes_of_interest = {0: "person", 2: "car"}
    names = ["v1"]  # , "v2"]

    def unit_test_change_data():
        data_packets = {k: {c: 0 for c in classes_of_interest.values()} for k in __tabs__}
        while True:
            cost = 0
            window.update_lock.lock()
            for i in range(len(names)):
                cur = time.time()
                data_packets["Current Frame Obj"] = {c: np.random.randint(6, 9) for c in classes_of_interest.values()}
                data_packets["Cumulative New Obj"] = None
                window.temp_data_packets_lst[i] = data_packets
                window.current_cache_frame[i] += 1
                cost += time.time() - cur
            window.update_lock.unlock()
            time.sleep(max(0, 0.1 - cost))

    thread = Thread(target=unit_test_change_data, daemon=True)

    app = QApplication(sys.argv)
    window = Stats_Visualize(None, names, unittest=True, classes_of_interest=classes_of_interest)
    window.setWindowTitle("动态折线图")
    window.resize(1100, 400)
    thread.start()
    window.show()

    ret = app.exec()

    sys.exit(ret)
