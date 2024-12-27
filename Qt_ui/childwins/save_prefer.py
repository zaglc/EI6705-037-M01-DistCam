import json
import os
import sys
from functools import partial
from typing import List, Callable

from PyQt6.QtCore import QPoint, QRect
from PyQt6.QtCore import QRegularExpression as QRE
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QImage, QMoveEvent, QPixmap
from PyQt6.QtGui import QRegularExpressionValidator as QRegExpV
from PyQt6.QtGui import QResizeEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QWidget,
)


class preview_win(QLabel):
    """
    QClass for crop box preview
    """

    def __init__(self, parent: QWidget):
        """
        initialize two box
        """

        super().__init__(parent=parent)
        self.setFixedSize(400, 225)
        self.setScaledContents(True)
        self.setStyleSheet("QLabel{border-style: solid;border-width: 2px;border-color: rgba(0, 0, 0, 150)}")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.cropbox = QLabel(parent)
        self.cbox_size = [0, 0]
        self.cbox_delta = [0, 0]
        self.full_size = [self.width(), self.height()]
        self.cropbox.setStyleSheet("QLabel{border-style: solid;border-width: 1px;border-color: rgba(240, 45, 32, 150)}")

    def _update_cbox(self, new_val: dict):
        """
        function updating geometry of crop box
        """

        fbox = self.geometry()
        self.full_size = [new_val["res_w"], new_val["res_h"]]
        r = [self.width() / new_val["res_w"], self.height() / new_val["res_h"]]
        self.cbox_delta = [new_val["cp_x"] * r[0], new_val["cp_y"] * r[1]]
        st_point = QPoint(int(fbox.left() + self.cbox_delta[0]), int(fbox.top() + self.cbox_delta[1]))
        self.cbox_size = [new_val["cp_w"], new_val["cp_h"]]
        ed_point = QPoint(int(st_point.x() + new_val["cp_w"] * r[0]), int(st_point.y() + new_val["cp_h"] * r[1]))
        self.cropbox.setGeometry(QRect(st_point, ed_point))

    def moveEvent(self, event: QMoveEvent) -> None:
        """
        Event for red crop box align with preview box
        """

        fbox = self.geometry()
        ed_point = QPoint(int(fbox.left() + self.cbox_delta[0]), int(fbox.top() + self.cbox_delta[1]))
        self.cropbox.move(ed_point)
        return super().moveEvent(event)

    def resizeEvent(self, a0: QResizeEvent) -> None:
        """
        Event for red crop box align with preview box
        """

        fbox = self.geometry()
        ed_point = QPoint(int(fbox.left() + self.cbox_delta[0]), int(fbox.top() + self.cbox_delta[1]))
        n_r = [self.width() / self.full_size[0], self.height() / self.full_size[1]]
        w, h = n_r[0] * self.cbox_size[0], n_r[1] * self.cbox_size[1]
        self.cropbox.setGeometry(QRect(ed_point.x(), ed_point.y(), int(w), int(h)))
        return super().resizeEvent(a0)


class ResCropWidget(QFrame):
    """
    child widget of childWindow
    micro panel controlling crop box size and resolution
    """

    def __init__(self, parent: QWidget, idx: int, reso: list, name: str, func: Callable = None) -> None:
        super().__init__(parent=parent)
        self.idx = idx
        self.step = 8
        self.reso = reso
        self.name = name

        self.lab = QPushButton(self)
        self.lab.setText(name)
        if func is not None:
            self.lab.clicked.connect(partial(func, idx))
        
        self.res_w = QLineEdit(self)
        self.res_w.setText(f"{reso[0]}")
        self.res_w.setValidator(QRegExpV(QRE(r"[0-9]{4}")))
        self.res_h = QLineEdit(self)
        self.res_h.setText(f"{reso[1]}")
        self.res_h.setValidator(QRegExpV(QRE(r"[0-9]{4}")))
        self.class_id = QLineEdit(self)
        self.class_id.setPlaceholderText("eg: C(car)")
        self.class_id.setValidator(QRegExpV(QRE(r"[a-zA-Z0-9]{16}")))
        
        flayout1 = QFormLayout()
        flayout1.addRow("view:", self.lab)
        flayout1.addRow("class-id:", self.class_id)
        flayout1.addRow("width:", self.res_w)
        flayout1.addRow("height:", self.res_h)
        outer1 = QWidget(self)
        outer1.setLayout(flayout1)

        self.cp_x = QSpinBox(self)
        self.cp_x.setMaximum(reso[0])
        self.cp_y = QSpinBox(self)
        self.cp_y.setMaximum(reso[1])
        self.cp_w = QSpinBox(self)
        self.cp_w.setMaximum(reso[0])
        self.cp_h = QSpinBox(self)
        self.cp_h.setMaximum(reso[1])
        self.cp_w.setValue(512)
        self.cp_x.setValue(0)
        self.cp_h.setValue(512)
        self.cp_y.setValue(0)

        flayout2 = QFormLayout()
        flayout2.addRow("box-x:", self.cp_x)
        flayout2.addRow("box-y:", self.cp_y)
        flayout2.addRow("box-w:", self.cp_w)
        flayout2.addRow("box-h:", self.cp_h)
        outer2 = QWidget(self)
        outer2.setLayout(flayout2)

        grid = QGridLayout()
        grid.setSpacing(5)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.addWidget(outer1, 0, 0)
        grid.addWidget(outer2, 0, 1)
        self.setLayout(grid)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def _init_boxSignal(self, func):
        """
        funtion for associating slot
        """

        def range_check(type, func2):
            def deco():
                flag = 0
                if type == "cp_w":
                    x, w = self.cp_x.value(), self.cp_w.value()
                    if x + w > self.reso[0]:
                        self.cp_x.setValue(self.reso[0] - w)
                        flag = 1
                    self.cp_x.setMaximum(self.reso[0] - w)
                elif type == "cp_h":
                    y, h = self.cp_y.value(), self.cp_h.value()
                    if y + h > self.reso[1]:
                        self.cp_y.setValue(self.reso[1] - h)
                        flag = 1
                    self.cp_y.setMaximum(self.reso[1] - h)

                if not flag:
                    func2()

            return deco

        attrs = ["cp_x", "cp_y", "cp_w", "cp_h"]
        func2 = partial(func, self.idx)
        for att in attrs:
            getattr(self, att).valueChanged.connect(range_check(att, func2))
            getattr(self, att).setSingleStep(self.step)

    def gather_infos(self):
        """
        get information of each child widget
        """

        return dict(
            {
                "class_id": self.class_id.text(),
                "res_w": int(self.res_w.text()),
                "res_h": int(self.res_h.text()),
                "cp_x": int(self.cp_x.text()),
                "cp_y": int(self.cp_y.text()),
                "cp_w": int(self.cp_w.text()),
                "cp_h": int(self.cp_h.text()),
            }
        )

    def load_config(self, dicts):
        """
        load information of each child widget
        """

        for attr in ["class_id", "res_w", "res_h"]:
            getattr(self, attr).setText(str(dicts[attr]))
        for attr in ["cp_x", "cp_y", "cp_w", "cp_h"]:
            getattr(self, attr).setValue(dicts[attr])


class childWindow(QDialog):
    """
    child dialog of QMainwindow
    set saving preference
    """

    def __init__(
        self, parent: QMainWindow | None, num_cam: int, infos: list, img_lst: list, box_config: str, cur_time: str = ""
    ):
        super().__init__(parent=parent)
        self.camera_btn_defalut_path = os.getcwd() + "/data"
        self.preview_id = 0
        self.cur_time = cur_time
        self.reso = infos[0]
        self.names = infos[1]
        self.num_cam = num_cam

        if isinstance(img_lst[0], str):
            self.img_lst = [QPixmap.fromImage(QImage(f)) for f in img_lst]
        elif isinstance(img_lst[0], bytes):
            self.img_lst = [
                QPixmap.fromImage(QImage(img, img.shape[0], img.shape[1], QImage.Format.Format_BGR888))
                for img in img_lst
            ]
        elif isinstance(img_lst[0], QImage):
            self.img_lst = [QPixmap.fromImage(i) for i in img_lst]
        else:
            self.img_lst = [None] * num_cam

        self.resize(QSize(400, 700))
        self.setWindowTitle("Saving Preference Setting")
        self.rcw_lst: List[ResCropWidget] = []
        grid = QGridLayout()

        inner = QWidget(self)
        hbox0 = QHBoxLayout()
        self.pathline = QLineEdit(self)
        self.pathline.setPlaceholderText("Default path: 'data' in current work path")
        path_sel_btn = QPushButton(self)
        path_sel_btn.setText("..")
        path_sel_btn.setFixedWidth(path_sel_btn.height())
        path_sel_btn.clicked.connect(self.select_path_btn_slot)
        outer0 = QWidget(self)
        pathrow = QFormLayout()
        pathrow.addRow("Path select:", self.pathline)
        inner.setLayout(pathrow)
        hbox0.addWidget(inner, 1)
        hbox0.addWidget(path_sel_btn, 0)
        outer0.setLayout(hbox0)
        outer0.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        grid.addWidget(outer0, 0, 0, 1, 3)

        hbox = QHBoxLayout()
        ckb1 = QCheckBox("Adjustment sync.", self)
        ckb1.setCheckable(False)
        ckb2 = QCheckBox("Lock w-h ratio", self)
        ckb2.setCheckable(False)
        ckb3 = QCheckBox("Apply crop box", self)
        self.ckb3 = ckb3

        hbox.addWidget(ckb1, 1)
        hbox.addWidget(ckb2, 1)
        hbox.addWidget(ckb3, 1)
        outer1 = QWidget(self)
        outer1.setLayout(hbox)
        outer1.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        grid.addWidget(outer1, 1, 0, 1, 3)
        maxi = (num_cam + 2) // 3
        pos = [
            (i + 2, j)
            for i in range((num_cam + 2) // 3)
            for j in range(3 if i + 1 != maxi else num_cam % 3 + (num_cam % 3 == 0) * 3)
        ]
        for idx, (p, r, n) in enumerate(zip(pos, self.reso, self.names)):
            child = ResCropWidget(self, idx, r, n, self.preview_cam_switch_slot)
            self.rcw_lst.append(child)
            grid.addWidget(child, *p)
            child._init_boxSignal(self.preview_move_slot)

        self.preview_window = preview_win(self)
        grid.addWidget(self.preview_window, pos[-1][0] + 1, 0, 1, 3, Qt.AlignmentFlag.AlignCenter)
        grid.setRowStretch(pos[-1][0] + 1, 1)
        self.load_config(box_config)
        if isinstance(self.img_lst[0], QPixmap):
            self.preview_window.setPixmap(self.img_lst[self.preview_id])

        buttonBox = QDialogButtonBox(parent=self)
        buttonBox.setOrientation(Qt.Orientation.Horizontal)  # 设置为水平方向
        buttonBox.setStandardButtons(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        grid.addWidget(buttonBox, pos[-1][0] + 2, 0, 1, 3)

        self.setLayout(grid)

    def select_path_btn_slot(self):
        """
        slot func invoke after pushing SEL_PATH button
        """

        folder = QFileDialog.getExistingDirectory(self, "Choose folder to save", directory=self.camera_btn_defalut_path)
        if folder:
            self.pathline.setText(folder)
            self.camera_btn_defalut_path = folder

    def gather_infos(self):
        """
        gather information of the whole dialog when return
        """

        dicts = dict(
            {
                "select_path": self.camera_btn_defalut_path + f"/{self.cur_time}",
                "preview_id": self.preview_id,
                "apply_cbox": self.ckb3.isChecked(),
            }
        )
    
        sub_dicts = {}
        for idx, child in enumerate(self.rcw_lst):
            sub_dicts.update({self.names[idx]: child.gather_infos()})

        dicts.update({"box_list": sub_dicts})

        return dicts

    def preview_move_slot(self, idx):
        """
        slot fun trigger when params of current active preview
        adjusted by user
        """

        if idx == self.preview_id:
            self.preview_window._update_cbox(self.rcw_lst[self.preview_id].gather_infos())

    def preview_cam_switch_slot(self, idx: int):
        """
        slot func triggered when switching preview camera
        """

        if self.preview_id != idx:
            self.preview_id = idx
            if isinstance(self.img_lst[idx], QPixmap):
                self.preview_window.setPixmap(self.img_lst[idx])
                self.preview_window._update_cbox(self.rcw_lst[idx].gather_infos())

    def load_config(self, file: str):
        """
        load information of the whole dialog when return
        """

        if os.path.exists(file):
            with open(file, "r") as f:
                dicts = json.load(f)
            self.preview_id = min(self.num_cam - 1, dicts["preview_id"])
            self.ckb3.setChecked(dicts["apply_cbox"])
            for idx, child in enumerate(self.rcw_lst):
                try:
                    child.load_config(dicts[self.names[idx]])
                except KeyError:
                    pass


# unit test
if __name__ == "__main__":
    app = QApplication(sys.argv)
    testwin = childWindow(
        None,
        6,
        [[1920, 1080, 0], [1920, 1080, 0], [1920, 1080, 0], [2560, 1440, 0], [2688, 1520, 0], [2688, 1520, 0]],
        [None],
        "data/temp/box_config.json",
    )  # ["doc/figs/"+f for f in sorted(os.listdir("doc/figs"))]
    ret = testwin.exec()
    if ret:
        if not os.path.exists("data/temp/"):
            os.makedirs("data/temp/")
            with open("data/temp/box_config.json", "r") as f:
                old_dicts = json.load(f)
            with open("data/temp/box_config.json", "w") as f:
                old_dicts.update(testwin.gather_infos())
                json.dump(old_dicts, f, indent=4)
    testwin.destroy()
    sys.exit(ret)
