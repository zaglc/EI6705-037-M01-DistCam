import json
import os
import sys
from functools import partial
from typing import List

from PyQt6.QtCore import QPoint, QRect
from PyQt6.QtCore import QRegularExpression as QRE
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QImage, QMoveEvent, QPixmap
from PyQt6.QtGui import QRegularExpressionValidator as QRegExpV
from PyQt6.QtGui import QResizeEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


# TODO: NICKNAME不改
class vid_src_config_window(QDialog):
    def __init__(
        self, parent: QMainWindow | None, vid_srcs: dict, current_vid_src: str, index: int, unselected: List[tuple]
    ):
        """
        vid_srcs:{
            "type":[
                    {
                        "name": "name1",
                        "path": "path1"},
                    {
                        "name": "name2",
                        "path": "path2"
                    }
                ]
        }
        """

        super().__init__(parent=parent)
        self.vid_srcs = vid_srcs
        self.setWindowTitle("Video Source Configuration")

        # left part
        start_row = list(vid_srcs.keys()).index(current_vid_src)
        self.mask_pairs = [(list(vid_srcs.keys()).index(src), idx) for (src, idx) in unselected]
        self.combobox = QComboBox(self)
        self.combobox.addItems([k for k in vid_srcs.keys()])
        self.combobox.setCurrentText(list(vid_srcs.keys())[start_row])
        self.combobox.currentTextChanged.connect(self.combobox_change_slot)
        self.combobox.setMaximumWidth(200)

        start_index = index
        self.listwidget = QListWidget(self)
        self.combobox_change_slot()
        self.listwidget.setCurrentRow(start_index)
        self.listwidget.itemDoubleClicked.connect(self.show_configuration_slot)
        self.listwidget.setMaximumWidth(200)

        vbox = QVBoxLayout()
        vbox.addWidget(self.combobox)
        vbox.addWidget(self.listwidget)

        # right part
        self.flayout = QFormLayout()
        self.buttonBox = QDialogButtonBox(parent=self)
        self.buttonBox.setOrientation(Qt.Orientation.Horizontal)  # 设置为水平方向
        self.buttonBox.setStandardButtons(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Apply
        )
        self.show_configuration_slot()
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.buttonBox.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.update_config_slot)
        self.buttonBox.button(QDialogButtonBox.StandardButton.Apply).setEnabled(False)
        self.buttonBox.setMinimumWidth(350)

        vbox2 = QVBoxLayout()
        vbox2.addLayout(self.flayout)
        vbox2.addWidget(self.buttonBox)

        hbox = QHBoxLayout()
        hbox.addLayout(vbox)
        hbox.addLayout(vbox2)
        self.setLayout(hbox)

    def combobox_change_slot(self):
        """
        switch video source type
        show new listwidget affliated to the type
        """

        self.listwidget.clear()
        row = self.combobox.currentIndex()
        for idx, v in enumerate(self.vid_srcs[self.combobox.currentText()]):
            item = QListWidgetItem(v["NICKNAME"])
            if (row, idx) in self.mask_pairs:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.listwidget.addItem(item)
        if hasattr(self, "buttonBox"):
            self.buttonBox.button(QDialogButtonBox.StandardButton.Apply).setEnabled(False)

    def show_configuration_slot(self):
        row_count = self.flayout.rowCount()
        for i in range(row_count - 1, -1, -1):
            self.flayout.removeRow(i)

        text = self.combobox.currentText()
        index = self.listwidget.currentRow()
        apply_active_func = partial(self.buttonBox.button(QDialogButtonBox.StandardButton.Apply).setEnabled, True)
        for k, v in self.vid_srcs[text][index].items():
            qline = QLineEdit(self)
            qline.textChanged.connect(apply_active_func)
            if v != "":
                qline.setText(v)
            else:
                qline.setPlaceholderText("enter content")
            self.flayout.addRow(k, qline)
        self.buttonBox.button(QDialogButtonBox.StandardButton.Apply).setEnabled(False)

    def update_config_slot(self):
        text = self.combobox.currentText()
        index = self.listwidget.currentRow()
        for i in range(self.flayout.rowCount()):
            k = self.flayout.itemAt(i, QFormLayout.ItemRole.LabelRole).widget().text()
            v = self.flayout.itemAt(i, QFormLayout.ItemRole.FieldRole).widget().text()
            if k == "NICKNAME":
                self.listwidget.item(index).setText(v)
            self.vid_srcs[text][index][k] = v
        self.buttonBox.button(QDialogButtonBox.StandardButton.Apply).setEnabled(False)

    def exec(self):
        text = self.combobox.currentText()
        index = self.listwidget.currentRow()
        ret = QDialog.exec(self)

        return ret, text, self.vid_srcs[text][index]["NICKNAME"]


if __name__ == "__main__":
    with open(os.path.join("configs", "video_source_pool.json"), "r") as f:
        vid_srcs = json.load(f)

    app = QApplication(sys.argv)
    window = vid_src_config_window(None, vid_srcs, "local-vid", 1, [("local-vid", 3), ("local-vid", 4)])
    ret = window.exec()
    print(ret)
    window.destroy()
    sys.exit(ret[0])
