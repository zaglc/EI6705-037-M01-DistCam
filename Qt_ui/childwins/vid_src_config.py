import json
import os
import sys
from functools import partial
from typing import List

from PyQt6.QtCore import QPoint, QRect
from PyQt6.QtCore import QRegularExpression as QRE
from PyQt6.QtCore import QSize, Qt
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
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


# TODO: NICKNAME不改
class vid_src_config_window(QDialog):
    def __init__(
        self,
        parent: QWidget | None,
        vid_srcs: dict,
        current_vid_src: str,
        index: int,
        src_dir: str,
        unselected: List[tuple],
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
        self.src_dir = src_dir
        self.setWindowTitle("Video Source Configuration")

        # left part
        start_row = list(vid_srcs.keys()).index(current_vid_src)
        self.name_dicts = {
            src_type: [vv["NICKNAME"] for vv in v] for src_type, v in vid_srcs.items()
        }  # store current selected name, avoid conflict
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

            if k == "NICKNAME":
                qline.setValidator(QRegExpV(QRE(r"[a-zA-Z0-9_]")))
            elif k == "IP":
                qline.setValidator(QRegExpV(QRE(r"[\dx]{1,3}\.[\dx]{1,3}\.[\dx]{1,3}\.[\dx]{1,3}")))
            elif k == "PORT" or k == "CHANNEL":
                qline.setValidator(QRegExpV(QRE(r"\d{1,5}")))

            self.flayout.addRow(k, qline)
        self.buttonBox.button(QDialogButtonBox.StandardButton.Apply).setEnabled(False)

    def update_config_slot(self):
        """
        update video source configuration after clicking apply button
        """

        nickname_repeat_flag, file_not_exist_flag = False, False
        text = self.combobox.currentText()
        index = self.listwidget.currentRow()
        for i in range(self.flayout.rowCount()):
            k = self.flayout.itemAt(i, QFormLayout.ItemRole.LabelRole).widget().text()
            v = self.flayout.itemAt(i, QFormLayout.ItemRole.FieldRole).widget().text()
            if k == "NICKNAME":
                if v in self.name_dicts[text] and v != self.vid_srcs[text][index][k]:
                    nickname_repeat_flag = True
                else:
                    self.listwidget.item(index).setText(v)
            elif k == "PATH":
                if not os.path.exists(os.path.join(self.src_dir, v)):
                    file_not_exist_flag = True

            if not nickname_repeat_flag and not file_not_exist_flag:
                self.vid_srcs[text][index][k] = v
        if nickname_repeat_flag:
            QMessageBox.warning(self, "Warning", "The nickname already exists.")
        elif file_not_exist_flag:
            QMessageBox.warning(self, "Warning", "The video file does not exist.")
        else:
            self.buttonBox.button(QDialogButtonBox.StandardButton.Apply).setEnabled(False)

        return nickname_repeat_flag or file_not_exist_flag

    def accept(self):
        """
        slot when click ok button
        """

        flag = self.update_config_slot()
        if not flag:
            ret = super().accept()
        else:
            ret = None

        return ret

    def exec(self):

        ret = QDialog.exec(self)
        text = self.combobox.currentText()
        index = self.listwidget.currentRow()
        return ret, text, index, self.vid_srcs


if __name__ == "__main__":
    with open(os.path.join("configs", "video_source_cfgs", "video_source_pool.json"), "r") as f:
        vid_srcs = json.load(f)

    app = QApplication(sys.argv)
    window = vid_src_config_window(
        None, vid_srcs["sources"], "local-vid", 1, "data/src", [("local-vid", 3), ("local-vid", 4)]
    )
    ret = window.exec()
    print(ret[:3])
    window.destroy()
    sys.exit(ret[0])
