import json
import os, sys
import sys
from functools import partial
from typing import List

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
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

sys.path.append(os.getcwd())
from src.Qt_ui.utils import add_html_color_tag


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
        self.is_adding_new = False

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

        self.add_item_btn = QPushButton("Add")
        self.add_item_btn.clicked.connect(self.add_new_src_slot)
        self.del_item_btn = QPushButton("Remove")
        self.del_item_btn.clicked.connect(self.delete_src_slot)
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.add_item_btn)
        hbox1.addWidget(self.del_item_btn)

        vbox = QVBoxLayout()
        vbox.addWidget(self.combobox)
        vbox.addWidget(self.listwidget)
        vbox.addLayout(hbox1)

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

        self.hint_add_label = QLabel(add_html_color_tag("Hint: adding a new video source", color="red"))
        self.hint_add_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.hint_add_label.setVisible(False)
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        vbox2 = QVBoxLayout()
        vbox2.addLayout(self.flayout)
        vbox2.addSpacerItem(spacer)
        vbox2.addWidget(self.hint_add_label)
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

    def show_configuration_slot(self, *args, empty: bool = False):
        """
        show video source configuration after change listwidget
        """

        if hasattr(self, "hint_add_label"):
            self.hint_add_label.setVisible(empty)
        row_count = self.flayout.rowCount()
        for i in range(row_count - 1, -1, -1):
            self.flayout.removeRow(i)

        text = self.combobox.currentText()
        index = self.listwidget.currentRow()
        apply_active_func = partial(self.buttonBox.button(QDialogButtonBox.StandardButton.Apply).setEnabled, True)
        for k, v in self.vid_srcs[text][index].items():
            qline = QLineEdit(self)
            qline.textChanged.connect(apply_active_func)
            if v != "" and not empty:
                qline.setText(v)
            else:
                qline.setPlaceholderText("enter content")

            if k == "NICKNAME":
                qline.setValidator(QRegExpV(QRE(r"[a-zA-Z0-9_-]{0,15}")))
            elif k == "IP":
                qline.setValidator(QRegExpV(QRE(r"[\dx]{1,3}\.[\dx]{1,3}\.[\dx]{1,3}\.[\dx]{1,3}")))
            elif k == "PORT" or k == "CHANNEL":
                qline.setValidator(QRegExpV(QRE(r"\d{1,5}")))

            self.flayout.addRow(k, qline)
        self.buttonBox.button(QDialogButtonBox.StandardButton.Apply).setEnabled(False)

        if empty:
            self.listwidget.setCurrentRow(-1)

    def update_config_slot(self):
        """
        update video source configuration after clicking apply button
        """

        is_adding_new = self.is_adding_new
        self.is_adding_new = False

        update_dict = {}
        nickname_repeat_flag, file_not_exist_flag, item_empty_flag = False, False, False
        text = self.combobox.currentText()
        index = self.listwidget.currentRow()
        for i in range(self.flayout.rowCount()):
            k = self.flayout.itemAt(i, QFormLayout.ItemRole.LabelRole).widget().text()
            v = self.flayout.itemAt(i, QFormLayout.ItemRole.FieldRole).widget().text()
            if k == "NICKNAME":
                if v in self.name_dicts[text] and v != self.vid_srcs[text][index][k]:
                    nickname_repeat_flag = True
            elif v == "":
                item_empty_flag = True
            elif k == "PATH":
                if not os.path.exists(os.path.join(self.src_dir, v)):
                    file_not_exist_flag = True

            if not nickname_repeat_flag and not file_not_exist_flag:
                update_dict[k] = v
        if nickname_repeat_flag:
            QMessageBox.warning(self, "Warning", "The nickname already exists.")
        elif file_not_exist_flag:
            QMessageBox.warning(self, "Warning", "The video file does not exist.")
        elif item_empty_flag:
            QMessageBox.warning(self, "Warning", "The item cannot be empty.")
        else:
            self.buttonBox.button(QDialogButtonBox.StandardButton.Apply).setEnabled(False)
            if is_adding_new:
                self.listwidget.addItem(update_dict["NICKNAME"])
                self.vid_srcs[text].append(update_dict)
                self.name_dicts[text].append(update_dict["NICKNAME"])
                self.hint_add_label.setVisible(False)
            else:
                self.vid_srcs[text][index] = update_dict
                self.listwidget.item(index).setText(update_dict["NICKNAME"])
                self.name_dicts[text][index] = update_dict["NICKNAME"]

        return nickname_repeat_flag or file_not_exist_flag or item_empty_flag

    def add_new_src_slot(self):
        """
        add new video source
        """

        self.is_adding_new = True
        self.show_configuration_slot(empty=True)

    def delete_src_slot(self):
        """
        delete video source
        """

        if self.listwidget.count() == 1:
            QMessageBox.warning(self, "Warning", "At least one element.")
            return
        index = self.listwidget.currentRow()
        if index == -1:
            QMessageBox.warning(self, "Warning", "Please select a video source to delete.")
        else:
            # msg_box = QMessageBox()
            # msg_box.setText("Are you sure you want to delete this video source?")
            # msg_box.setStandardButtons(QMessageBox.StandardButton.No
            # | QMessageBox.StandardButton.Yes)
            # ret = msg_box.exec()
            # print(ret, QMessageBox.DialogCode.)
            # if ret == QMessageBox.:
            text = self.combobox.currentText()
            self.vid_srcs[text].pop(index)
            self.name_dicts[text].pop(index)
            self.listwidget.takeItem(index)
            self.listwidget.setCurrentRow(-1)
            self.show_configuration_slot()

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
