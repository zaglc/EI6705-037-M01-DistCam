import json
import os
import sys
from functools import partial
from typing import List

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QDoubleValidator, QIntValidator
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ModelSelectionWindow(QDialog):
    """
    child dialog of QMainwindow
    set active model
    """

    def __init__(
        self,
        parent: QMainWindow | None,
        num_cam: int,
        model_config: dict,
        current_active_model: str,
        is_active: bool,
        selected_class: List[int],
    ):
        """
        different model share the same class mask
        """

        super().__init__(parent=parent)
        self.model_config = model_config
        self.num_cam = num_cam
        self.is_active = is_active
        self.current_active_model = current_active_model
        with open(self.model_config[self.current_active_model]["names"]) as f:
            class_names = [c.replace("\n", "") for c in f.readlines()]
        self.class_length = len(class_names)

        self.setMinimumSize(QSize(450, 600))
        self.setWindowTitle("Set Inference Model")
        vbox = QVBoxLayout()

        # Menu for model selection
        self.model_select_comb = QComboBox(self)
        self.model_select_comb.addItems(list(model_config.keys()))
        self.model_select_comb.setCurrentText(current_active_model)
        self.model_select_comb.currentTextChanged.connect(self.model_select_slot)
        self.enable_infer_box = QCheckBox("Enable Inference", self)
        self.enable_infer_box.setChecked(self.is_active)
        self.enable_infer_box.clicked.connect(self.click_ckbx_slot)
        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.model_select_comb)
        vbox1.addWidget(self.enable_infer_box)
        group1 = QGroupBox("Model Selection", self)
        group1.setLayout(vbox1)
        vbox.addWidget(group1)

        # Details for hyper params
        contents = ["conf_thre", "iou_thre", "img_size", "batch_size"]
        hint_texts_validator = [
            ("confidence threshold, range: [0, 1]", QDoubleValidator(0, 1, 2)),
            ("iou threshold, range: [0, 1]", QDoubleValidator(0, 1, 2)),
            ("image size, an integer, e.g. 512", QIntValidator(128, 1920)),
            ("batch size, an integer, e.g. 2", QIntValidator(1, 4)),
        ]
        self.flayout = QFormLayout()
        for cont, (ht, valid) in zip(contents, hint_texts_validator):
            qline = QLineEdit(self)
            qline.setValidator(valid)
            qline.setPlaceholderText(ht)
            self.flayout.addRow(cont, qline)
        self.model_select_slot()
        group2 = QGroupBox("Hyper Parameters", self)
        group2.setLayout(self.flayout)
        vbox.addWidget(group2)

        # Classes mask
        vbox2 = QVBoxLayout()
        self.all_choose_button = QPushButton("Choose All", self)
        self.all_choose_button.clicked.connect(partial(self.flush_cls_slot, True))
        self.all_cancel_button = QPushButton("Cancel All", self)
        self.all_cancel_button.clicked.connect(partial(self.flush_cls_slot, False))
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.all_choose_button)
        hbox1.addWidget(self.all_cancel_button)
        vbox2.addLayout(hbox1)

        self.list_widget = QListWidget(self)
        self.list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        for idx, cls in enumerate(class_names):
            ckbx = QCheckBox(f"{str(idx).zfill(2)}: {cls}", self)
            if len(selected_class) > 0 and idx == selected_class[0]:
                ckbx.setChecked(True)
                selected_class.pop(0)

            item = QListWidgetItem(self.list_widget)
            item.setSizeHint(QSize(0, 25))
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, ckbx)
        vbox2.addWidget(self.list_widget)
        group3 = QGroupBox("Classes Mask", self)
        group3.setLayout(vbox2)
        vbox.addWidget(group3)

        # Cancel and OK button
        vbox.addStretch(1)
        self.buttonBox = QDialogButtonBox(parent=self)
        self.buttonBox.setOrientation(Qt.Orientation.Horizontal)  # 设置为水平方向
        self.buttonBox.setStandardButtons(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Apply
        )
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.buttonBox.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.update_model_config_slot)
        vbox.addWidget(self.buttonBox)

        self.setLayout(vbox)
        self.click_ckbx_slot()

    def flush_cls_slot(self, set_enable: bool):
        """
        for Choose All and Cancel All button
        """

        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            widget = self.list_widget.itemWidget(item)
            widget.setChecked(set_enable)

    def click_ckbx_slot(self):
        """
        slot func invoke after clicking a checkbox
        """

        state = self.enable_infer_box.isChecked()
        self.is_active = state
        self.model_select_comb.setEnabled(not state)
        self.list_widget.setEnabled(not state)
        self.all_choose_button.setEnabled(not state)
        self.all_cancel_button.setEnabled(not state)
        for i in range(self.flayout.rowCount()):
            self.flayout.itemAt(i, QFormLayout.ItemRole.FieldRole).widget().setEnabled(not state)

    def model_select_slot(self):
        """
        slot func invoke after selecting a model
        """

        self.current_active_model = self.model_select_comb.currentText()
        for i in range(self.flayout.rowCount()):
            k = self.flayout.itemAt(i, QFormLayout.ItemRole.LabelRole).widget().text()
            v_obj = self.flayout.itemAt(i, QFormLayout.ItemRole.FieldRole).widget()
            v_obj.setText(str(self.model_config[self.current_active_model][k]))

    def update_model_config_slot(self):
        """
        slot func invoke after clicking apply button
        """

        current_active_model = self.model_select_comb.currentText()
        for i in range(self.flayout.rowCount()):
            k = self.flayout.itemAt(i, QFormLayout.ItemRole.LabelRole).widget().text()
            v = self.flayout.itemAt(i, QFormLayout.ItemRole.FieldRole).widget().text()
            self.model_config[current_active_model][k] = float(v) if "thre" in k else int(v)

    def accept(self):
        """
        slot when click ok button
        """

        self.update_model_config_slot()
        ret = super().accept()

        return ret

    def exec(self):

        ret = QDialog.exec(self)
        text = self.model_select_comb.currentText()
        model_config = self.model_config

        selected_class = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            widget = self.list_widget.itemWidget(item)
            if widget.isChecked():
                selected_class.append(i)

        return ret, text, self.is_active, model_config, selected_class


if __name__ == "__main__":
    with open(os.path.join("configs", "model_cfgs", "model_config.json"), "r") as f:
        model_config = json.load(f)

    app = QApplication(sys.argv)
    window = ModelSelectionWindow(None, 2, model_config, "yolov3-detect", False, [0, 2])

    ret = window.exec()
    print(ret)
    window.destroy()
    sys.exit(ret[0])
