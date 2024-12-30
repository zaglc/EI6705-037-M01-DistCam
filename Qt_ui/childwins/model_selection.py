import os, json, sys

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QComboBox,
    QMainWindow,
    QSizePolicy,
    QWidget,
)


class ModelSelectionWindow(QDialog):
    """
    child dialog of QMainwindow
    set active model 
    """

    def __init__(
        self, parent: QMainWindow | None, num_cam: int, model_list: list
    ):
        super().__init__(parent=parent)
        self.camera_btn_defalut_path = os.getcwd() + "/data"
        self.model_name              = "None"
        self.num_cam                 = num_cam

        self.resize(QSize(400, 200))
        self.setWindowTitle("Set Inference Model")
        grid = QGridLayout()

        #
        # Menu for model selection
        #

        inner = QWidget(self)
        self.model_selection = QComboBox(self)
        self.model_selection.addItems(model_list)
        inner.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        grid.addWidget(inner, 0, 0, 1, 3)

        maxi = (num_cam + 2) // 3
        pos = [
            (i + 2, j)
            for i in range((num_cam + 2) // 3)
            for j in range(3 if i + 1 != maxi else num_cam % 3 + (num_cam % 3 == 0) * 3)
        ]

        #
        # Cancel and OK button
        #

        buttonBox = QDialogButtonBox(parent=self)
        buttonBox.setOrientation(Qt.Orientation.Horizontal)  # 设置为水平方向
        buttonBox.setStandardButtons(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        grid.addWidget(buttonBox, pos[-1][0] + 2, 0, 1, 3)

        self.setLayout(grid)

    def select_model_slot(self):
        """
        slot func invoke after pushing SEL_PATH button
        """
        self.model_name = self.model_selection.currentText()

    def get_model_selected(self):
        """
        """

        return self.model_selection.currentText()


if __name__ == "__main__":
    with open(os.path.join("configs", "model_cfgs", "model_config.json"), "r") as f:
        model_list = json.load(f)

    app = QApplication(sys.argv)
    window = ModelSelectionWindow(None, 2, model_list)

    ret = window.exec()
    window.destroy()
    sys.exit(ret)
