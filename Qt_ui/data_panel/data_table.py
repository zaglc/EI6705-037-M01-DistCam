from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QTableWidget, QTableWidgetItem, QHeaderView, QGridLayout,
    QToolBar, QWidget,
)


class Realtime_Datatab(QToolBar):
    """
    data panel in QT mainwindow
    now it only includes FRAME RATE and DROP RATE

    method of computing: exponetial wighted average
    """
    
    def __init__(self,
                 parent: QWidget,
                 num_cam: int,
    ):
        super().__init__("tool bar", parent)
        self.num_cam = num_cam
        self.value_dicts = [{
            "fps": [0.0, 0.9, 2],
            "drop": [0.0, 0.95, 2],
        } for _ in range(num_cam)]
        self.valud_keys = ["fps", "drop"]

        self.rs, self.cs = num_cam, 2
        self.table = QTableWidget(parent=self)

        self.table.setColumnCount(self.cs)
        colhead = ["FRAME RATE", "DROP RATE"]
        self.table.setHorizontalHeaderLabels(colhead)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch) 

        self.table.setRowCount(self.rs)
        rowhead = [f"CAM {i}" for i in range(self.rs)]
        self.table.setVerticalHeaderLabels(rowhead)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # initial table
        for i in range(self.rs):
            self._updateTabItem(i)

        grid = QGridLayout()
        grid.setSpacing(5)
        grid.addWidget(self.table, 0, 0, 5, 0)
        outer = QWidget(self)
        outer.setMinimumSize(200, 200)
        # outer.setMaximumWidth(200)
        outer.setLayout(grid)
        self.addWidget(outer)


    def _updateTabItem(self, row: int):
        """
        update data panel with current data
        """
        
        for col in range(self.cs):
            val, _, fmt = self.value_dicts[row][self.valud_keys[col]]
            if self.valud_keys[col] == "fps": val = 1/(val+1e-10)
            item = QTableWidgetItem(str(round(val, fmt)))
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, col, item)


    def _compute_slide_exp_average(self, new_val: list, row: int):
        """
        use new_val to update all value for camera row internally
        """

        for col, val in zip(range(self.cs), new_val):
            aver, rate, _ = self.value_dicts[row][self.valud_keys[col]]
            aver = aver * rate + val * (1 - rate)
            self.value_dicts[row][self.valud_keys[col]][0] = aver