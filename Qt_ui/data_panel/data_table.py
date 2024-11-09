from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView, QGridLayout


class Realtime_Datatab(QtWidgets.QToolBar):
    """
    data panel in QT mainwindow
    now it only includes FRAME RATE and DROP RATE

    method of computing: exponetial wighted average
    """
    
    def __init__(self,
                 parent: QtWidgets.QWidget,
                 num_cam: int,
    ):
        super().__init__("tool bar", parent)
        self.num_cam = num_cam
        self.value_dicts = [{
            "fps": [0.0, 0.9, 2],
            "drop": [0.0, 0.95, 2],
        } for _ in range(num_cam)]
        self.valud_keys = ["fps", "drop"]

        self.rs, self.cs = 2, num_cam
        self.table = QTableWidget(parent=self)
        self.table.setRowCount(2)
        rowhead = ["FRAME RATE/fps", "DROP RATE"]
        self.table.setVerticalHeaderLabels(rowhead)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch) 

        self.table.setColumnCount(num_cam)
        colhead = [f"CAM {i}" for i in range(num_cam)]
        self.table.setHorizontalHeaderLabels(colhead)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # initial table
        for j in range(self.cs):
            self._updateTabItem(j)

        grid = QGridLayout()
        grid.setSpacing(5)
        grid.addWidget(self.table, 0, 0, 0, 14)
        outer = QtWidgets.QWidget(self)
        outer.setMinimumSize(800, 60)
        outer.setLayout(grid)

        # self.setLayout(grid)
        self.grid = grid
        self.addWidget(outer)


    def _updateTabItem(self, col: int):
        """
        update data panel with current data
        """
        
        for row in range(self.rs):
            val, _, fmt = self.value_dicts[col][self.valud_keys[row]]
            if self.valud_keys[row] == "fps": val = 1/(val+1e-10)
            item = QTableWidgetItem(str(round(val, fmt)))
            item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, col, item)


    def _compute_slide_exp_average(self, new_val: list, col: int):
        """
        use new_val to update all value for camera col internally
        """

        for row, val in zip(range(self.rs), new_val):
            aver, rate, _ = self.value_dicts[col][self.valud_keys[row]]
            aver = aver * rate + val * (1 - rate)
            self.value_dicts[col][self.valud_keys[row]][0] = aver

        