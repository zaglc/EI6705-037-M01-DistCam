from PyQt6.QtWidgets import QApplication, QDialog, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem, QLabel, QWidget, QGraphicsPolygonItem, QFormLayout, QHBoxLayout, QLineEdit, QGraphicsPixmapItem, QGroupBox, QDialogButtonBox, QVBoxLayout, QTabWidget
from PyQt6.QtGui import QPen, QColor, QPixmap, QPolygonF, QCursor, QDoubleValidator
from PyQt6.QtCore import Qt, QRectF, QPointF, QPoint, pyqtSignal

from typing import List
import math
import os, sys
from functools import partial

sys.path.append(os.getcwd())
from Qt_ui.utils import FRAME_RATIO

class DraggableRectItem(QGraphicsRectItem):
    """
    each draggable point on the figure
    """

    def __init__(self, x, y, size, idx, outer, parent=None):
        super().__init__(x, y, size, size, parent)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
                     QGraphicsItem.GraphicsItemFlag.ItemIsFocusable | 
                     QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
                     QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)

        self._w = size
        self._h = size
        self.idx = idx
        self.x_offset = x
        self.y_offset = y
        self.auto_align = False
        self.change_from_text = False
        self.outer = outer
        self.has_reference_point = False
        self.reference_tuple = None
        self.setBrush(QColor(255, 0, 0))

    def set_rect(self, rect: QGraphicsPolygonItem):
        self.slave_rect = rect

    def base_pos(self, basex=None, basey=None):
        """
        basex, basey: base position of leftup cornor in this rect node, axis is itself
        return: center position of this rect node, axis is scene
        """

        if basex is None:
            basex = self.x()
        if basey is None:
            basey = self.y()
        return QPointF(self.x_offset + basex + self._w/2, self.y_offset + basey + self._h/2)
    
    def reverse_base_pos(self, outerx=0, outery=0):

        return QPointF(outerx - self.x_offset - self._w/2, outery - self.y_offset - self._h/2)

    def w(self):
        return self._w, self.acceptHoverEvents

    def h(self):
        return self._h

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange or change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # limit x and y's coordinates
            new_pos = value
            center_pos = QPointF(QPoint(int(new_pos.x() + self._w/2 + self.x_offset), int(new_pos.y() + self._h/2 + self.y_offset)))
            # print("begin", new_pos, center_pos)
            rect = self.scene().sceneRect()
            if not rect.contains(center_pos) and not self.change_from_text:
                new_pos.setX(min(rect.right() - self._w/2 - self.x_offset, max(rect.left() - self._w/2 - self.x_offset, new_pos.x())))
                new_pos.setY(min(rect.bottom() - self._h/2 - self.y_offset, max(rect.top() - self._h/2 - self.y_offset, new_pos.y())))
            
            if self.auto_align and not self.change_from_text:
                if not self.has_reference_point:
                    is_horizontal, val = self.outer.find_nearest_point(self.idx)
                    self.reference_tuple = (is_horizontal, val)
                    self.has_reference_point = True
                else:
                    is_horizontal, val = self.reference_tuple
                # print(is_horizontal, val)
                if is_horizontal:
                    new_pos.setX(self.reverse_base_pos(outerx=val).x())
                else:
                    new_pos.setY(self.reverse_base_pos(outery=val).y())

            self.change_from_text = False

            polygen = self.slave_rect.polygon()
            polygen.replace(self.idx, self.base_pos(new_pos.x(), new_pos.y()))
            self.slave_rect.setPolygon(polygen)

            if hasattr(self.outer, "show_func"):
                self.outer.show_func(self.outer.get_coords())
                        
            return new_pos

        return super().itemChange(change, value)
    
    def distance_to(self, point: QPointF):
        return math.sqrt((self.base_pos().x() - point.x()) ** 2 + (self.base_pos().y() - point.y()) ** 2)
    
    def mousePressEvent(self, event):
        self.has_reference_point = False
        self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        return super().mouseReleaseEvent(event)


class DraggableAreaWidget(QWidget):
    """
    QObject wrapper
    default sequence
        [0]---[1]
         |     |
        [3]---[2]
    """

    pointChange = pyqtSignal(tuple)
    def __init__(self, scene, poly=4, size=10, parent=None):
        super().__init__(parent)
        self.poly = poly
        self.scene = scene
        self.dragpint_size = size
        self._draggableRectItem_lst: List[DraggableRectItem] = []

    def init_polygen(self):
        bounding_rect = self.scene.sceneRect()
        cx, cy = bounding_rect.center().x(), bounding_rect.center().y()
        size = min(bounding_rect.width(), bounding_rect.height()) // 2
        for i in range(self.poly):
            x = cx + size * math.cos(2 * math.pi * i / self.poly) - self.dragpint_size/2
            y = cy + size * math.sin(2 * math.pi * i / self.poly) - self.dragpint_size/2
            self._addPoint(x, y, i)
        self.rectItem = QGraphicsPolygonItem(QPolygonF([self[i].base_pos(0, 0) for i in range(self.poly)]))
        self.rectItem.setBrush(QColor(255, 0, 0, 50))
        self.rectItem.setPen(QPen(QColor(255, 0, 0), 2))
        for i in range(self.poly):
            self[i].set_rect(self.rectItem)

    def _addPoint(self, x, y, idx):
        point = DraggableRectItem(x, y, self.dragpint_size, idx, self)
        self._draggableRectItem_lst.append(point)

    def __getitem__(self, index: int):
        return self._draggableRectItem_lst[index]

    def __len__(self):
        return len(self._draggableRectItem_lst)
    
    def set_auto_align(self, auto_align: bool):
        """
        'shift' key slot by outer QWidget
        """

        for i in range(self.poly):
            self[i].auto_align = auto_align
            if not auto_align:
                self[i].has_reference_point = False

    def find_nearest_point(self, point_idx):
        """
        when 'shift' is pressed, the anchor point should move horizontally or vertically
        so need to find the nearest point as the reference
        """

        # find nearest point
        fixed_position = self[point_idx].base_pos()
        nearest_idx = min([i for i in range(self.poly) if i != point_idx], key=lambda i: self[i].distance_to(fixed_position))
        
        is_horizontal = abs(self[point_idx].base_pos().x() - self[nearest_idx].base_pos().x()) < abs(self[point_idx].base_pos().y() - self[nearest_idx].base_pos().y())
        if is_horizontal:
            val = self[nearest_idx].base_pos().x()
        else:
            val = self[nearest_idx].base_pos().y()

        return is_horizontal, val
    
    def get_coords(self, size=None):
        """
        format: list of tuple int
        """
        if size is not None:
            # normalize
            xx, yy = size
        else:
            xx, yy = 1, 1
        return [(self[i].base_pos().x()/xx, self[i].base_pos().y()/yy) for i in range(self.poly)]
    
    def set_show_func(self, func):
        """
        binding function in XY_coords
        """

        self.show_func = func


class Draggable_previewWidget(QWidget):
    def __init__(self, poly=4, qimage=None, parent=None, height=360):
        super().__init__(parent)
        # 创建一个 QLabel 对象，作为背景
        pixmap = QPixmap(os.path.join("doc", "figs", "layout", "v2.png")) if qimage is None else QPixmap.fromImage(qimage)
    
        # 创建一个 QGraphicsScene 对象
        scene = QGraphicsScene()
        scene.setBackgroundBrush(QColor(255, 0, 0, 0))
        self.scene = scene

        # 创建一个 QGraphicsView 对象，并将 QGraphicsScene 设置为其场景
        view_size, ratio = (int(FRAME_RATIO * height), height), 0.95
        inner_view_size = (int(view_size[0] * ratio), int(view_size[1] * ratio))
        view = QGraphicsView(scene, self)
        view.setGeometry(0, 0, pixmap.width()//2, pixmap.height()//2)
        view.setMinimumSize(*view_size)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setStyleSheet("background-color: rgba(255, 0, 0, 0);")
        scene.setSceneRect(0, 0, *inner_view_size)
        self.view = view

        pixmap = pixmap.scaled(*inner_view_size, Qt.AspectRatioMode.KeepAspectRatio)
        item = QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.pixmap = pixmap

        # 创建四个 QGraphicsEllipseItem 对象，作为四边形的角点
        points = DraggableAreaWidget(parent=self, poly=poly, scene=scene)
        points.init_polygen()
        scene.addItem(points.rectItem)
        for i in range(points.poly):
            scene.addItem(points[i])
        self.points = points
        self.inner_view_size = inner_view_size


class XY_coord(QWidget):
    def __init__(self, scene_size, poly=4, parent=None):
        super().__init__(parent)
        self.poly = poly
        self.scene_size = scene_size

        self.flayoutx = QFormLayout()
        self.flayouty = QFormLayout()
        self.xlines_list: List[QLineEdit] = []
        self.ylines_list: List[QLineEdit] = []
        for i in range(self.poly):
            xline = QLineEdit(self)
            xline.setValidator(QDoubleValidator(0, 1, 3))
            self.flayoutx.addRow(f"X{i}", xline)
            self.xlines_list.append(xline)
            yline = QLineEdit(self)
            yline.setValidator(QDoubleValidator(0, 1, 3))
            self.flayouty.addRow(f"Y{i}", yline)
            self.ylines_list.append(yline)

        hbox = QHBoxLayout()
        hbox.addLayout(self.flayoutx)
        hbox.addLayout(self.flayouty)
        groupbox = QGroupBox("X-Y coordinates")
        groupbox.setLayout(hbox)

        # self.label = QLabel("Set Alert Area: The system will alert you when any target is in the area")
        self.buttonbox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        vbox = QVBoxLayout()
        # vbox.addWidget(self.label)
        vbox.addWidget(groupbox)
        vbox.addWidget(self.buttonbox)

        self.setLayout(vbox)

    def show_xycoords_slot(self, points: List[tuple]):
        """
        fill qline with xy coords (normalized)
        """

        for i in range(self.poly):
            x = round(points[i][0] / self.scene_size[0], 3)
            y = round(points[i][1] / self.scene_size[1], 3)
            self.flayoutx.itemAt(i, QFormLayout.ItemRole.FieldRole).widget().setText(str(x))
            self.flayouty.itemAt(i, QFormLayout.ItemRole.FieldRole).widget().setText(str(y))

    def gather_xycoords_slot(self, tar_idx=None):
        """
        get xy coords from qline in list of tuple int format
        """

        points = []
        for i in range(self.poly):
            if tar_idx is not None and tar_idx != i:
                continue

            x = self.flayoutx.itemAt(i, QFormLayout.ItemRole.FieldRole).widget().text()
            y = self.flayouty.itemAt(i, QFormLayout.ItemRole.FieldRole).widget().text()
            if x == "" or y == "":
                return None
            x, y = float(x), float(y)            
            points.append((round(x * self.scene_size[0]), round(y * self.scene_size[1])))
        
        return points

class Advanced_window(QDialog):
    def __init__(self, num_cam: int, names: List[str], poly=4, qimage_lst=None):
        super().__init__()
        self.num_cam = num_cam
        self.current_active = 0
        self.setWindowTitle("Advanced Setting Window")

        # preview window
        self.preview_lst: List[Draggable_previewWidget] = []
        for i in range(self.num_cam):
            self.preview_lst.append(Draggable_previewWidget(poly=poly, qimage=qimage_lst[i] if qimage_lst is not None else None, parent=self))

        # preview arranged in tab
        self.preview_tab = QTabWidget()
        for i in range(self.num_cam):
            self.preview_tab.addTab(self.preview_lst[i].view, names[i])
        self.preview_tab.currentChanged.connect(self.preview_tab_changed_slot)

        # xy coord and layout
        self.xy_coord = XY_coord(self.preview_lst[0].inner_view_size, poly=poly)
        self.xy_coord.show_xycoords_slot(self.preview_lst[0].points.get_coords())
        for i in range(self.num_cam):
            self.preview_lst[i].points.set_show_func(self.xy_coord.show_xycoords_slot)
        for i in range(poly):
            self.xy_coord.xlines_list[i].textChanged.connect(partial(self.update_points_coord_slot, i))
            self.xy_coord.ylines_list[i].textChanged.connect(partial(self.update_points_coord_slot, i))
        self.xy_coord.buttonbox.accepted.connect(self.accept)
        self.xy_coord.buttonbox.rejected.connect(self.reject)
        
        hbox = QHBoxLayout()
        hbox.addWidget(self.preview_tab)
        hbox.addWidget(self.xy_coord)
        self.setLayout(hbox)
        
    def keyPressEvent(self, a0):

        if a0.key() == Qt.Key.Key_Shift:
            self.preview_lst[self.current_active].points.set_auto_align(True)
        return super().keyPressEvent(a0)
    
    def keyReleaseEvent(self, a0):
        if a0.key() == Qt.Key.Key_Shift:
            self.preview_lst[self.current_active].points.set_auto_align(False)
        return super().keyReleaseEvent(a0)
    
    def update_points_coord_slot(self, tar_idx: int):
        """
        when value in lineEdit change, change position of point
        """

        points = self.xy_coord.gather_xycoords_slot(tar_idx)
        if points is None:
            return

        pointsItem = self.preview_lst[self.current_active].points
        pointsItem[tar_idx].setPos(pointsItem[tar_idx].reverse_base_pos(points[0][0], points[0][1]))

    def preview_tab_changed_slot(self):
        """
        click tab
        """

        self.current_active = self.preview_tab.currentIndex()
        self.xy_coord.show_xycoords_slot(self.preview_lst[self.current_active].points.get_coords())

    def exec(self):
        
        ret = QDialog.exec(self)
        points_lst = []
        for i in range(self.num_cam):
            points_lst.append(self.preview_lst[i].points.get_coords(self.preview_lst[self.current_active].inner_view_size))

        return ret, points_lst


if __name__ == '__main__':
    app = QApplication([])
    window = Advanced_window(2, ["test1", "test2"])
    window.resize(700, 300)
    ret = window.exec()
    print(ret)
    sys.exit()
