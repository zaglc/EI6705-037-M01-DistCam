from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication


def generate_pos(size: tuple, num_cam: int):
    max_num = 6
    length = size[0] // 4
    width = int(length/1.3)
    index = 0
    while index < num_cam:
        yield (
            length//4 + (index % 3) * (length//4 + length), 
            length//4 + (index // 3) * (width//2 + width), 
            length, 
            width
        )
        index += 1


class Stream(QObject):
    """
    Adopt from https://blog.csdn.net/quay_sue/article/details/133841837
    """

    newText_signal = pyqtSignal(str)

    def write(self, text: str):
        self.newText_signal.emit(text)
        QApplication.processEvents()

gpc_stream = Stream()