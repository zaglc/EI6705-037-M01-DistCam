import datetime
import os
from multiprocessing import Queue
from queue import Empty
from queue import Queue as TQueue
from typing import Callable

import cv2
import numpy as np
from PyQt6.QtWidgets import QGridLayout, QLayout, QWidget

# suffix 'Q' means cmd set by QThread
# otherwise by frame main
FV_QTHREAD_READY_Q = 2
FV_FRAME_PROC_READY_F = 3
FV_SWITCH_CHANNEL_Q = 4
FV_PKGLOSS_OCCUR_F = 5
FV_CAPTURE_IMAGE_Q = 6
FV_RECORD_VIDEO_Q = 7
FV_FLIP_SIMU_STREAM_Q = 8
FV_FLIP_MODEL_ENABLE_Q = 9
FV_PTZ_CTRL_Q = 10
FV_UPDATE_VID_INFO_F = 11
FV_QTHREAD_PAUSE_Q = 16

RS_STOP = 0
RS_WAITING = 1
RS_RUNNING = 2

FV_STOP = 0
FV_RUNNING = 2

MAX_QUEUE_WAIT_TIME = 5

# path to save box_config.json
BOX_JSON_PATH = os.path.join("configs", "crop_box_cfgs", "box_config_temp.json")
if not os.path.exists(BOX_JSON_PATH):
    BOX_JSON_PATH = BOX_JSON_PATH.replace("box_config_temp", "box_config")

# path to load model_config.json
MODEL_JSON_PATH = os.path.join("configs", "model_cfgs", "model_config_temp.json")
if not os.path.exists(MODEL_JSON_PATH):
    MODEL_JSON_PATH = MODEL_JSON_PATH.replace("model_config_temp", "model_config")

# path to load video source pool
VIDEO_SOURCE_POOL_PATH = os.path.join("configs", "video_source_cfgs", "video_source_pool_temp.json")
if not os.path.exists(VIDEO_SOURCE_POOL_PATH):
    VIDEO_SOURCE_POOL_PATH = VIDEO_SOURCE_POOL_PATH.replace("video_source_pool_temp", "video_source_pool")

# frame window ratio
FRAME_RATIO = 16 / 9

# total zoom level
FRAME_ZOOM_LEVEL = 10

# type of video sources
LOCAL_VID = "local-vid"
IP_CAM = "ip-cam"
HIKVISION = "hikvision"

# data panel: 
DATA_PANEL_UPDATE_INTERVAL = 5

def compute_best_size4view_panel(
    inner_widget: QWidget, outer_widget: QWidget, outer_widget_layout: QLayout, middle_widget_layout: QLayout
):
    """
    compute best size for view panel, when single view is activated

    margins exist between ctw and dis, dis and frame_win
    ctw automatically has Layout: QVBoxLayout
    """

    left, up, right, bottom = outer_widget_layout.getContentsMargins()
    m_left, m_up, m_right, m_bottom = middle_widget_layout.getContentsMargins()
    inner_size = inner_widget.width(), inner_widget.height()
    pad_x = outer_widget.width() - left - right - m_left - m_right - inner_size[0]
    pad_y = outer_widget.height() - up - bottom - m_up - m_bottom - inner_size[1]
    # print("inner", inner_widget.width(), inner_widget.height())
    # print("outer", outer_widget.width(), outer_widget.height(), pad_x, pad_y)

    # expand or shrink
    min_pad_w = min(round(pad_y * FRAME_RATIO), pad_x)
    min_pad_h = round(min_pad_w / FRAME_RATIO)
    target_size = (inner_size[0] + min_pad_w + m_left + m_right, inner_size[1] + min_pad_h + m_up + m_bottom)

    # print("target size", target_size)

    return target_size


def add_html_color_tag(text: str, color: str, bold: bool = False) -> str:
    """
    add html color tag to text
    """

    if not bold:
        return f"<font color={color}>{text}</font>"
    else:
        return f"<font color={color} style='font-weight:bold'>{text}</font>"


def generate_pos(size: tuple, num_cam: int):
    max_num = 6
    length = size[0] // 4
    width = int(length / 1.3)
    index = 0
    while index < num_cam:
        yield (
            length // 4 + (index % 3) * (length // 4 + length),
            length // 4 + (index // 3) * (width // 2 + width),
            length,
            width,
        )
        index += 1


def pad_with_fixed_ratio(img: np.ndarray, width: int, height: int, id: int):
    old_h, old_w, _ = img.shape
    if old_h / old_w > height / width:
        w_cmp = int(old_h * width / height) // 2 - old_w // 2
        img_new = cv2.copyMakeBorder(img, 0, 0, w_cmp, w_cmp, cv2.BORDER_CONSTANT, value=[127, 127, 127])
    else:
        h_cmp = int(old_w * height / width) // 2 - old_h // 2
        img_new = cv2.copyMakeBorder(img, h_cmp, h_cmp, 0, 0, cv2.BORDER_CONSTANT, value=[127, 127, 127])

    return img_new, img_new.shape[1], img_new.shape[0]


def safe_get(queue: Queue, name: str, quit_func: Callable = None):
    """
    for blocking queue get, if
    """

    try:
        return 1, queue.get(timeout=MAX_QUEUE_WAIT_TIME)
    except Empty:
        print(f"ERROR, queue {name} waiting time out!!!")
        if quit_func is not None:
            quit_func()
        return 0, None


class Stream:
    """
    Adopt from https://blog.csdn.net/quay_sue/article/details/133841837

    log format:
    [time] [process] content, separate by '\t'

    """

    def __init__(self) -> None:

        self.log_buffer = Queue()
        self.proc_table = dict()

    def write(self, text: str) -> None:
        pid = os.getpid()
        try:
            name = self.proc_table[pid]
        except:
            name = "None-proc"

        # python `print` will invoke this method causing `end` is not None
        if len(text) > 2:
            new_text = f"{name}@" + text
        else:
            new_text = text

        self.log_buffer.put(new_text)

    def flush(self) -> None:
        # TODO: if needed
        pass

    def _add_item(self, pid: int, name: str):
        self.proc_table.update({pid: name})


gpc_stream = Stream()
