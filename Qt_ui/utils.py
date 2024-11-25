import datetime
import os
from multiprocessing import Queue
from queue import Empty
from queue import Queue as TQueue
from typing import Callable

import cv2
import numpy as np

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
FV_QTHREAD_PAUSE_Q = 16

RS_STOP = 0
RS_WAITING = 1
RS_RUNNING = 2

FV_STOP = 0
FV_RUNNING = 2

MAX_QUEUE_WAIT_TIME = 5


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


def safe_get(queue: Queue | TQueue, name: str, quit_func: Callable = None):
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
    [MODEL/FRAME/CTRL x at time]: (content...)

    """

    def __init__(self) -> None:

        self.log_buffer = Queue()
        self.proc_table = dict()

    def write(self, text: str) -> None:
        pid = os.getpid()
        curtime = datetime.datetime.now().strftime("%H:%M:%S")
        try:
            name = self.proc_table[pid]
        except:
            name = "None-proc"

        # python `print` will invoke this method causing `end` is not None
        if len(text) > 2:
            new_text = f"[{name} at {curtime}]: " + text + "\n"
        else:
            new_text = text

        self.log_buffer.put(new_text)

    def flush(self) -> None:
        # TODO: if needed
        pass

    def _add_item(self, pid: int, name: str):
        self.proc_table.update({pid: name})


gpc_stream = Stream()
