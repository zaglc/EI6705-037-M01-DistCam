from multiprocessing import Array, Value
from multiprocessing.synchronize import Condition
from multiprocessing.sharedctypes import SynchronizedBase, SynchronizedArray
import cv2, numpy as np, os, datetime

# suffix 'Q' means cmd set by QThread
# otherwise by frame main
FV_QTHREAD_READY_Q = 0
FV_FRAME_PROC_READY_F = 1
FV_SWITCH_CHANNEL_Q = 2
FV_PKGLOSS_OCCUR_F = 3
FV_CAPTURE_IMAGE_Q = 4
FV_RECORD_VIDEO_Q = 5
FV_FLIP_SIMU_STREAM_Q = 6
FV_FLIP_MODEL_ENABLE_Q = 7
FV_QTHREAD_PAUSE_Q = 16

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


def pad_with_fixed_ratio(img: np.ndarray, width: int, height: int, id: int):
    old_h, old_w, _ = img.shape
    if old_h / old_w > height / width:
        w_cmp = int(old_h * width / height) // 2 - old_w // 2
        img_new = cv2.copyMakeBorder(img, 0 ,0, w_cmp, w_cmp, cv2.BORDER_CONSTANT, value=[127,127,127])
    else:
        h_cmp = int(old_w * height / width) // 2 - old_h // 2
        img_new = cv2.copyMakeBorder(img, h_cmp, h_cmp, 0, 0, cv2.BORDER_CONSTANT, value=[127,127,127])

    return img_new, img_new.shape[1], img_new.shape[0]

# TODO: useful if we need to arbitrarily pause frame window
def sync_processes(cond: Condition, cnt: SynchronizedBase, total: int):
    with cnt.get_lock():
        cnt.value += 1
        tmp = cnt.value

    with cond:
        if tmp < total:
            cond.wait()
        else:
            with cnt.get_lock():
                cnt.value = 0
            cond.notify_all()


def count_nonactive_proc(shape_cnt: SynchronizedArray, local_size: int):
    return local_size - sum([shape_cnt[4*i+3] for i in range(local_size)])


class Stream():
    """
    Adopt from https://blog.csdn.net/quay_sue/article/details/133841837
    
    log format:
    [MODEL/FRAME/CTRL x at time]: (content...)
    
    """

    def __init__(self) -> None:
        self.log_buffer = Array('c', 8192)
        self.offset = Value('i', 0)
        self.proc_table = dict()


    def write(self, text: str) -> None:
        pid = os.getpid()
        curtime = datetime.datetime.now().strftime('%H:%M:%S')
        try:
            name = self.proc_table[pid]
        except:
            name = "None-proc"

        # python `print` will invoke this method causing `end` is not None
        if len(text) > 2:
            new_text = f"[{name} at {curtime}]: " + text
        else:
            new_text = text

        with self.log_buffer.get_lock():
            with self.offset.get_lock():
                size, ofs = len(new_text), self.offset.value
                self.log_buffer[ofs: size+ofs] = new_text.encode()
                self.offset.value += size


    def _add_item(self, pid: int, name: str):
        self.proc_table.update({pid: name})


gpc_stream = Stream()