import json
import os
import sys
import time
from multiprocessing import Process, Queue
from queue import Empty
from queue import Queue as TQueue
from typing import List
import argparse

import numpy as np
import torch
from PyQt6.QtWidgets import QApplication

from central_monitor.camera_top import Camera
from Qt_ui.mainwin import custom_window
from Qt_ui.utils import (
    FV_CAPTURE_IMAGE_Q,
    FV_FLIP_MODEL_ENABLE_Q,
    FV_FLIP_SIMU_STREAM_Q,
    FV_FRAME_PROC_READY_F,
    FV_PKGLOSS_OCCUR_F,
    FV_PTZ_CTRL_Q,
    FV_QTHREAD_PAUSE_Q,
    FV_RECORD_VIDEO_Q,
    FV_RUNNING,
    FV_STOP,
    FV_SWITCH_CHANNEL_Q,
    RS_RUNNING,
    RS_STOP,
    RS_WAITING,
    Stream,
    gpc_stream,
    safe_get,
)
from running_models.extern_model import (
    initialize_model,
    non_max_suppression,
    preprocess_img,
    process_result,
)

WAITING_TIME = 0.01

# TODO: so many todo...
def model_Main(
    local_num_cam: int,
    ddp_id: int,
    data_queue: Queue,
    result_queue: List[Queue],
    stream: Stream,
) -> None:
    """
    process for model inference
    """

    def quit_func():
        sys.stdout = sys.__stdout__

    # sys.stdout = stream
    stream._add_item(os.getpid(), f"MODEL {ddp_id}")

    # model loading and cuda memory pre-allocating
    batch = 1
    running_status = RS_WAITING
    model, device, classes, colors = initialize_model(local_num_cam)
    for queue in result_queue:
        queue.put((classes, colors))

    while True:
        # if model inference is needed, data to be inferenced will be in data_queue
        # rank-0 send start and end signal
        fetch, tsr_lst, sender_lst = 0, [], []
        while fetch < batch:
            if running_status == RS_WAITING:
                (vid_id, running_status, _) = data_queue.get()

            if running_status == RS_WAITING:
                continue
            elif running_status == RS_STOP:
                break
            else:
                try:
                    (vid_id, running_status, tsr) = data_queue.get(timeout=WAITING_TIME)
                    if running_status == RS_RUNNING:
                        if tsr is not None:
                            tsr_lst.append(tsr)
                            sender_lst.append(vid_id)
                            fetch += 1
                    else:
                        size = data_queue.qsize()
                        for _ in range(size):
                            data_queue.get()
                        break
                except Empty:
                    break

        # no matter which frame submit data, inference will be executed when enough data is collected
        if fetch > 0:
            chunk_tsr = torch.stack(tsr_lst, dim=0)
            chunk_tsr = chunk_tsr.to(device)
            results = model(chunk_tsr, augment=False)[0]
            results = results.clone().detach().cpu()

        # dispatch results to corresponding process
        for i in range(fetch):
            results = non_max_suppression(results, conf_thres=0.2, iou_thres=0.4, multi_label=False)
            result_queue[sender_lst[i]].put((results[i], ))

        # process image and execute inference
        # preprocess_img(chunk_tsr, img_lst, img_size, device, active_id)
        # process_result(img_lst, img_size, results, classes, colors)

        if running_status == RS_STOP:
            break

    quit_func()


def frame_Main(
    camera: Camera,
    frame_write_queue: Queue,  # main -> frame_main
    command_read_queue: Queue,  # frame_main -> main
    data_queue: Queue,  # frame_main, main -> model_main
    result_queue: Queue,  # model_main -> frame_main
    stream: Stream,
) -> None:
    """
    process for image display
    """

    # start thread which continously reading videocapture
    # sys.stdout = stream
    stream._add_item(os.getpid(), f"FRAME {camera.id}")
    classes, colors = result_queue.get()

    frame_read_queue = TQueue()
    local_command_queue = TQueue()
    # TODO: 可以早点开始
    camera.start_thread(frame_read_queue, local_command_queue)
    frame_write_queue.put((camera.resolution, camera.name))

    model_run = RS_WAITING
    ret_val_main = FV_FRAME_PROC_READY_F
    frame_status = FV_RUNNING
    while True:
        # get frame
        (frame, pkg_loss) = frame_read_queue.get()
        if pkg_loss:
            ret_val_main = FV_PKGLOSS_OCCUR_F
        if model_run == RS_RUNNING:
            tsr = preprocess_img(frame)
            data_queue.put((camera.id, model_run, tsr))
            try:
                (result,) = result_queue.get(WAITING_TIME)
                frame = process_result(frame, result, classes, colors)
            except Empty:
                pass

        shape = frame.shape
        if len(shape) == 3:
            box, bright = camera.frame_config
            x1 = round((box[0] - box[2] / 2) * shape[1])
            y1 = round((box[1] - box[3] / 2) * shape[0])
            x2 = round((box[0] + box[2] / 2) * shape[1])
            y2 = round((box[1] + box[3] / 2) * shape[0])
            # if camera.id == 0: 
                # print(frame.shape, y2-y1, x2-x1)
            frame = frame[y1:y2, x1:x2, :].copy()
        frame_write_queue.put((frame, (ret_val_main, pkg_loss)))
        ret_val_main = FV_FRAME_PROC_READY_F

        # get command
        (frame_status, cmd, cmd_val) = command_read_queue.get()
        if frame_status == FV_RUNNING:
            # get some commands needed by current cycle
            # TODO: set button enable/disable for these functions
            need_switch = cmd == FV_SWITCH_CHANNEL_Q
            need_capture = cmd == FV_CAPTURE_IMAGE_Q
            need_record = cmd == FV_RECORD_VIDEO_Q
            need_refresh = cmd == FV_FLIP_SIMU_STREAM_Q
            need_model = cmd == FV_FLIP_MODEL_ENABLE_Q
            need_pause = cmd == FV_QTHREAD_PAUSE_Q
            need_ctrl = cmd == FV_PTZ_CTRL_Q

            if need_pause:
                camera.viewer.flip_inter_val("need_send")
            elif need_switch:
                # TODO: 挂窗口槽函数
                camera.switch_vid_src(*cmd_val)
            elif need_refresh:
                camera.viewer.flip_inter_val("simu_stream")
            elif need_model:
                model_run = RS_RUNNING if model_run == RS_WAITING else RS_WAITING
                strr = "enabled" if model_run == RS_RUNNING else "disabled"
                print(f"model inference {strr}", end=None)
            elif need_capture:  # TODO: viewer自己把图片存好后翻转
                camera.viewer.flip_inter_val("need_capture")
            elif need_record:
                camera.viewer.flip_inter_val("need_record")
            elif need_ctrl:
                camera.controller.handle_ctrl(cmd_val)

        else:
            camera.viewer.flip_inter_val("need_send")
            camera.viewer.flip_inter_val("is_running")
            break

    for thread in camera.viewer.threads.values():
        if thread.is_alive():
            thread.join()
    camera.controller.is_running = False
    camera.controller._local_command_queue.put((0, 0))
    camera.controller.thread.join()
    sys.stdout = sys.__stdout__


def initialize(file: str, num_cam: int = 6):
    """
    initial all things from config file
    """

    # sys.stdout = gpc_stream
    gpc_stream._add_item(os.getpid(), "MAIN")

    with open(file, "r") as f:
        config = json.load(f)

    # number of cameras, size of data parallel
    default = config["choices"]
    srcs = config["sources"]
    ddp = 1
    camera_lst = [Camera(default[i][0], srcs[default[i][0]][-1], i, num_cam, ddp) for i in range(num_cam)]

    assert num_cam % ddp == 0

    # for data to be inferenced
    data_queues = [Queue() for _ in range(ddp)]
    # for result of inference
    result_queues = [Queue() for _ in range(num_cam)]
    # for frame to be transmitted to main process
    frame_write_queues = [Queue() for _ in range(num_cam)]
    # for receiving command from main process
    command_queues = [Queue() for _ in range(num_cam)]

    # prepare processes
    cam_pool: List[Process] = []
    length = num_cam // ddp
    for ddp_idx in range(ddp):
        cam_pool.append(
            Process(
                target=model_Main,
                args=(
                    length,
                    ddp_idx,
                    data_queues[ddp_idx],
                    result_queues[ddp_idx * length : (ddp_idx + 1) * length],
                    gpc_stream,
                ),
            )
        )

    for cam_idx in range(num_cam):
        cam_pool.append(
            Process(
                target=frame_Main,
                args=(
                    camera_lst[cam_idx],
                    frame_write_queues[cam_idx],
                    command_queues[cam_idx],
                    data_queues[cam_idx // length],
                    result_queues[cam_idx],
                    gpc_stream,
                ),
            )
        )

    for proc in cam_pool:
        proc.start()

    ret = {
        "num_cam": num_cam,
        "data_queues": data_queues,
        "result_queues": result_queues,
        "frame_write_queues": frame_write_queues,
        "command_queues": command_queues,
        "pool": cam_pool,
    }

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arguments for main process")
    parser.add_argument("-n", "--num_cam", type=int, default=1, help="number of cameras to be monitored")
    args = parser.parse_args()

    num_cam = args.num_cam
    gpc = initialize("./configs/video_source_pool.json", num_cam)
    try:
        app = QApplication(sys.argv)
        MainWindow = custom_window(gpc)
        MainWindow.show()
        ret = app.exec()
    except Exception as e:
        print(e)
        MainWindow.close()
        ret = 0
        print("Illegal exit")
    sys.exit(ret)
