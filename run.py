import argparse
import json
import os
import sys

sys.path.append(os.path.join("yolov3"))
import datetime
import time
from multiprocessing import Process, Queue, current_process
from queue import Empty
from queue import Queue as TQueue
from typing import List

import cv2
import numpy as np
import qdarkstyle
import setproctitle
import torch
from PyQt6.QtWidgets import QApplication
from qdarkstyle.light.palette import LightPalette

from src.central_monitor.camera_top import Camera
from src.Qt_ui.mainwin import custom_window
from src.Qt_ui.utils import (
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
    FV_UPDATE_VID_INFO_F,
    MODEL_JSON_PATH,
    RS_RUNNING,
    RS_STOP,
    RS_WAITING,
    VIDEO_SOURCE_POOL_PATH,
    Stream,
    gpc_stream,
    safe_get,
)
from src.running_models.extern_model import (
    YOLOV3_DETECT,
    YOLOV11_TRACK,
    initialize_model_engine,
    preprocess_img,
    process_result,
)

WAITING_TIME = 0.01

# TODO: so many todo...
def model_Main(
    local_num_cam: int,
    ddp_id: int,
    inference_device: str,
    model_config: dict,
    data_queue: Queue,
    result_queue: List[Queue],
    stream: Stream,
    debug: bool,
    curtime: datetime.datetime,
) -> None:
    """
    process for model inference
    """

    def quit_func():
        name = current_process().name
        print(f"{name} normally quit: Model Main Process {ddp_id}")
        sys.stdout = sys.__stdout__

    setproctitle.setproctitle(f"MODEL {ddp_id}")
    if not debug:
        sys.stdout = stream
    stream._add_item(os.getpid(), f"MODEL {ddp_id}")

    # model loading and cuda memory pre-allocating
    batch = min(local_num_cam, 1)
    running_status = RS_WAITING
    log_file = os.path.join("data", curtime.strftime("%Y-%m-%d_%H-%M-%S"), "advanced_yolo.log")
    model, classes, colors, traj_colors, track_history = initialize_model_engine(
        local_num_cam, inference_device, model_config, log_file
    )
    for queue in result_queue:
        queue.put((classes, colors, traj_colors, track_history))
    model._hot_init()

    while True:
        # if model inference is needed, data to be inferenced will be in data_queue
        # rank-0 send start and end signal
        fetch, tsr_lst, sender_lst = 0, [], []

        while fetch < batch:
            # here, only 'None' tsr will activate model inference
            if running_status == RS_WAITING:
                new_data = data_queue.get()
                # hyper params include: (data_model_request, polygens, img_size, selected_class, conf_thre, iou_thre)
                (vid_id, running_status, _, hyper_params_dict) = new_data
                if running_status == RS_RUNNING and len(hyper_params_dict) > 0:
                    model.set_model(**hyper_params_dict)

            if running_status == RS_WAITING:
                continue
            elif running_status == RS_STOP:
                break
            else:
                try:
                    # data stuck not exist here
                    new_data = data_queue.get(timeout=WAITING_TIME)
                    (vid_id, running_status, tsr, hyper_params_dict) = new_data

                    if running_status == RS_RUNNING:
                        if tsr is not None:
                            tsr_lst.append(tsr)
                            sender_lst.append(vid_id)
                            fetch += 1
                        else:
                            # 'None' tsr represents model setting
                            model.set_model(**hyper_params_dict)
                            break  # avoid model discrepancy in one batch
                    else:
                        size = data_queue.qsize()
                        for _ in range(size):
                            data_queue.get()
                        break
                except Empty:
                    break

        # no matter which frame submit data, inference will be executed when enough data is collected
        # TODO: this part is MEMORY intensive
        results = []
        if fetch > 0:
            if model.type == YOLOV11_TRACK:
                chunk_tsr = [(ii, tt) for ii, tt in zip(sender_lst, tsr_lst)]
            else:
                chunk_tsr = torch.stack(tsr_lst, dim=0)
                # chunk_tsr = chunk_tsr.to(device=inference_device)
            results = model(chunk_tsr, augment=False)

        # dispatch results to corresponding process
        for i in range(fetch):
            result_queue[sender_lst[i]].put((results[i],))

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
    debug: bool,
) -> None:
    """
    process for image display
    """

    setproctitle.setproctitle(f"FRAME {camera.id}")
    # start thread which continously reading videocapture
    if not debug:
        sys.stdout = stream
    stream._add_item(os.getpid(), f"FRAME {camera.id}")
    classes, colors, traj_colors, track_history = result_queue.get()

    frame_read_queue = TQueue()
    local_command_queue = TQueue()
    camera.start_thread(frame_read_queue, local_command_queue)
    frame_write_queue.put((camera.resolution, camera.name))

    model_type = "None"
    img_size = None
    selected_class = None
    is_active = None

    model_run = RS_WAITING
    ret_status = FV_FRAME_PROC_READY_F
    ret_val_main = None
    frame_status = FV_RUNNING

    temp_resolution = None
    cnt_pkt = (None, None)
    model_inference_cost = 0.0

    # here polygons only include polys of current camera
    polygons = []
    while True:
        # get frame
        (frame, pkg_loss) = frame_read_queue.get()

        # update value that: frame_main -> main
        if temp_resolution is not None:
            ret_val_main = temp_resolution
            ret_status = FV_UPDATE_VID_INFO_F
            temp_resolution = None
        elif pkg_loss:
            ret_val_main = pkg_loss
            ret_status = FV_PKGLOSS_OCCUR_F
        else:
            ret_val_main = None
            ret_status = FV_FRAME_PROC_READY_F

        if model_run == RS_RUNNING:
            inference_start_time = time.time()
            tsr, new_shape = preprocess_img(frame, img_size, model_type)
            data_queue.put((camera.id, model_run, tsr, {}))
            try:
                (result,) = result_queue.get(timeout=WAITING_TIME * 100)  # Large but not blocking in case of deadlock
                frame, cnt_pkt = process_result(
                    frame,
                    result,
                    classes,
                    colors,
                    new_shape,
                    model_type,
                    selected_class,
                    traj_colors,
                    track_history,
                    polygons,
                )
                model_inference_cost = time.time() - inference_start_time
            except Empty:
                cnt_pkt = (None, None)
                model_inference_cost = 0.0

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
        frame_write_queue.put(((frame, cnt_pkt, model_inference_cost), (ret_status, ret_val_main)))

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
                temp_resolution = camera.switch_vid_src(*cmd_val)
            elif need_refresh:
                camera.viewer.flip_inter_val("simu_stream")
            elif need_model:
                model_type, is_active, polygons, img_size, selected_class = cmd_val
                model_run = RS_WAITING if not is_active else RS_RUNNING
                strr = "enabled" if model_run == RS_RUNNING else "disabled"
                print(f"model inference {strr} with model {model_type}, input size {img_size}", end=None)
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

    name = current_process().name
    print(f"{name} normally quit: Frame Main Process {camera.id}")
    sys.stdout = sys.__stdout__


def initialize(file: str, num_cam: int, model_config: str, log_file: bool, debug: bool, curtime: datetime.datetime):
    """
    initial all things from config file
    """

    if not debug:
        sys.stdout = gpc_stream
    gpc_stream._add_item(os.getpid(), "MAIN")

    with open(file, "r") as f:
        config = json.load(f)

    with open(model_config, "r") as f:
        model_config_dict = json.load(f)

    # number of cameras, size of data parallel
    default: list = config["choices"]
    srcs: dict = config["sources"]
    ddp = 1

    # if in last invoke, camera number diffenent, then calibrate
    if num_cam > len(default):
        default.extend([default[0] for _ in range(num_cam - len(default))])
    else:
        default = default[:num_cam]
    default = [[d[0], [dicts["NICKNAME"] for dicts in srcs[d[0]]].index(d[1])] for d in default]
    camera_lst = [Camera(default[i][0], srcs[default[i][0]][default[i][1]], i, num_cam, ddp) for i in range(num_cam)]

    assert num_cam % ddp == 0

    # for data to be inferenced
    data_queues = [Queue() for _ in range(ddp)]
    # for result of inference
    result_queues = [Queue() for _ in range(num_cam)]
    # for frame to be transmitted to main process
    frame_write_queues = [Queue() for _ in range(num_cam)]
    # for receiving command from main process
    command_queues = [Queue() for _ in range(num_cam)]
    # The device model inference executes on
    inference_device = "cuda:0" if torch.cuda.is_available() else "cpu"

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
                    inference_device,
                    model_config_dict,
                    data_queues[ddp_idx],
                    result_queues[ddp_idx * length : (ddp_idx + 1) * length],
                    gpc_stream,
                    debug,
                    curtime,
                ),
                name=f"model_{ddp_idx}",
                daemon=True,
            )
        )

    for cam_idx in range(num_cam):
        process = Process()
        process.name
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
                    debug,
                ),
                name=f"frame_{cam_idx}",
                daemon=True,
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
        "model_config": model_config_dict,
        "pool": cam_pool,
        "current_chosen_video_source": default,
        "video_source_info_lst": srcs,
        "log": log_file,
    }

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arguments for main process")
    parser.add_argument("-n", "--num_cam", type=int, default=1, help="number of cameras to be monitored")
    parser.add_argument(
        "-v",
        "--video_config",
        type=str,
        default=VIDEO_SOURCE_POOL_PATH,
        help="Path to the json file of video source pool",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, default=MODEL_JSON_PATH, help="Path to the json file of model config"
    )
    parser.add_argument("-l", "--log_file", default=True, action="store_false", help="enable logging")
    parser.add_argument("-d", "--debug", default=False, action="store_true", help="debug mode")
    args = parser.parse_args()

    num_cam = args.num_cam
    model_config = args.model_config
    video_config = args.video_config
    log_file = args.log_file
    debug = args.debug
    curtime = datetime.datetime.now()
    gpc = initialize(video_config, num_cam, model_config, log_file, debug, curtime)
    try:
        app = QApplication(sys.argv)
        MainWindow = custom_window(gpc, curtime)
        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyqt6", palette=LightPalette()))
        MainWindow.show()
        ret = app.exec()
    except Exception as e:
        ret = 1
        sys.stdout = sys.__stdout__
        pool = gpc["pool"]
        for proc in pool:
            proc.terminate()
        print(f"Illegal exit: {e}")
    sys.exit(ret)
