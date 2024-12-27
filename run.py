import argparse
import json
import os
import sys
import time
from multiprocessing import Process, Queue
from queue import Empty
from queue import Queue as TQueue
from typing import List

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
    FV_UPDATE_VID_INFO_F,
    RS_RUNNING,
    RS_STOP,
    RS_WAITING,
    Stream,
    gpc_stream,
    safe_get,
)
from running_models.extern_model import (
    initialize_model_engine,
    preprocess_img,
    process_result,
    YOLOV11_TRACK,
    YOLOV3_DETECT
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
) -> None:
    """
    process for model inference
    """

    def quit_func():
        sys.stdout = sys.__stdout__

    # sys.stdout = stream
    stream._add_item(os.getpid(), f"MODEL {ddp_id}")

    # model loading and cuda memory pre-allocating
    batch = min(local_num_cam, 4)
    running_status = RS_WAITING
    model, classes, colors = initialize_model_engine(local_num_cam, inference_device, model_config)
    for queue in result_queue:
        queue.put((classes, colors))
    model._hot_init()

    while True:
        # if model inference is needed, data to be inferenced will be in data_queue
        # rank-0 send start and end signal
        fetch, tsr_lst, sender_lst, model_type_lst = 0, [], [], []

        while fetch < batch:
            if running_status == RS_WAITING:
                new_data = data_queue.get()
                (vid_id, running_status, _, data_model_request) = new_data

            if running_status == RS_WAITING:
                continue
            elif running_status == RS_STOP:
                break
            else:
                try:
                    new_data = data_queue.get(timeout=WAITING_TIME)
                    (vid_id, running_status, tsr, data_model_request) = new_data

                    if running_status == RS_RUNNING:
                        if tsr is not None:
                            tsr_lst.append(tsr)
                            sender_lst.append(vid_id)
                            model_type_lst.append(data_model_request)
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
            # all are v3, v11
            if set(model_type_lst) == 1:
                model.set_model(model_type_lst[0])
                if model_type_lst[0] == YOLOV11_TRACK:
                    chunk_tsr = tsr_lst
                else:
                    chunk_tsr = torch.stack(tsr_lst, dim=0)
                    chunk_tsr = chunk_tsr.to(device=inference_device)
                results = model(chunk_tsr, augment=False)
            # process each one respectively for type discrepancy
            else:
                results = []
                for i in range(fetch):
                    model.set_model(model_type_lst[i])
                    if model_type_lst[i] == YOLOV3_DETECT:
                        chunk_tsr = tsr_lst[i].unsqueeze(0).to(device=inference_device)
                    else:
                        chunk_tsr = tsr_lst[i]
                    result = model(chunk_tsr, augment=False)[0]
                    results.append(result)

        # dispatch results to corresponding process
        for i in range(fetch):
            result_queue[sender_lst[i]].put((results[i],))

        # process image and execute inference
        # preprocess_img(chunk_tsr, img_lst, img_size, device, active_id)
        # process_result(img_lst, img_size, results, classes, colors)

        if running_status == RS_STOP:
            break

    quit_func()


def frame_Main(
    camera: Camera,
    inference_device: str,
    model_config: dict,
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
    camera.start_thread(frame_read_queue, local_command_queue)
    frame_write_queue.put((camera.resolution, camera.name))

    model_type = "None"

    model_run = RS_WAITING
    ret_status = FV_FRAME_PROC_READY_F
    ret_val_main = None
    frame_status = FV_RUNNING

    temp_resolution = None
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
            tsr = preprocess_img(frame, model_config, model_type, device=inference_device)
            data_queue.put((camera.id, model_run, tsr, model_type))
            try:
                (result,) = result_queue.get(timeout=WAITING_TIME*100) # Large but not blocking in case of deadlock
                frame = process_result(frame, result, classes, colors, model_config, model_type)
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
        frame_write_queue.put((frame, (ret_status, ret_val_main)))

        # get command
        (frame_status, cmd, cmd_val) = command_read_queue.get()
        if frame_status == FV_RUNNING:
            # get some commands needed by current cycle
            # TODO: set button enable/disable for these functions
            need_switch  = cmd == FV_SWITCH_CHANNEL_Q
            need_capture = cmd == FV_CAPTURE_IMAGE_Q
            need_record  = cmd == FV_RECORD_VIDEO_Q
            need_refresh = cmd == FV_FLIP_SIMU_STREAM_Q
            need_model   = cmd == FV_FLIP_MODEL_ENABLE_Q
            need_pause   = cmd == FV_QTHREAD_PAUSE_Q
            need_ctrl    = cmd == FV_PTZ_CTRL_Q

            if need_pause:
                camera.viewer.flip_inter_val("need_send")
            elif need_switch:
                temp_resolution = camera.switch_vid_src(*cmd_val)
            elif need_refresh:
                camera.viewer.flip_inter_val("simu_stream")
            elif need_model:
                model_type = cmd_val
                model_run  = RS_WAITING if model_type == "None" else RS_RUNNING
                strr = "enabled" if model_run == RS_RUNNING else "disabled"
                print(f"model inference {strr} with model {model_type}", end=None)
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


def initialize(file: str, num_cam: int, model_config: str):
    """
    initial all things from config file
    """

    # sys.stdout = gpc_stream
    gpc_stream._add_item(os.getpid(), "MAIN")

    with open(file, "r") as f:
        config = json.load(f)

    with open(model_config, "r") as f:
        model_config_dict = json.load(f)

    # number of cameras, size of data parallel
    default = config["choices"]
    srcs = config["sources"]
    ddp = 1
    camera_lst = [Camera(default[i][0], srcs[default[i][0]][-1], i, num_cam, ddp) for i in range(num_cam)]
    default = [[d[0], [dicts["NICKNAME"] for dicts in srcs[d[0]]].index(d[1])] for d in default][:num_cam]

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
                ),
            )
        )

    for cam_idx in range(num_cam):
        cam_pool.append(
            Process(
                target=frame_Main,
                args=(
                    camera_lst[cam_idx],
                    inference_device,
                    model_config_dict,
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
        "model_config": model_config_dict,
        "pool": cam_pool,
        "current_chosen_video_source": default,
        "video_source_info_lst": srcs,
    }

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arguments for main process")
    parser.add_argument("-n", "--num_cam", type=int, default=1, help="number of cameras to be monitored")
    parser.add_argument("-v", "--video_config",  type=str, default=os.path.join("configs","video_source_pool.json"), help="Path to the json file of video source pool")
    parser.add_argument("-m", "--model_config",  type=str, default=os.path.join("configs","model_config.json"), help="Path to the json file of model config")
    args = parser.parse_args()

    num_cam      = args.num_cam
    model_config = args.model_config
    video_config = args.video_config
    gpc = initialize(video_config, num_cam, model_config)
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
