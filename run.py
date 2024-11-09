from Qt_ui.mainwin import custom_window
import sys, os, numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
import json
from multiprocessing import Value, Array, Pipe, Process, Pool, Condition as cond
from multiprocessing.synchronize import Condition
from typing import List

from central_monitor.camera_top import Camera
from Qt_ui.utils import gpc_stream
import time, ctypes


# 分两个子进程处理按帧取图和相机控制
def frame_Main(camera: Camera, frame_flag: Condition, frame_pipeObj, frame_val4exec_seq, frame_val4save_pth, frame_buffer, run_flag):
    # sys.stdout = gpc_stream
    out_pipeObj, in_pipeObj = frame_pipeObj
    out_pipeObj.close()
    camera.viewer.start_thread(frame_buffer)
    while True:
        time0 = time.time()
        with frame_flag:
            with frame_val4exec_seq.get_lock():
                need_switch = frame_val4exec_seq.value == 2
                need_capture = frame_val4exec_seq.value == 4
                need_record = frame_val4exec_seq.value == 5
                with frame_val4save_pth.get_lock():
                    capture_pth = frame_val4save_pth.value
                frame_val4exec_seq.value = 1
                with camera.viewer._lock:
                    pkg_loss = camera.viewer.package_loss > 0
                    camera.viewer.package_loss = 0
                if pkg_loss:
                    frame_val4exec_seq.value = 3

            frame_flag.wait()

            with run_flag.get_lock():
                if run_flag.value == 0: break

            
            if need_switch:
                camera.viewer.switch_cam()
            time1 = time.time()
            shape = camera.viewer.fetch_frame(need_capture, need_record, capture_pth.decode())
            time2 = time.time()
            if shape is not None:
                in_pipeObj.send(shape)
            time3 = time.time()
            # if camera.id == 0:
            #     print("send: ", round(time2-time1, 6), round(time3-time2, 6))

    with camera.viewer._lock:
        camera.viewer.run_flag = False
    camera.viewer.thread.join()
    in_pipeObj.close()
    # sys.stdout = sys.__stdout__
    

def ctrl_Main(cameras: List[Camera], ctrl_flag, ctrl_pipeObj, ctrl_val4exec_seq, run_flag):
    # sys.stdout = gpc_stream
    out_pipeObj, in_pipeObj = ctrl_pipeObj
    in_pipeObj.close()
    while True:
        with ctrl_flag:
            with ctrl_val4exec_seq.get_lock():
                ctrl_val4exec_seq.value = 1
            ctrl_flag.wait()

            with run_flag.get_lock():
                if run_flag.value == 0: break

            ctrl = out_pipeObj.recv()
            assert isinstance(ctrl, tuple), "type error"
            assert len(ctrl) == 3, "expecting trible tuple control signal"
            cur_active = int(ctrl[0])
            if cur_active != -1:
                cameras[cur_active].controller.handle_ctrl(ctrl[1:])

    out_pipeObj.close()
    # sys.stdout = sys.__stdout__


def initialize(file: str):
    # sys.stdout = gpc_stream
    with open(file, 'r') as f:
        config = json.load(f)
    num_cam = config["cam_num"]
    resolution = config["resolution"]
    camera_lst = [Camera(config["login"][i], i) for i in range(num_cam)]

    run_flag = Value('i', 1)
    ctrl_flag = cond()
    frame_flag = [cond() for _ in range(num_cam)]
    ctrl_pipe = Pipe()
    frame_pipe_lst = [Pipe() for _ in range(num_cam)]
    ctrl_val4exec_seq = Value('i', 1)
    frame_val4exec_seq = [Value('i', 1) for _ in range(num_cam)]
    frame_val4save_pth = [Array('c', 1024) for _ in range(num_cam)]
    frame_buffer = [Array(ctypes.c_uint8, s[0]*s[1]*3, lock=True) for _, s in zip(range(num_cam), resolution)]

    cam_pool: List[Process] = []
    for i in range(num_cam):
        cam_pool.append(
            Process(            
                target=frame_Main,
                args=(
                    camera_lst[i],
                    frame_flag[i],
                    frame_pipe_lst[i],
                    frame_val4exec_seq[i],
                    frame_val4save_pth[i],
                    frame_buffer[i],
                    run_flag, 
                ),
            )
        )
    
    cam_pool.append(
        Process(target=ctrl_Main, args=(camera_lst, ctrl_flag, ctrl_pipe, ctrl_val4exec_seq, run_flag),)
    )

    # run_flag.acquire(True)
    # for cd in frame_flag: cd.acquire(False)
    for proc in cam_pool: proc.start()
    for pipe in frame_pipe_lst: pipe[1].close()
    ctrl_pipe[0].close()    

    ret = {
        "num_cam": num_cam,
        "run_flag": run_flag,
        "frame_flag": frame_flag,
        "ctrl_flag": ctrl_flag,
        "ctrl_pa_conn": ctrl_pipe[1],
        "ctrl_val4exec_seq": ctrl_val4exec_seq,
        "frame_pa_conn": [frame_pipe_lst[i][0] for i in range(num_cam)],
        "frame_val4exec_seq": frame_val4exec_seq,
        "frame_val4save_pth": frame_val4save_pth,
        "frame_buffer": frame_buffer,
        "num_channel": [len(config["login"][i]) for i in range(num_cam)],
        "resolution": resolution,
        "pool": cam_pool,
    }

    return ret


if __name__ == '__main__':

    gpc = initialize("./configs/camera_template.json")
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = custom_window(gpc)

    MainWindow.show()
    run_flag = gpc["run_flag"]

    sys.exit(app.exec())