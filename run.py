from Qt_ui.mainwin import custom_window
import sys, os, numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets, QtMultimedia
import json
from multiprocessing import Value, Pipe, Process, Pool, Condition as cond
from multiprocessing.synchronize import Condition

from central_monitor.camera_top import Camera
import time


# 分两个子进程处理按帧取图和相机控制
def frame_Main(camera: Camera, frame_flag: Condition, frame_pipeObj):
    out_pipeObj, in_pipeObj = frame_pipeObj
    out_pipeObj.close()
    while True:
        with frame_flag:
            frame_flag.wait()
            time1 = time.time()
            frame = camera.viewer.fetch_frame()
            time2 = time.time()
            if frame is not None:
                in_pipeObj.send(frame)
            time3 = time.time()
            # if camera.id == 0:
                # print("send: ", round(time2-time1, 6), round(time3-time2, 6))

def ctrl_Main(camera: Camera, ctrl_pipeObj):
    out_pipeObj, in_pipeObj = ctrl_pipeObj
    in_pipeObj.close()
    while True:
        # while ctrl_pipeObj.poll():
        ctrl = out_pipeObj.recv()
        assert isinstance(ctrl, tuple), "type error"
        camera.controller.handle_ctrl(ctrl)

def camManager_Main(
        camera: Camera, 
        ctrl_pipeObj,
        frame_flag: Condition,
        frame_pipeObj,
        run_flag: Condition,
    ):

    frame_proc = Process(
        target=frame_Main, 
        args=(camera, frame_flag, frame_pipeObj, ),
    )
    ctrl_proc = Process(
        target=ctrl_Main,
        args=(camera, ctrl_pipeObj, )
    )
    frame_proc.start()
    ctrl_proc.start()

    frame_pipeObj[0].close()
    frame_pipeObj[1].close()
    ctrl_pipeObj[0].close()
    ctrl_pipeObj[1].close()

    with run_flag:
        run_flag.wait()

    frame_proc.terminate()
    ctrl_proc.terminate()



def initialize(file: str):
    with open(file, 'r') as f:
        config = json.load(f)
    num_cam = config["cam_num"]
    camera_lst = [Camera(config["login"][i], i) for i in range(num_cam)]

    run_flag = cond()
    frame_flag = [cond() for _ in range(num_cam)]
    ctrl_lst = [Pipe() for _ in range(num_cam)]
    frame_lst = [Pipe() for _ in range(num_cam)]

    cam_pool = []
    for i in range(num_cam):
        cam_pool.append(
            Process(            
                target=camManager_Main,
                args=(
                    camera_lst[i],
                    ctrl_lst[i],
                    frame_flag[i],
                    frame_lst[i],
                    run_flag, 
                ),
            )
        )
    
    # run_flag.acquire(True)
    # for cd in frame_flag: cd.acquire(False)
    for proc in cam_pool: proc.start()
    for pipe in ctrl_lst: pipe[0].close()
    for pipe in frame_lst: pipe[1].close()
    

    ret = {
        "num_cam": num_cam,
        "run_flag": run_flag,
        "frame_flag": frame_flag,
        "ctrl_pa_conn": [ctrl_lst[i][1] for i in range(num_cam)],
        "frame_pa_conn": [frame_lst[i][0] for i in range(num_cam)],
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
