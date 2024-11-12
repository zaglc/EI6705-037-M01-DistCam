import time, ctypes, json
import sys, numpy as np, os
from PyQt6.QtWidgets import QApplication
from multiprocessing import (
    Value, Array, Pipe, Process, Condition as cond
)
from multiprocessing.synchronize import Condition
from multiprocessing.connection import Connection
from multiprocessing.sharedctypes import (
    SynchronizedBase, SynchronizedArray,
)
from typing import List

from central_monitor.camera_top import Camera
from Qt_ui.mainwin import custom_window
from Qt_ui.utils import (
    gpc_stream, Stream, sync_processes, count_nonactive_proc
)
from Qt_ui.utils import (
    FV_FRAME_PROC_READY_F, 
    FV_SWITCH_CHANNEL_Q, FV_PKGLOSS_OCCUR_F, 
    FV_CAPTURE_IMAGE_Q, FV_RECORD_VIDEO_Q, 
    FV_FLIP_SIMU_STREAM_Q, FV_FLIP_MODEL_ENABLE_Q, 
    FV_QTHREAD_PAUSE_Q,
)
from running_models.extern_model import (
    initialize_model, process_result, preprocess_img,
)


# TODO: so many todo...
def model_Main(
        local_num_cam: int, 
        ddp_id: int, 
        frame_buffer: SynchronizedArray, 
        frame_buffer_out: SynchronizedArray, 
        frame_shape_buffer: SynchronizedArray, 
        frame_shape_cnt: SynchronizedBase, 
        inference_fg: Condition, 
        inference_fg2: Condition, 
        run_flag: Condition,
        stream: Stream,
    ) -> None:
    """
    process for model inference
    """
    
    # sys.stdout = stream
    # stream._add_item(os.getpid(), f"MODEL {ddp_id}")
    inference_fg2.acquire(True)

    # model loading and cuda memory pre-allocating
    model, img_size, device, classes, colors, chunk_tsr = initialize_model(local_num_cam)
    
    while True:
        # if model inference is needed, this process will be awaked
        inference_fg2.wait()
        with run_flag.get_lock():
            if run_flag.value == 0: 
                inference_fg2.release()
                break

        # read memory of images
        img_lst: List[np.ndarray] = []
        active_id = 0
        time_1 = time.time()
        for i in range(local_num_cam):
            with frame_buffer[i].get_lock():
                img_buf = np.frombuffer(
                    frame_buffer[i].get_obj(),
                    dtype=ctypes.c_uint8
                )
                active = frame_shape_buffer[4*i+3] == 0
                if not active: continue
                active_id += 1

                shape = (frame_shape_buffer[4*i],
                         frame_shape_buffer[4*i+1],
                         frame_shape_buffer[4*i+2])
                nbytes = shape[0] * shape[1] * shape[2]

                img_npy = img_buf[:nbytes].copy().reshape(shape)
                img_lst.append(img_npy)

        # process image and execute inference
        time0 = time.time()
        preprocess_img(chunk_tsr, img_lst, img_size, device, active_id)
        time1 = time.time()
        results = model(chunk_tsr[:active_id], augment=False)[0]
        results = results.clone().detach().cpu()
        time2 = time.time()
        process_result(img_lst, img_size, results, classes, colors)
        time3 = time.time()

        # save memory for images with bbx  
        for i in range(local_num_cam):
            with frame_buffer_out[i].get_lock():
                img_buf = np.frombuffer(
                    frame_buffer_out[i].get_obj(),
                    dtype=ctypes.c_uint8
                )
                shape = frame_shape_buffer[4*i: 4*i+3]
                nbytes = shape[0] * shape[1] * shape[2]
                img_buf[:nbytes] = img_lst[i].reshape(-1)
        
        # consumer: reset semaphore
        with frame_shape_cnt.get_lock():
            frame_shape_cnt.value = 0
        time4 = time.time()

        with inference_fg:
            inference_fg.notify_all()
        time5 = time.time()

        # print(f"read mem: {round(time0-time_1, 4)}, prepare: {round(time1-time0, 4)}, infer: {round(time2-time1, 4)}, post: {round(time3-time2, 4)}, write mem: {round(time4-time3, 4)}, notify: {round(time5-time4, 4)}, total: {round(time4-time_1, 4)}")
    sys.stdout = sys.__stdout__


def frame_Main(
        camera: Camera, 
        frame_flag: Condition, 
        frame_pipeObj: Connection, 
        frame_cmd: SynchronizedBase, 
        frame_buffer: SynchronizedArray, 
        frame_buffer_out: SynchronizedArray, 
        frame_shape_buffer: SynchronizedArray, 
        frame_shape_cnt: SynchronizedBase, 
        inference_fg: Condition, 
        inference_fg2: Condition, 
        run_flag: Condition,
        stream: Stream,
    ) -> None:
    """
    process for image display
    """

    # start thread which continously reading videocapture
    # sys.stdout = stream
    # stream._add_item(os.getpid(), F"FRAME {camera.id}")

    out_pipeObj, in_pipeObj = frame_pipeObj
    out_pipeObj.close()
    camera.viewer.start_thread(frame_buffer, frame_buffer_out)
    model_run = False
    active_id = camera.local_size()
    while True:
        with frame_flag:
            # get some commands needed by current cycle
            # TODO: set button enable/disable for these functions
            with frame_cmd.get_lock():
                need_switch = frame_cmd.value == FV_SWITCH_CHANNEL_Q
                need_capture = frame_cmd.value == FV_CAPTURE_IMAGE_Q
                need_record = frame_cmd.value == FV_RECORD_VIDEO_Q
                need_refresh = frame_cmd.value == FV_FLIP_SIMU_STREAM_Q
                need_model = frame_cmd.value == FV_FLIP_MODEL_ENABLE_Q
                need_pause = frame_cmd.value == FV_QTHREAD_PAUSE_Q
                frame_cmd.value = FV_FRAME_PROC_READY_F
                with camera.viewer._lock:
                    pkg_loss = camera.viewer.package_loss > 0
                    camera.viewer.package_loss = 0
                if pkg_loss:
                    frame_cmd.value = FV_PKGLOSS_OCCUR_F

            if need_pause:
                camera.viewer.flip_inter_val("need_read")
            # TODO: useful if we need to arbitrarily pause frame window
            #     frame_shape_buffer[camera.local_id()*4+3] = 1
            # sync_processes(inference_fg, frame_shape_cnt, active_id)
            # active_id = count_nonactive_proc(frame_shape_buffer, camera.local_size())

            frame_flag.wait()
            with run_flag.get_lock():
                if run_flag.value == 0: break

            if need_pause:
                camera.viewer.flip_inter_val("need_read")
            #     active_id = camera.local_size()
            #     frame_shape_buffer[camera.local_id()*4+3] = 0
            # sync_processes(inference_fg, frame_shape_cnt, active_id)
            # active_id = count_nonactive_proc(frame_shape_buffer, camera.local_size())

            # command for switch channel, flip simo_stream, enable model inference
            if need_switch:
                camera.viewer.switch_cam()
            elif need_refresh:
                camera.viewer.flip_inter_val("simu_stream")
            elif need_model:
                model_run = not model_run
                strr = "enabled" if model_run else "disabled"
                print(f"model inference {strr}",end=None)
            
            # fetch newest frame and save it in share_mem buffer: frame buffer
            shape = camera.viewer.fetch_frame(need_capture, need_record)

            if model_run:
                # awake model process
                frame_shape_buffer[camera.local_id()*4] = shape[0]
                frame_shape_buffer[camera.local_id()*4+1] = shape[1]
                frame_shape_buffer[camera.local_id()*4+2] = shape[2]

                # producer: add 1 when enter wait, the last one notify model main
                frame_shape_cnt.acquire(True)
                frame_shape_cnt.value += 1
                with inference_fg:
                    if frame_shape_cnt.value == camera.local_size():
                        with inference_fg2:
                            inference_fg2.notify_all()
                    frame_shape_cnt.release()
                    inference_fg.wait()
            else:
                # when no model running
                # moving data from in_buffer to out_buffer
                nbytes = shape[0] * shape[1] * shape[2]
                with frame_buffer.get_lock(), frame_buffer_out.get_lock():
                    frame_in = np.frombuffer(
                        frame_buffer.get_obj(),
                        dtype=ctypes.c_uint8,
                    )
                    frame_out = np.frombuffer(
                        frame_buffer_out.get_obj(),
                        dtype=ctypes.c_uint8,
                    )
                    frame_out[:nbytes] = frame_in[:nbytes]

            with run_flag.get_lock():
                if run_flag.value == 0: break

            # send shape to trigger new image display
            if shape is not None:
                in_pipeObj.send(shape)

    with camera.viewer._lock:
        camera.viewer.run_flag = False
    for _, thread in enumerate(camera.viewer.threads):
        if thread.is_alive(): 
            thread.join()
    in_pipeObj.close()
    sys.stdout = sys.__stdout__
    

def ctrl_Main(
        cameras: List[Camera], 
        ctrl_flag: Condition,
        ctrl_pipeObj: Connection,
        ctrl_val4exec_seq: SynchronizedBase,
        run_flag: Condition,
        stream: Stream,
    ) -> None:
    
    """
    ctrl process sending PTZ ctrl cmds
    """

    # sys.stdout = stream
    # stream._add_item(os.getpid(), "CTRL 0")

    out_pipeObj, in_pipeObj = ctrl_pipeObj
    in_pipeObj.close()
    while True:
        with ctrl_flag:
            # TODO: may not need...
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
    sys.stdout = sys.__stdout__


def initialize(file: str):
    """
    initial all things from config file
    """

    sys.stdout = gpc_stream
    gpc_stream._add_item(os.getpid(), "MAIN")

    with open(file, 'r') as f:
        config = json.load(f)
    
    # number of cameras, resolution of each cameras, size of data parallel
    num_cam = config["cam_num"]
    resolution = config["resolution"]
    ddp = config["data_parallel"]
    camera_lst = [Camera(config["login"][i], i, num_cam, ddp) for i in range(num_cam)]
    
    assert num_cam % ddp == 0

    run_flag = Value('i', 1)
    
    ctrl_flag = cond()
    ctrl_pipe = Pipe()
    ctrl_val4exec_seq = Value('i', 1)

    # frame main:
    # sync condition, pipe sending shape, share mem indicating command
    # share mem indicating img save path, input/ouput frame buffer for each camera
    frame_flag = [cond() for _ in range(num_cam)]
    frame_pipe_lst = [Pipe() for _ in range(num_cam)]
    frame_val4exec_seq = [Value('i', 1) for _ in range(num_cam)]
    frame_buffer = [Array(ctypes.c_uint8, s[0][0]*s[0][1]*3*2) for _, s in zip(range(num_cam), resolution)]
    frame_buffer_out = [Array(ctypes.c_uint8, s[0][0]*s[0][1]*3*2) for _, s in zip(range(num_cam), resolution)]
    
    # model main:
    # condition for frame main to wait and model main to notify, and inverse
    # share mem indicating frame shape, number of current awaiting frame process
    inference_fg = [cond() for _ in range(ddp)]
    inference_fg2 = [cond() for _ in range(ddp)]
    frame_shape_buffer = [Array('i', 4 * (num_cam//ddp)) for _ in range(ddp)]
    frame_shape_cnt = [Value('i', 0) for _ in range(ddp)]

    # prepare processes
    cam_pool: List[Process] = []
    length = num_cam//ddp
    for i in range(ddp):
        cam_pool.append(
            Process(
                target=model_Main, 
                args=(
                    length, 
                    i, 
                    frame_buffer[i*length:(i+1)*length], 
                    frame_buffer_out[i*length:(i+1)*length], 
                    frame_shape_buffer[i], 
                    frame_shape_cnt[i], 
                    inference_fg[i], 
                    inference_fg2[i], 
                    run_flag,
                    gpc_stream,
                ),
            )
        )

    for i in range(num_cam):
        cam_pool.append(
            Process(            
                target=frame_Main,
                args=(
                    camera_lst[i],
                    frame_flag[i],
                    frame_pipe_lst[i],
                    frame_val4exec_seq[i],
                    frame_buffer[i],
                    frame_buffer_out[i],
                    frame_shape_buffer[i//length],
                    frame_shape_cnt[i//length],
                    inference_fg[i//length],
                    inference_fg2[i//length],
                    run_flag,
                    gpc_stream,
                ),
            )
        )
    
    cam_pool.append(
        Process(
            target=ctrl_Main, 
            args=(
                camera_lst, 
                ctrl_flag, 
                ctrl_pipe, 
                ctrl_val4exec_seq, 
                run_flag,
                gpc_stream,
            ),
        )
    )

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
        "frame_buffer": frame_buffer_out,
        "inference_fg": inference_fg,
        "inference_fg2": inference_fg2,
        "num_channel": [len(config["login"][i]) for i in range(num_cam)],
        "resolution": resolution,
        "pool": cam_pool,
    }

    return ret


if __name__ == '__main__':
    gpc = initialize("./configs/camera_template.json")
    app = QApplication(sys.argv)
    MainWindow = custom_window(gpc)
    MainWindow.show()
    sys.exit(app.exec())