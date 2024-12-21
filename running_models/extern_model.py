import random
from typing import List

import cv2
import numpy as np
import torch

from detector.advanced.counter import ObjectCounter
from running_models.yolov3_utils import (
    letterbox,
    load_classes,
    non_max_suppression,
    plot_one_box,
    scale_coords,
)

YOLOV3_DETECT = "yolov3-detect"
YOLOV3_DETECT_PROP = {
    "name":         YOLOV3_DETECT,
    "img_size":     512,
    "weight":       "running_models/yolov3/weights/110-0.447.pt",
    "config":       "./running_models/yolov3/cfg/yolov3-t.cfg"
}

YOLOV11_TRACK = "YOLOV11_TRACK"
YOLOV11_TRACK_PROP = { 
    "name":         YOLOV11_TRACK,
    "img_size":     1920,
    "weight":       "yolo11n.pt",
}


MODEL_LIST = {
    YOLOV3_DETECT: YOLOV3_DETECT_PROP,
    YOLOV11_TRACK: YOLOV11_TRACK_PROP
}

def get_model_name_list():
    return list(MODEL_LIST.keys())


class Engine:
    def __init__(self, classes, inference_device, num_cam, log_file):
        self.classes    = classes
        self.model      = None
        self.device     = inference_device
        self.type       = "None"
        self.num_cam    = num_cam
        self.log_file   = log_file


    def __call__(self, chunk_tsr, model_type, **kwargs):
        if self.type != model_type:
            # Switch model 
            if model_type == YOLOV3_DETECT:
                from running_models.yolov3.models import Darknet

                weight   = MODEL_LIST[model_type]['weight']
                config   = MODEL_LIST[model_type]['config']
                img_size = MODEL_LIST[model_type]['img_size']

                model = Darknet(config, img_size)
                torch_model = torch.load(weight, map_location=self.device)
                try:    
                    model.load_state_dict(torch_model["model"])
                except:
                    print("Warning: failed load model, try again using relaxed mode")
                    model.load_state_dict(torch_model["model"], strict=False)
                model.to(self.device).eval()
            elif model_type == YOLOV11_TRACK:
                weight   = MODEL_LIST[model_type]['weight']
                img_size = MODEL_LIST[model_type]['img_size']
                
                model = ObjectCounter(
                    device=self.device,
                    weights=weight,
                    classes_of_interest=self.classes,
                    log_file=self.log_file,
                )
            elif model_type == "None":
                model = None     
            else:
                raise RuntimeError(f"Unknown model type: {model_type}")

            self.type  = model_type
            self.model = model

        results = None
        if self.type == YOLOV3_DETECT:
            pred = self.model(chunk_tsr, **kwargs)[0]
            pred = pred.clone().detach().cpu()
            results = non_max_suppression(pred, conf_thres=0.2, iou_thres=0.4, multi_label=False)
        elif self.type == YOLOV11_TRACK:
            # inputs = np.array([input.numpy() for input in args]).squeeze(1)
            for input in chunk_tsr:
                pred = self.model.online_predict(input.squeeze(0).numpy())
            results = [None] * len(chunk_tsr)
        else:
            results = [None] * len(chunk_tsr)

        return results

    def set_model(self, model, device, type):
        self.model = model
        self.device = device
        self.type = type


def initialize_model_engine(
    local_num_cam: int,
    inference_device: str   = "cpu",
    data_class: str         = "configs/names.txt",
    log_file: str           = "logs/tracking_log.txt",
    seedid: int             = 1024,
):
    random.seed(seedid)
    classes = load_classes(data_class)
    colors  = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    engine  = Engine(classes, inference_device, local_num_cam, log_file)
    return engine, classes, colors


def preprocess_img(ori_img: np.ndarray, type: str, device: str):
    img_size = MODEL_LIST[type]['img_size']
    inf_shape = (int(img_size * 9 / 16), img_size)

    img = None
    if type == YOLOV3_DETECT:
        img = letterbox(ori_img.copy(), new_shape=inf_shape)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)

        img = img.float() / 255.0
    else:
        img = torch.from_numpy(ori_img).to(device)

    return img


def process_result(ori_img: np.ndarray, det, classes, colors, type):
    img_size = MODEL_LIST[type]['img_size']
    inf_shape = (int(img_size * 9 / 16), img_size)

    # process detections
    if type == YOLOV3_DETECT:
        if det is not None and len(det):
            # if i == 0: print(f"DETECT: total {len(det)} boxes")
            det[:, :4] = scale_coords(inf_shape, det[:, :4], ori_img.shape).round()

            for *xyxy, conf, cls in reversed(det):
                label = "%s %.2f" % (classes[int(cls)], conf)
                plot_one_box(xyxy, ori_img, label=label, color=colors[int(cls)])

    return ori_img
