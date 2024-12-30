import random

import numpy as np
import torch

from running_models.yolov3.models import Darknet
from detector.advanced.counter import ObjectCounter
from running_models.yolov3_utils import (
    letterbox,
    load_classes,
    non_max_suppression,
    plot_one_box,
    scale_coords,
)

YOLOV3_DETECT = "yolov3-detect"
YOLOV11_TRACK = "yolov11-track"
__model_choices__ = [YOLOV3_DETECT, YOLOV11_TRACK]

class Engine:
    def __init__(self, classes, inference_device, model_config, num_cam, log_file):
        """
        general engine for model inference
        """

        self.classes = classes
        self.model = None
        self.device = inference_device
        self.model_config = model_config
        self.type = "None"
        self.num_cam = num_cam
        self.log_file = log_file
        self.model_pool = {}

    def _hot_init(self):
        """
        pre-initiate model pool
        store them in cuda memory, or at list in cpu memory
        """

        for model_type in __model_choices__:
            if model_type == YOLOV3_DETECT:
                weight = self.model_config[model_type]['weight']
                config = self.model_config[model_type]['config']
                img_size = self.model_config[model_type]['img_size']

                model = Darknet(config, img_size)
                torch_model = torch.load(weight, map_location=self.device)
                model.load_state_dict(torch_model["model"], strict=False)
                # print("Warning: failed load model, try again using relaxed mode")
                model.to(self.device).eval()
                # with torch.no_grad():
                #     model(torch.zeros(1, 3, img_size, img_size).to(self.device), augment=False)
            elif model_type == YOLOV11_TRACK:
                weight = self.model_config[model_type]['weight']
                img_size = self.model_config[model_type]['img_size']
                
                model = ObjectCounter(
                    device=self.device,
                    weights=weight,
                    classes_of_interest=self.classes,
                    log_file=self.log_file,
                )
            self.model_pool[model_type] = model

    def __call__(self, chunk_tsr, **kwargs):
        """
        engine runtime call
        """

        self.model = self.model_pool[self.type]
        results = None
        if self.type == YOLOV3_DETECT:
            pred = self.model(chunk_tsr.to(self.device), **kwargs)[0]
            pred = pred.clone().detach().cpu()
            results = non_max_suppression(pred, conf_thres=0.2, iou_thres=0.4, multi_label=False)
        elif self.type == YOLOV11_TRACK:
            # inputs = np.array([input.numpy() for input in args]).squeeze(1)
            for input in chunk_tsr:
                pred = self.model.online_predict(input)
            results = [None] * len(chunk_tsr)
        else:
            results = [None] * len(chunk_tsr)

        return results

    def set_model(self, type):
        if type != self.type:
            self.type = type
            self.model = self.model_pool[type]
            print(f"model changed to {type}")


def initialize_model_engine(
    local_num_cam: int,
    inference_device: str   = "cpu",
    model_config: str       = "running_models/model_config.json",
    log_file: str           = "logs/tracking_log.txt",
    seedid: int             = 1024,
):
    random.seed(seedid)
    classes = load_classes(model_config[YOLOV3_DETECT]["names"])
    colors  = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    engine  = Engine(classes, inference_device, model_config, local_num_cam, log_file)
    return engine, classes, colors


def preprocess_img(ori_img: np.ndarray, model_config: dict, type: str, device: str):
    img_size = model_config[type]['img_size']
    inf_shape = (int(img_size * 9 / 16), img_size)

    img = None
    if type == YOLOV3_DETECT:
        img = letterbox(ori_img.copy(), new_shape=inf_shape)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img)# .to(device)
        img = img.float() / 255.0
    else:
        img = img

    return img


def process_result(ori_img: np.ndarray, det, classes, colors, model_config, type):
    img_size = model_config[type]['img_size']
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
