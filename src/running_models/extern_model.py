from collections import defaultdict
from typing import List
import random
import os

import numpy as np
import torch
import cv2
from shapely.geometry import Polygon

from src.running_models.detector.advanced.counter import ObjectCounter
from yolov3.models import Darknet
from src.running_models.yolov3_utils import (
    letterbox,
    load_classes,
    non_max_suppression,
    plot_one_box,
    scale_coords,
    xywh2xyxy,
    plot_trajectory,
    fill_restrict_area,
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

        self.conf_thre = 0.5
        self.iou_thre = 0.5
        self.img_size = 512

        if not os.path.exists(os.path.dirname(self.log_file)):
            os.mkdir(os.path.dirname(self.log_file))

    def _hot_init(self):
        """
        pre-initiate model pool
        store them in cuda memory, or at list in cpu memory
        """

        for model_type in __model_choices__:
            if model_type == YOLOV3_DETECT:
                weight = self.model_config[model_type]["weight"]
                config = self.model_config[model_type]["config"]
                img_size = self.model_config[model_type]["img_size"]

                model = Darknet(config, img_size)
                torch_model = torch.load(weight, map_location=self.device)
                model.load_state_dict(torch_model["model"], strict=False)
                # print("Warning: failed load model, try again using relaxed mode")
                model.to(self.device).eval()
                with torch.no_grad():
                    model(torch.randn(1, 3, img_size, img_size).to(self.device), augment=False)
                print(f"Finish initialize {model_type}")

            elif model_type == YOLOV11_TRACK:
                weight = self.model_config[model_type]["weight"]
                img_size = self.model_config[model_type]["img_size"]

                if not os.path.exists(os.path.dirname(weight)):
                    os.mkdir(os.path.dirname(weight))
                model = [ObjectCounter(
                    device=self.device,
                    weights=weight,
                    classes_of_interest=self.classes,
                    log_file=self.log_file,
                    cam_id=i,
                ) for i in range(self.num_cam)]
                print(f"Finish initialize {model_type}")
            else:
                model = None
            
            self.model_pool[model_type] = model

    def __call__(self, chunk_tsr, **kwargs):
        """
        engine runtime call
        """

        self.model = self.model_pool[self.type]
        results = None
        if self.type == YOLOV3_DETECT:
            self.model: Darknet
            with torch.no_grad():
                pred = self.model(chunk_tsr.to(self.device), **kwargs)[0]
            pred = pred.clone().detach().cpu()
            results = non_max_suppression(pred, conf_thres=self.conf_thre, iou_thres=self.iou_thre, multi_label=False)
        elif self.type == YOLOV11_TRACK:
            results = []
            self.model: List[ObjectCounter]
            with torch.no_grad():
                for cam_id, input in chunk_tsr:
                    cam_id: int
                    pred = self.model[cam_id].online_predict(input, conf_thre=self.conf_thre, iou_thre=self.iou_thre, imgsz=self.img_size)
                    results.append(pred)
        else:
            results = [None] * len(chunk_tsr)

        return results

    def set_model(self, type, conf_thre, iou_thre, selected_class, img_size, polygons):
        """
        polygons: list of camera_polys -> list of polygons -> list of vertices
        """

        if type is not None:
            self.type = type
            self.model = self.model_pool[type]
            self.conf_thre = conf_thre
            self.iou_thre = iou_thre
            self.img_size = img_size
            if type == YOLOV11_TRACK:
                selected_class = [self.classes[i] for i in selected_class]
                for i in range(self.num_cam):
                    self.model[i].reset_cumulative_counts(selected_class)
                    self.model[i].restricted_areas.clear()
                    # TODO: each camera may have multiple polygons, though now only 1
                    for poly in polygons[i]:
                        self.model[i].add_restricted_area(poly)
            print(f"model changed to {type}")


def initialize_model_engine(
    local_num_cam: int,
    inference_device: str = "cpu",
    model_config: str = "running_models/model_config.json",
    log_file: str = "logs/tracking_log.txt",
    seedid: int = 1024,
):
    random.seed(seedid)
    classes = load_classes(model_config[YOLOV3_DETECT]["names"])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    engine = Engine(classes, inference_device, model_config, local_num_cam, log_file)
    traj_colors = [[random.randint(50, 255) for _ in range(3)] for _ in range(100)]
    track_history = defaultdict(list)

    return engine, classes, colors, traj_colors, track_history


def preprocess_img(ori_img: np.ndarray, img_size: int, type: str):
    """
    resize and transform np.ndarray to torch.Tensor
    """

    img = None
    if type == YOLOV3_DETECT:
        img = letterbox(ori_img.copy(), new_shape=img_size)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to bsx3x416x416
        shape = img.shape[1:]
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img)
        img = img.float() / 255.0
    else:
        # img = img
        img = letterbox(ori_img.copy(), new_shape=img_size)
        shape = img.shape[:2]

    return img, shape


def process_result(ori_img: np.ndarray, det, classes, colors, new_shape, type, selected_class, traj_colors, track_history, polygons):
    """
    render results on original image
    polygons: list of polygons -> list of vertices
    """

    # process detections
    if type == YOLOV3_DETECT:
        # draw bounding boxes
        if det is not None and len(det):
            count_current_frame = {classes[k]: 0 for k in selected_class}
            det[:, :4] = scale_coords(new_shape, det[:, :4], ori_img.shape).round()

            for *xyxy, conf, cls in reversed(det):
                if int(cls) not in selected_class:
                    continue
                count_current_frame[classes[int(cls)]] += 1
                label = "%s %.2f" % (classes[int(cls)], conf)
                plot_one_box(xyxy, ori_img, label=label, color=colors[int(cls)])
        else:
            count_current_frame = {classes[k]: 0 for k in selected_class}

        cnt_packet = (count_current_frame, None)

    elif type == YOLOV11_TRACK:
        # draw bounding boxes and trajectory
        names, confs, boxes, cumulative_counts, track_ids, alert = det
        ori_img = fill_restrict_area(ori_img, polygons, alert)
        if len(boxes):
            count_current_frame = {classes[k]: 0 for k in selected_class}
            boxes = scale_coords(new_shape, xywh2xyxy(torch.tensor(boxes)), ori_img.shape).round()
            for xyxy, conf, cls, tid in zip(boxes, confs, names, track_ids):
                if int(cls) not in selected_class:
                    continue
                count_current_frame[classes[int(cls)]] += 1
                label = "%s %.2f" % (classes[int(cls)], conf)
                plot_one_box(xyxy, ori_img, label=label, color=colors[int(cls)])
                plot_trajectory(xyxy, ori_img, tid, track_history, traj_colors)
        else:
            count_current_frame = {classes[k]: 0 for k in selected_class}

        cnt_packet = (count_current_frame, cumulative_counts)


    return ori_img, cnt_packet
