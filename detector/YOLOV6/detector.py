#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import os.path as osp
import math
from tqdm import tqdm
import numpy as np
import cv2
import torch
from PIL import ImageFont

from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.utils.torch_utils import get_model_info


class YOLOv6(object):
    def __init__(self, weightfile="", 
    score_thresh=0.0, conf_thresh=0.25, nms_thresh=0.45,
                 is_xywh=True, use_cuda=True, imgsz=640, half=False, dataset_config=""):    
        # net definition
        self.half = half
        self.device = "cuda" if use_cuda else "cpu"
        self.net = DetectBackend(weightfile, device=self.device)  # load FP32 model
        self.size = self.check_img_size(imgsz, s=self.net.stride)  # check img_size
        if half and self.device == "cuda":
            self.net.model.half()  # to FP16     
        else:
            self.net.model.float()
        try:
            self.class_names = self.net.module.names if hasattr(self.net.model, 'module') else self.net.names
        except:
            if dataset_config !="":
                self.class_names = load_yaml(dataset_config)['names']
            else:
                raise Exception("Can not find calss names!")

        # # constants
        self.score_thresh = score_thresh
        self.conf_thresh = conf_thresh
        self.is_xywh = is_xywh          # 未用到
        self.num_classes = len(self.class_names)
        self.iou_thres = nms_thresh
    
    def xyxy_to_xywh(self, boxes_xyxy):
        if isinstance(boxes_xyxy, torch.Tensor):
            boxes_xywh = boxes_xyxy.clone()
        elif isinstance(boxes_xyxy, np.ndarray):
            boxes_xywh = boxes_xyxy.copy()

        boxes_xywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.
        boxes_xywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.
        boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

        return boxes_xywh

    def __call__(self, ori_img):
        # img to tensor
        assert isinstance(ori_img, np.ndarray), "input must be a numpy array!"
        
        # resize
        img = letterbox(ori_img, new_shape=self.size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)        
        img = img.astype(np.float) / 255.   # 0 - 255 to 0.0 - 1.0
        img = torch.from_numpy(img).float().unsqueeze(0)
        # forward
        with torch.no_grad():
            img = img.to(self.device)
            out_boxes = self.net(img)
            pred = non_max_suppression(out_boxes, self.conf_thresh, self.iou_thres)
            boxes = pred[0]
            if str(self.score_thresh)  == "0.0":
                pass
            else:
                boxes = boxes[boxes[:, -2] > self.score_thresh, :]  # bbox xmin ymin xmax ymax;     Detections matrix nx6 (xyxy, conf, cls)

        if len(boxes) == 0:
            bbox = torch.FloatTensor([]).reshape([0, 4])
            cls_conf = torch.FloatTensor([])
            cls_ids = torch.LongTensor([])
        else:
            # Rescale boxes from img_size to im0 size
            img_infer = img
            det_box = boxes
            im0_original = ori_img
            det_box[:, :4] = self.rescale(img_infer.shape[2:], det_box[:, :4], im0_original.shape).round()            
            bbox = det_box[:, :4]
            if self.is_xywh:
                # bbox x y w h
                bbox = self.xyxy_to_xywh(bbox)
                pass
            # bbox *= torch.FloatTensor([[width, height, width, height]])     # bbox 比例 ==》 实际的像素位置
            cls_conf = boxes[:, 4]
            cls_ids = boxes[:, 5].long()
        return bbox.cpu().numpy(), cls_conf.cpu().numpy(), cls_ids.cpu().numpy()

    def load_class_names(self, namesfile):
        with open(namesfile, 'r', encoding='utf8') as fp:
            class_names = [line.strip() for line in fp.readlines()]
        return class_names

    def make_divisible(self, x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

    def check_img_size(self, img_size, s=32, floor=0):
        """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
        if isinstance(img_size, int):  # integer i.e. img_size=640
            new_size = max(self.make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
            new_size = [max(self.make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size if isinstance(img_size,list) else [new_size]*2
    
    @staticmethod
    def rescale(ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes