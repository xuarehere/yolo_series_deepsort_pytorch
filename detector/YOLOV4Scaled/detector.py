import torch
import logging
import numpy as np
import cv2

from detector.YOLOV4Scaled.models.experimental import attempt_load
from detector.YOLOV4Scaled.utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer
from detector.YOLOV4Scaled.utils.datasets import  letterbox

class YOLOv4Scaled(object):
    def __init__(self, weightfile="", 
    score_thresh=0.0, conf_thresh=0.25, nms_thresh=0.45,
                 is_xywh=True, use_cuda=True, imgsz=640, half=False):    
        # net definition
        self.half = half
        self.device = "cuda" if use_cuda else "cpu"
        self.net = attempt_load(weightfile, map_location=self.device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=self.net.stride.max())  # check img_size
        if half:
            self.net.half()  # to FP16        

        self.class_names = self.net.module.names if hasattr(self.net, 'module') else self.net.names

        # # constants
        self.size = imgsz , imgsz
        self.score_thresh = score_thresh
        self.conf_thresh = conf_thresh
        self.is_xywh = is_xywh          # 未用到
        self.num_classes = self.net.nc
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
        img = img.astype(np.float) / 255.
        img = torch.from_numpy(img).float().unsqueeze(0)
        # forward
        with torch.no_grad():
            img = img.to(self.device)
            out_boxes = self.net(img)[0]
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
            det_box[:, :4] = scale_coords(img_infer.shape[2:], det_box[:, :4], im0_original.shape).round()            
            bbox = det_box[:, :4]
            if self.is_xywh:
                # bbox x y w h
                bbox = self.xyxy_to_xywh(bbox)
                pass

            cls_conf = boxes[:, 4]
            cls_ids = boxes[:, 5].long()
        return bbox.cpu().numpy(), cls_conf.cpu().numpy(), cls_ids.cpu().numpy()

    def load_class_names(self, namesfile):
        with open(namesfile, 'r', encoding='utf8') as fp:
            class_names = [line.strip() for line in fp.readlines()]
        return class_names

