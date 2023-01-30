import torch
import numpy as np
from detector.YOLOV8.ultralytics import YOLO
from detector.YOLOV8.utils.general import check_img_size
from detector.YOLOV8.utils.augmentations import letterbox
from detector.YOLOV8.utils.torch_utils import select_device
from detector.YOLOV8.ultralytics.nn.autobackend import AutoBackend
from detector.YOLOV8.ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
class YOLOv8(object):
    def __init__(self, weightfile="", 
    score_thresh=0.0, conf_thresh=0.25, nms_thresh=0.45,
                 is_xywh=True, use_cuda=True, imgsz=(640, 640),**kwargs):    
        # net definition
        # self.device = "cuda" if use_cuda else "cpu"
        config = kwargs['config']
        DEVICE = config['DEVICE']
        self.device = select_device(DEVICE)
        self.OpenCV_DNN = config["OpenCV_DNN"]
        self.DATA_CONFIG = config["DATA_CONFIG"]
        self.half = config["HALF"]
        self.net = YOLO(weightfile).model
        self.max_det   = config['MAX_DET']
        self.net = AutoBackend(self.net, device=self.device, dnn=self.OpenCV_DNN, fp16=self.half)
        self.net.warmup(imgsz=(1, 3, *imgsz))  # warmup
        imgsz = check_img_size(imgsz, s=self.net.stride)  # check img_size
        self.class_names = self.net.module.names if hasattr(self.net, 'module') else self.net.names

        # constants
        self.size = imgsz 
        self.score_thresh = score_thresh
        self.conf_thresh = conf_thresh
        self.is_xywh = is_xywh          

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
        img = torch.from_numpy(img).float().unsqueeze(0)
        img /=  255.        
        # forward
        with torch.no_grad():
            img = img.to(self.device)
            out_boxes = self.net(img)
            preds = self.postprocess(out_boxes, img, ori_img)
            boxes = preds[0]
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
            # img_infer = img
            det_box = boxes
            # im0_original = ori_img
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

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.device)
        img =  img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img
    
    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        conf_thres=self.conf_thresh,
                                        iou_thres=self.iou_thres,
                                        agnostic=False,
                                        max_det=self.max_det)

        for i, pred in enumerate(preds):
            shape =  orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds