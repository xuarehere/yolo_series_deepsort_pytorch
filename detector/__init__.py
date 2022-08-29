from .YOLOv3 import YOLOv3
from .MMDet import MMDet



__all__ = ['build_detector']        # 

def build_detector(cfg, use_cuda):
    if cfg.USE_MMDET:
        return MMDet(cfg.MMDET.CFG, cfg.MMDET.CHECKPOINT,
                    score_thresh=cfg.MMDET.SCORE_THRESH,
                    is_xywh=True, use_cuda=use_cuda)
    else:
        if cfg.DETECT_MODEL == "yolov3":
            return YOLOv3(cfg.YOLOV3.CFG, cfg.YOLOV3.WEIGHT, cfg.YOLOV3.CLASS_NAMES, 
                        score_thresh=cfg.YOLOV3.SCORE_THRESH, nms_thresh=cfg.YOLOV3.NMS_THRESH, 
                        is_xywh=True, use_cuda=use_cuda)
        elif cfg.DETECT_MODEL == "yolov4":
            from .YOLOV4 import YOLOv4  # 当前文件导入对应的包
            return YOLOv4(  
                model_cfg=cfg.YOLOV4.CFG,
                weightfile=cfg.YOLOV4.WEIGHT,
                score_thresh=cfg.YOLOV4.SCORE_THRESH, 
                conf_thresh=cfg.YOLOV4.CONF_THRESH,
                nms_thresh=cfg.YOLOV4.NMS_THRESH, 
                is_xywh=cfg.YOLOV4.IS_XYWH, 
                use_cuda=use_cuda,
                imgsz=cfg.YOLOV4.IMG_SIZE,
                dataset_config=cfg.YOLOV4.DATASET
                )                          
        elif cfg.DETECT_MODEL == "yolov4Scaled":
            from .YOLOV4Scaled import YOLOv4Scaled  # 当前文件导入对应的包
            return YOLOv4Scaled(  
                weightfile=cfg.YOLOV4Scaled.WEIGHT,
                score_thresh=cfg.YOLOV4Scaled.SCORE_THRESH, 
                conf_thresh=cfg.YOLOV4Scaled.CONF_THRESH,
                nms_thresh=cfg.YOLOV4Scaled.NMS_THRESH, 
                is_xywh=cfg.YOLOV4Scaled.IS_XYWH, 
                use_cuda=use_cuda,
                imgsz=cfg.YOLOV4Scaled.IMG_SIZE
                )                                         
        elif cfg.DETECT_MODEL == "yolov5":
            from .YOLOV5 import YOLOv5  # 当前文件导入对应的包
            return YOLOv5(  
                weightfile=cfg.YOLOV5.WEIGHT,
                score_thresh=cfg.YOLOV5.SCORE_THRESH, 
                conf_thresh=cfg.YOLOV5.CONF_THRESH,
                nms_thresh=cfg.YOLOV5.NMS_THRESH, 
                is_xywh=cfg.YOLOV5.IS_XYWH, 
                use_cuda=use_cuda,
                imgsz=cfg.YOLOV5.IMG_SIZE
                )
        elif cfg.DETECT_MODEL == "yolov6":
            from .YOLOV6 import YOLOv6  # 当前文件导入对应的包
            return YOLOv6(  
                weightfile=cfg.YOLOV6.WEIGHT,
                score_thresh=cfg.YOLOV6.SCORE_THRESH, 
                conf_thresh=cfg.YOLOV6.CONF_THRESH,
                nms_thresh=cfg.YOLOV6.NMS_THRESH, 
                is_xywh=cfg.YOLOV6.IS_XYWH, 
                use_cuda=use_cuda,
                imgsz=cfg.YOLOV6.IMG_SIZE,
                dataset_config=cfg.YOLOV6.DATASET
                )                
        elif cfg.DETECT_MODEL == "yolov7":
            from .YOLOV7 import YOLOv7  # 当前文件导入对应的包
            return YOLOv7(  
                weightfile=cfg.YOLOV7.WEIGHT,
                score_thresh=cfg.YOLOV7.SCORE_THRESH, 
                conf_thresh=cfg.YOLOV7.CONF_THRESH,
                nms_thresh=cfg.YOLOV7.NMS_THRESH, 
                is_xywh=cfg.YOLOV7.IS_XYWH, 
                use_cuda=use_cuda,
                imgsz=cfg.YOLOV7.IMG_SIZE
                )                
        elif cfg.DETECT_MODEL == "yolox":
            from .YOLOX import YOLOx  # 当前文件导入对应的包
            return YOLOx(  
                weightfile=cfg.YOLOV7.WEIGHT,
                score_thresh=cfg.YOLOV7.SCORE_THRESH, 
                conf_thresh=cfg.YOLOV7.CONF_THRESH,
                nms_thresh=cfg.YOLOV7.NMS_THRESH, 
                is_xywh=cfg.YOLOV7.IS_XYWH, 
                use_cuda=use_cuda,
                imgsz=cfg.YOLOV7.IMG_SIZE
                )    
        elif cfg.DETECT_MODEL == "yolor":
            from .YOLOR import YOLOr  # 当前文件导入对应的包
            return YOLOr(  
                model_cfg=cfg.YOLOV4.CFG,
                weightfile=cfg.YOLOV4.WEIGHT,
                score_thresh=cfg.YOLOV4.SCORE_THRESH, 
                conf_thresh=cfg.YOLOV4.CONF_THRESH,
                nms_thresh=cfg.YOLOV4.NMS_THRESH, 
                is_xywh=cfg.YOLOV4.IS_XYWH, 
                use_cuda=use_cuda,
                imgsz=cfg.YOLOV4.IMG_SIZE,
                dataset_config=cfg.YOLOV4.DATASET
                )              
        else:
            raise Exception("Need to specify the detection model!")

