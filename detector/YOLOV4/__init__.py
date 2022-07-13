import sys
sys.path.append("detector/YOLOV4")


# from .detector import hello
from .detector import YOLOv4
from . import models, utils
__all__ = ['YOLOv4',
            'models',
            'utils'
            ]



