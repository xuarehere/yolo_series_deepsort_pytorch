import sys
sys.path.append("detector/YOLOV5")


# from .detector import hello
from .detector import YOLOv5
from . import models, utils
__all__ = ['YOLOv5',
            'models',
            'utils'
            ]



