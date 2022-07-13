import sys
sys.path.append("detector/YOLOV4Scaled")


from .detector import YOLOv4Scaled
from . import models, utils
__all__ = ['YOLOv4Scaled',
            'models',
            'utils',
            ]



