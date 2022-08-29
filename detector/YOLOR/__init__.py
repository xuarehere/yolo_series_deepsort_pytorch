import sys
sys.path.append("detector/YOLOR")


from .detector import YOLOr
from . import models, utils
__all__ = ['YOLOr',
            'models',
            'utils'
            ]



