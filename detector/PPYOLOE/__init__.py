#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import sys
sys.path.append("detector/PPYOLOE")


from .detector import PPYOLOe
from . import tools
__all__ = ['PPYOLOe',
            'tools',
            ]



