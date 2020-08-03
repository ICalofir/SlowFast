#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""

from fvcore.common.config import CfgNode

def add_custom_config(_C):
    # Add your own customized configs.
    _C.CUSTOM_CONFIG = CfgNode()

    _C.CUSTOM_CONFIG.WEIGHT = _C.MODEL.NUM_CLASSES * [1.]
