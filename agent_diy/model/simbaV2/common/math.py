#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
SimbaV2 数学工具函数
"""

import torch
import math


def l2normalize(tensor, dim=0, eps=1e-12):
    """L2归一化"""
    norm = tensor.norm(p=2, dim=dim, keepdim=True)
    return tensor / (norm + eps)
