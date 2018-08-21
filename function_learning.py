import math
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import MLP, NAC, NALU

NORMALIZE = True
NUM_LAYERS = 2
HIDDEN_DIM = 2
LEARNING_RATE = 1e-3
NUM_ITERS = int(1e5)
RANGE = [5, 10]
ARITHMETIC_FUNCTIONS = {
    'add': lambda x, y: x + y,
    'sub': lambda x, y: x - y,
    'mul': lambda x, y: x * y,
    'div': lambda x, y: x / y,
    'squared': lambda x, y: torch.pow(x, 2),
    'root': lambda x, y: torch.sqrt(x),
}
