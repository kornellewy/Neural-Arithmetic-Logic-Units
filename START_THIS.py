import math
import random
import numpy as np
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *

def train(model, optimizer, data, target, num_iters):
    for i in range(num_iters):
        out = model(data)
        loss = F.mse_loss(out, target)
        mea = torch.mean(torch.abs(target - out))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 5000 == 0:
            print("\t{}/{}: loss: {:.3f} - mea: {:.3f}".format(
                i+1, num_iters, loss.item(), mea.item())
            )
#############################################################
# permute the first column with the third
A = torch.from_numpy(np.array([
    [0, 1, -1],
    [3, -1, 1],
    [1, 1, -2],
])).float()

B = torch.from_numpy(np.array([
    [-1, 1, -0],
    [1, -1, 3],
    [-2, 1, 1],
])).float()

P = torch.from_numpy(np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
])).float()

assert torch.allclose(torch.matmul(A, P), B)

net = NeuralAccumulatorCell(3,3)
optim = torch.optim.RMSprop(net.parameters(), lr=1e-4)
train(net, optim, A, B, int(4e4))
print(net.W.data)
###############################################################
# scale the first column by 5
A = torch.from_numpy(np.array([
    [0, 1, -1],
    [3, -1, 1],
    [1, 1, -2],
])).float()

B = torch.from_numpy(np.array([
    [0, 1, -1],
    [15, -1, 1],
    [5, 1, -2],
])).float()

P = torch.from_numpy(np.array([
    [5, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
])).float()

assert torch.allclose(torch.matmul(A, P), B)

net = NeuralAccumulatorCell(3, 3)
optim = torch.optim.RMSprop(net.parameters(), lr=1e-4)

train(net, optim, A, B, int(9e4))

print(net.W.data)
######################################################################3

def basis_vec(k, n):
    """Creates the k'th standard basis vector in R^n."""
    error_msg = "[!] k cannot exceed {}.".format(n)
    assert (k < n), error_msg
    b = np.zeros([n, 1])
    b[k] = 1
    return b

# add -3x the second column to the first => P = (I - (c)(e_k)(e_l.T))

A = torch.from_numpy(np.array([
    [3, 1, -1],
    [3, -1, 1],
    [1, 1, -2],
])).float()

B = torch.from_numpy(np.array([
    [0, 1, -1],
    [6, -1, 1],
    [-2, 1, -2],
])).float()

P = torch.from_numpy(
    np.eye(3) + (-3)*basis_vec(1, 3).dot(basis_vec(0, 3).T)
).float()

assert torch.allclose(torch.matmul(A, P), B)

net = NeuralAccumulatorCell(3, 3)
optim = torch.optim.RMSprop(net.parameters(), lr=1e-4)

train(net, optim, A, B, int(9e4))

print(net.W.data)
