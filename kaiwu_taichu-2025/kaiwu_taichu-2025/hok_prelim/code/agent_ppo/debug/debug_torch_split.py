import torch

x = torch.zeros([10, 1024])
x1, *x23, x4 = torch.split(x, [512, 128, 256, 128], -1)
print(x1.shape, x23[0].shape, x23[1].shape, x4.shape)
