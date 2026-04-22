import numpy as np

# x = np.concatenate([[1],[3,2], np.array([1,2,3]), [True]])
# print(x)

# x = np.array([1.2,1.4, 1.5], np.float32).round().astype(np.int32)
# print(x)

x = np.zeros((5, 5), np.float32)
pos = np.array([2, 2])
x[*pos] = 1.0
print(x)
print(np.argwhere(x == 1.0)[0])
