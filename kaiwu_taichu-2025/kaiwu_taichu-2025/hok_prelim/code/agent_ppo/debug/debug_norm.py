import numpy as np

def norm(x, min_x, max_x):
    """ clip到min, max并缩放到(-1, 1) """
    if not isinstance(x, np.ndarray):
        x = np.array(x, np.float32)
    x = np.maximum(np.minimum(max_x, x), min_x)
    return (x - min_x) / (max_x - min_x)

x = norm([1, 2, 3, 4], 0, 10)
print(x, x.dtype)
