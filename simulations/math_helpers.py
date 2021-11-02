import math
from random import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def normed(v):
    if not np.any(v):
        return np.zeros_like(v)
    return v / np.linalg.norm(v)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = normed(v1)
    v2_u = normed(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def rand_line(n, c, l):
    rng = np.random.default_rng()
    return rng.random(n) * l + c


def rand_cluster(n, c, r):
    """
    returns n random points in disk of radius r centered at c
    Source: https://stackoverflow.com/a/44356472
    """
    x, y = c
    for i in range(n):
        theta = 2 * math.pi * random()
        s = r * random()
        yield (x + s * math.cos(theta), y + s * math.sin(theta))


def rand_square(n, c, s):
    x0, y0 = c
    rng = np.random.default_rng()
    xs = rng.random(n) * s + x0
    ys = rng.random(n) * s + y0
    return zip(xs, ys)


def rotate(v, rot_vec, theta):
    rot_vec = normed(rot_vec)  # normalize the rotation vector first
    rot = Rotation.from_rotvec(theta * rot_vec)
    return rot.apply(v)


# clst = rand_cluster(10000, (0, 0), 1)
# x, y = zip(*clst)
# ax = plt.figure().add_subplot(projection='3d')
# ax.scatter(x, y, 1)
# plt.show()
