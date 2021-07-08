import math
from random import random

import numpy as np


def rand_cluster(n, c, r):
    """
    returns n random points in disk of radius r centered at c
    Source: https://stackoverflow.com/a/44356472
    """
    x, y = c
    points = []
    for i in range(n):
        theta = 2*math.pi*random()
        s = r*random()
        points.append((x+s*math.cos(theta), y+s*math.sin(theta)))
    return points


def rand_square(n, c, s):
    x0, y0 = c
    rng = np.random.default_rng()
    xs = rng.random(n) * s + x0
    ys = rng.random(n) * s + y0
    return list(zip(xs, ys))


# clst = rand_cluster(10000, (0, 0), 1)
# x, y = zip(*clst)
# ax = plt.figure().add_subplot(projection='3d')
# ax.scatter(x, y, 1)
# plt.show()
