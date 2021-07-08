from abc import ABC, abstractmethod

import numpy as np


def field_tot(r, field_sources):
    result = np.zeros(3)
    for source in field_sources:
        result += source.field(r)
    return result


class FieldSource(ABC):
    @abstractmethod
    def field(self, r):
        pass


class InfiniteWire(FieldSource):
    def __init__(self, p, I, R):
        self.p = np.array(p)
        self.I = np.array(I)
        self.R = R

    def dist_vec(self, r):
        n = self.I / np.linalg.norm(self.I)
        x = r - self.p
        return x - n * np.dot(x, n)

    def field(self, r):
        dist_vec = self.dist_vec(r)
        direction = np.cross(self.I, dist_vec)
        B_hat = direction / np.linalg.norm(direction)
        mu_over_2pi = 2e+2  # mu_0 / 2pi (approx), yields result in nanotesla
        dist = np.linalg.norm(dist_vec)
        I_norm = np.linalg.norm(self.I)
        if dist < self.R:
            return mu_over_2pi * I_norm * dist / (self.R * self.R) * B_hat
        else:
            return mu_over_2pi * I_norm / dist * B_hat
