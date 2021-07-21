from abc import ABC, abstractmethod
from math_helpers import rotate

import numpy as np


def field_tot(r, t, field_sources):
    overriders = [
        x for x in field_sources if isinstance(x, Overriding) and x.isinside(r)
    ]
    num_overriders = len(overriders)

    if num_overriders > 1:
        raise ValueError(
            f"More than 1 active overriding field source at (r, t) = ({r}, {t})",
            r,
            t,
            field_sources,
        )
    elif num_overriders == 1:
        return overriders[0].field(r, t)
    else:
        result = np.zeros(3)
        for source in field_sources:
            result += source.field(r, t)
        return result


class FieldSource(ABC):
    @abstractmethod
    def field(self, r, t):
        pass


class Vladimirskii(FieldSource):
    def __init__(self, Hx, H_dot):
        """
        [Hx, H_dot] must be in nanotesla.
        With Hx = 10:
            H_dot = -41.5 for ~50.0% expected realignment
            H_dot = -100 for ~75.0% expected realignment
            H_dot = -200 for ~86.6% expected realignment
        """
        self.Hx = Hx
        self.H_dot = H_dot

    def field(self, r, t):
        return [self.Hx, 0, t * self.H_dot]


class Bounded(FieldSource):
    @abstractmethod
    def isinside(self, r):
        pass


class Overriding(Bounded):
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

    def field(self, r, t):
        dist_vec = self.dist_vec(r)
        direction = np.cross(self.I, dist_vec)
        B_hat = direction / np.linalg.norm(direction)
        mu_over_2pi = 2e2  # mu_0 / 2pi (approx), yields result in nanotesla
        dist = np.linalg.norm(dist_vec)
        I_norm = np.linalg.norm(self.I)
        if dist < self.R:
            return mu_over_2pi * I_norm * dist / (self.R * self.R) * B_hat
        else:
            return mu_over_2pi * I_norm / dist * B_hat


class Box(Bounded):
    def __init__(self, p, dims, B):
        self.p = np.array(p)
        self.dims = np.array(dims)
        self.B = lambda r, t: np.asarray(B(r, t))

    def isinside(self, r):
        for i in range(3):
            ri = r[i]
            pi = self.p[i]
            di = self.dims[i]
            if not (pi <= ri <= pi + di):
                return False
        return True

    def field(self, r, t):
        if self.isinside(r):
            return self.B(r, t)
        else:
            return np.zeros(3)


class OverridingBox(Box, Overriding):
    pass


class Metglas(Overriding):
    def __init__(self, p, B_sat, rot_vec, sat, cell_dims, cells_per_dim):
        def generate_domain_field():
            rng = np.random.default_rng()
            theta = rng.random() * np.pi * 2
            return rng.choice([B_sat, rotate(B_sat, rot_vec, theta)], p=[sat, 1 - sat])

        self.p = np.array(p)
        self.cell_dims = np.array(cell_dims)
        self.cells_per_dim = np.array(cells_per_dim)

        cells = [generate_domain_field() for i in range(np.prod(cells_per_dim))]
        self.cells = np.reshape(cells, (*cells_per_dim, 3))

    def get_indices(self, r):
        return tuple((r - self.p) // self.cell_dims)

    def isinside(self, r):
        indices = self.get_indices(r)
        for i, n in zip(indices, self.cells_per_dim):
            if not (0 <= i <= n - 1):
                return False
        return True

    def field(self, r, t):
        indices = self.get_indices(r)
        return self.cells[indices]


# x = Metglas([0, 0, 0], [1, 2, 3], [0, 0, 1], 0.8, [10, 10, 10], [2, 2, 2])
# print(x.field([95, 41, 100], 0))
