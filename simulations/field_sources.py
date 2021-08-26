from abc import ABC, abstractmethod
from math_helpers import rotate

import numpy as np


def field_tot(r, t, field_sources):
    overriders = [
        s for s in field_sources if isinstance(s, Overriding) and s.isinside(r)
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
        for s in field_sources:
            if isinstance(s, Overriding):
                continue
            result += s.field(r, t)
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
        for ri, pi, di in zip(r, self.p, self.dims):
            if not (pi <= ri <= pi + di):
                return False
        return True

    def field(self, r, t):
        if self.isinside(r):
            return self.B(r, t)
        else:
            return np.zeros(3)


class OverridingBox(Box, Overriding):
    def field(self, r, t):
        # isinside(r) check not needed since Overriding field sources
        # are only used when inside them
        return self.B(r, t)


class Reiniting(FieldSource):
    @abstractmethod
    def reinit(self):
        pass


class Metglas(Box, Overriding, Reiniting):
    def __init__(self, p, dims, B_sat, rot_vec, sat, cell_length_range):
        rng = np.random.default_rng()

        def generate_domain_field():
            not_random = rng.choice([True, False], p=[sat, 1 - sat])
            if not_random:
                return B_sat
            else:
                return rotate(B_sat, rot_vec, rng.random() * np.pi * 2)

        length_min, length_max = cell_length_range
        x_min, _, _ = p
        x_max, _, _ = np.add(p, dims)

        def generate_cells_fields():
            extents = [x_min]

            while True:
                length = rng.random() * (length_max - length_min) + length_min
                last = extents[-1]
                new_last = last + length
                if new_last > x_max:
                    new_last = x_max
                extents.append(new_last)
                if new_last == x_max:
                    break

            fields = [generate_domain_field() for i in range(len(extents) - 1)]

            self.extents = np.array(extents)
            self.fields = np.reshape(fields, (len(extents) - 1, 3))

        self.generate_cells_fields = generate_cells_fields
        self.generate_cells_fields()
        self.p = p
        self.dims = dims

    def reinit(self):
        self.generate_cells_fields()

    def field(self, r, t):
        x, _, _ = r
        l = self.extents
        for (idx, e) in enumerate(l):
            if idx < len(l) - 1:
                x1, x2 = e, l[idx + 1]
                if x1 <= x <= x2:
                    return self.fields[idx]
        raise ValueError(f"r[0] = {x} was not within the extents", r, self.extents)
