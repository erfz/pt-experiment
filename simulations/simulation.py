import numpy as np
from scipy.integrate import solve_ivp

from field_sources import *
from math_helpers import *


class Particle:
    def __init__(self, v, r0, sources, t_bounds, S0=None):
        self.v = np.array(v)
        self.r0 = np.array(r0)
        self.sources = sources
        self.t_bounds = t_bounds
        self.S0 = normed(self.B0()) if S0 is None else np.array(S0)

    def r(self, t):
        return self.r0 + self.v * t

    def B0(self):
        t0, _ = self.t_bounds
        return field_tot(self.r(t0), t0, self.sources)

    def f(self, t, S):
        # mu_n (neutron) / h-bar scaled so that c*S x B (in nT) is in h-bar/second
        c = 2 * -9.162e-2
        return np.cross(c * S, field_tot(self.r(t), t, self.sources))

    def simulate_with_output(self):
        rtol, atol = (1e-8, 1e-8)
        sol = solve_ivp(
            self.f, self.t_bounds, self.S0, method="LSODA", rtol=rtol, atol=atol
        )
        Sf = [sol.y[i][-1] for i in range(3)]

        # print(f"Number of f evals: {sol.nfev}")
        # print(f"Number of time points: {len(sol.t)}")

        return Sf, sol.t, sol.y

    def simulate(self):
        return self.simulate_with_output()[0]


def naive(f, t_bounds, y0, num_pts):
    """Converges slowly"""
    t0, tf = t_bounds
    y = np.array(y0)
    range, step = np.linspace(t0, tf, num_pts, retstep=True)
    for t in range:
        y += step * f(t, y)
    return y


def generate_two_wires(d, w1, w2):
    I1, R1 = w1
    I2, R2 = w2
    z_hat = np.array([0, 0, 1])
    y_hat = np.array([0, 1, 0])
    return [
        InfiniteWire(d / 2 * y_hat, I1 * z_hat, R1),
        InfiniteWire(-d / 2 * y_hat, I2 * z_hat, R2),
    ]


def rand_line_sim(vx, d, oriented_sources, N, t_bounds):
    rng = np.random.default_rng()
    rand_floats = rng.random(N) * d / 2
    rand_bools = rng.choice([-1, 1], N)
    rand_ys = [x * b for x, b in zip(rand_floats, rand_bools)]
    rand_Sf = [
        Particle([vx, 0, 0], [0, y, 0], oriented_sources, t_bounds).simulate()
        for y in rand_ys
    ]
    # average over all final spin vectors
    return np.average(rand_Sf, axis=0)


def rand_shape_2D_sim(vx, d, oriented_sources, N, t_bounds, shape):
    if shape == "square":
        shape_points = rand_square(N, (-d / 2, 0), d)
    elif shape == "circle":
        shape_points = rand_cluster(N, (0, 0), d / 2)
    else:
        raise ValueError(f"'{shape}' is not a valid [shape] argument")

    rand_Sf = [
        Particle([vx, 0, 0], [0, y, z], oriented_sources, t_bounds).simulate()
        for y, z in shape_points
    ]
    # average over all final spin vectors
    return np.average(rand_Sf, axis=0)
