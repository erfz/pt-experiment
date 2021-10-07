import numpy as np
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp

from field_sources import *
from math_helpers import *


class Particle:
    def __init__(self, v, r0, sources, t_bounds, S0=None, max_step=np.inf):
        self.v = np.array(v)
        self.r0 = np.array(r0)
        self.sources = sources
        self.t_bounds = t_bounds
        self.S0 = normed(self.B0()) if S0 is None else np.array(S0)
        self.max_step = max_step

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
        # Generate any GenerateEachRun sources...temporary solution?
        for s in self.sources:
            if isinstance(s, GenerateEachRun):
                s.generate()

        rtol, atol = (1e-8, 1e-8)
        sol = solve_ivp(
            self.f,
            self.t_bounds,
            self.S0,
            method="LSODA",
            rtol=rtol,
            atol=atol,
            max_step=self.max_step,
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


def rand_shape_sim(
    vx_gen, d, oriented_sources, N, t_bounds, shape, S0=None, max_step=np.inf
):
    if shape == "line":
        shape_points = ((y, 0) for y in rand_line(N, -d / 2, d))
    elif shape == "square":
        shape_points = rand_square(N, (-d / 2, 0), d)
    elif shape == "circle":
        shape_points = rand_cluster(N, (0, 0), d / 2)
    else:
        raise ValueError(f"'{shape}' is not a valid [shape] argument")

    def f(y, z):
        return Particle(
            [vx_gen(), 0, 0], [0, y, z], oriented_sources, t_bounds, S0, max_step
        ).simulate()

    rand_Sf = Parallel(n_jobs=-1)(delayed(f)(y, z) for y, z in shape_points)

    # average over all final spin vectors
    return np.average(rand_Sf, axis=0)
