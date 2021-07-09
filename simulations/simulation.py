import numpy as np
from scipy.integrate import solve_ivp

from field_sources import *
from math_helpers import *


def r_particle(t, v, r0):
    """
    Position vector of particle throughout simulation,
    assuming only movement in x-direction with velocity [vx].
    Convention: t=0 corresponds to x=0
    """
    d = np.multiply(v, t)
    return np.add(r0, d)


def rhs(t, S, B, v, r0):
    """
    [t] is time
    [S] is spin vector
    [B] is B-field function B(r {3D vec}, t) (must return results in nanotesla)
    """
    # mu_n (neutron) / h-bar scaled so that c*S x B (in nT) is in h-bar/second
    c = 2 * -9.162e-2
    return np.cross(c * S, B(r_particle(t, v, r0), t))


def naive(f, t_bounds, y0, num_pts):
    """Converges slowly"""
    t0, tf = t_bounds
    y = np.array(y0)
    range, step = np.linspace(t0, tf, num_pts, retstep=True)
    for t in range:
        y += step * f(t, y)
    return y


def run_particle_outputs(v, r0, sources, t_bounds, S0):
    def f(t, S):
        return rhs(t, S, lambda r, t: field_tot(r, t, sources), v, r0)

    rtol, atol = (1e-8, 1e-8)
    sol = solve_ivp(f, t_bounds, S0, method="LSODA", rtol=rtol, atol=atol)
    Sf = [sol.y[i][-1] for i in range(3)]

    # print(f"Number of f evals: {sol.nfev}")
    # print(f"Number of time points: {len(sol.t)}")

    return Sf, sol.t, sol.y


def run_particle(v, r0, sources, t_bounds, S0):
    return run_particle_outputs(v, r0, sources, t_bounds, S0)[0]


def generate_two_wires(d, w1, w2):
    I1, R1 = w1
    I2, R2 = w2
    z_hat = np.array([0, 0, 1])
    y_hat = np.array([0, 1, 0])
    return [
        InfiniteWire(d / 2 * y_hat, I1 * z_hat, R1),
        InfiniteWire(-d / 2 * y_hat, I2 * z_hat, R2),
    ]


def run_two_wires_line(vx, d, w1, w2, N, t_bounds):
    two_wires = generate_two_wires(d, w1, w2)
    v = [vx, 0, 0]

    def S0(r0):
        t0, tf = t_bounds
        B0 = field_tot(r_particle(t0, v, r0), 0, two_wires)
        return B0 / np.linalg.norm(B0) / 2

    rng = np.random.default_rng()
    rand_floats = rng.random(N) * d / 2
    rand_bools = rng.choice([-1, 1], N)
    rand_ys = [x * b for x, b in zip(rand_floats, rand_bools)]
    rand_Sf = [
        run_particle(v, [0, y, 0], two_wires, t_bounds, S0([0, y, 0])) for y in rand_ys
    ]
    # average over all final spin vectors
    return np.average(rand_Sf, axis=0)


def run_two_wires_shape_2D(vx, d, w1, w2, N, t_bounds, shape):
    two_wires = generate_two_wires(d, w1, w2)
    v = [vx, 0, 0]

    def S0(r0):
        t0, tf = t_bounds
        B0 = field_tot(r_particle(t0, v, r0), 0, two_wires)
        return B0 / np.linalg.norm(B0) / 2

    if shape == "square":
        ys, zs = zip(*rand_square(N, (-d / 2, 0), d))
    elif shape == "circle":
        ys, zs = zip(*rand_cluster(N, (0, 0), d / 2))
    else:
        raise ValueError(f"'{shape}' is not a valid [shape] argument")

    rand_Sf = [
        run_particle(v, [0, y, z], two_wires, t_bounds, S0([0, y, z]))
        for y, z in zip(ys, zs)
    ]
    # average over all final spin vectors
    return np.average(rand_Sf, axis=0)
