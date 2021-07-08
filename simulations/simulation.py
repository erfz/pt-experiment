import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from random import random
import math
from field_sources import *


def vec_line_to_point(p, a, n):
    n = np.divide(n, np.linalg.norm(n))
    x = np.subtract(p, a)
    return x - n * np.dot(x, n)


def B_wire(r, r0, v, I, R):
    """
    [r] is field observation point (3D vector)
    [r0] is any point the infinite wire passes thru (3D vector)
    [v] is direction of current (3D vector)
    [I] is current (can be negative to have |[I]| current flow opposite [v])
    [R] is radius of wire
    Gives result in nanotesla
    """
    dist_vec = vec_line_to_point(r, r0, v)
    direction = np.sign(I) * np.cross(v, dist_vec)
    B_hat = direction / np.linalg.norm(direction)
    mu_over_2pi = 2e+2  # mu_0 / 2pi (approx), yields result in nanotesla
    dist = np.linalg.norm(dist_vec)
    if dist < R:
        return mu_over_2pi * abs(I) * dist / (R * R) * B_hat
    else:
        return mu_over_2pi * abs(I) / dist * B_hat


def B_tot(r, wires):
    """
    [r] is field observation point (3D vector)
    [wires] is array of (r0, v, I, R) tuples, one per infinite wire
    Gives result in nanotesla
    """
    result = np.zeros(3)
    for wire in wires:
        result += B_wire(r, *wire)
    return result


def B_two_wires(r, d, w1, w2):
    """
    Convention: y = 0 is the midline between the two wires
    and (x, y) = (0, 0) is the midpoint between the two wires.
    [r] is field observation point (3D vector)
    [d] is the distance between wires
    [w1] is the (current, radius) pair of wire 1
    [w2] is the (current, radius) pair of wire 2
    Gives result in nanotesla
    """
    top_wire_pos = np.array([0, d/2, 0])
    z_hat = np.array([0, 0, 1])
    wires = [(top_wire_pos, z_hat, *w1), (-top_wire_pos, z_hat, *w2)]
    return B_tot(r, wires)


def B_vladimirskii(t, Hx, H_dot):
    return [Hx, 0, t * H_dot]


def r_particle(vx, t, yz):
    """
    Position vector of particle throughout simulation,
    assuming only movement in x-direction with velocity [vx].
    Convention: t=0 corresponds to x=0
    """
    return [vx * t, *yz]


def rhs(t, S, B, vx, yz=(0, 0)):
    """
    [t] is time
    [S] is spin vector
    [B] is B-field function B(r {3D vec}, t) (must return results in nanotesla)
    """
    # mu_n (neutron) / h-bar scaled so that c*S x B (in nT) is in h-bar/second
    c = 2 * -9.162e-2
    return np.cross(c*S, B(r_particle(vx, t, yz), t))


def naive(f, t_bounds, y0, num_pts):
    """Converges slowly"""
    t0, tf = t_bounds
    y = np.array(y0)
    range, step = np.linspace(t0, tf, num_pts, retstep=True)
    for t in range:
        y += step * f(t, y)
    return y


def run_vladimirskii(Hx, H_dot, t_bounds=[-100, 100]):
    """
    Velocity independent.
    [Hx, H_dot] must be in nanotesla.
    With Hx = 10:
        H_dot = -41.5 for ~50.0% expected realignment
        H_dot = -100 for ~75.0% expected realignment
        H_dot = -200 for ~86.6% expected realignment
    """
    def f_vladimirskii(t, S): return rhs(
        t, S, lambda r, t: B_vladimirskii(t, Hx, H_dot), 0)

    rtol, atol = (1e-8, 1e-8)
    sol = solve_ivp(f_vladimirskii, t_bounds, [0, 0, 1/2],
                    method="LSODA", rtol=rtol, atol=atol)
    Sf = [sol.y[i][-1] for i in range(3)]

    # print(f"Number of f evals: {sol.nfev}")
    # print(f"Number of time points: {len(sol.t)}")

    return Sf


def run_particle(vx, yz, sources, t_bounds, S0):
    def f(t, S): return rhs(t, S, lambda r, t: field_tot(r, sources), vx, yz)

    rtol, atol = (1e-8, 1e-8)
    sol = solve_ivp(f, t_bounds, S0, method="LSODA", rtol=rtol, atol=atol)
    Sf = [sol.y[i][-1] for i in range(3)]

    # print(f"Number of f evals: {sol.nfev}")
    # print(f"Number of time points: {len(sol.t)}")

    return Sf


def generate_two_wires(d, w1, w2):
    I1, R1 = w1
    I2, R2 = w2
    z_hat = np.array([0, 0, 1])
    y_hat = np.array([0, 1, 0])
    return [InfiniteWire(d/2 * y_hat, I1 * z_hat, R1),
            InfiniteWire(-d/2 * y_hat, I2 * z_hat, R2)]


def run_two_wires_line(vx, d, w1, w2, N, t_bounds):
    two_wires = generate_two_wires(d, w1, w2)

    def S0(yz):
        t0, tf = t_bounds
        B0 = field_tot(r_particle(vx, t0, yz), two_wires)
        return B0 / np.linalg.norm(B0) / 2

    rng = np.random.default_rng()
    rand_floats = rng.random(N) * d/2
    rand_bools = rng.choice([-1, 1], N)
    rand_ys = [x * b for x, b in zip(rand_floats, rand_bools)]
    rand_Sf = [run_particle(vx, (y, 0), two_wires, t_bounds, S0((y, 0)))
               for y in rand_ys]
    # average over all final spin vectors
    return np.average(rand_Sf, axis=0)


# Sf_two_wires = run_particle(1000, (0, 0), generate_two_wires(
#     10, (10, 0), (-10, 0)), [-100, 100], [0, 1/2, 0])
# print(f"Final S (two wires): {Sf_two_wires}")

# Sf_vlad = run_vladimirskii(10, -41.5)
# print(f"Final S (Vladimirskii): {Sf_vlad}")
# print(f"-> Corresponding realignment probability: {100 * (Sf_vlad[2] + 1/2)}%")

# Sf_rand_line = run_two_wires_line(
#     1000, 10, (10, 0), (-10, 0), 100, [-100, 100])
# print(f"Final S (rand line): {Sf_rand_line}")


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


def run_two_wires_shape_2D(vx, d, w1, w2, N, t_bounds, shape):
    two_wires = generate_two_wires(d, w1, w2)

    def S0(yz):
        t0, tf = t_bounds
        B0 = field_tot(r_particle(vx, t0, yz), two_wires)
        return B0 / np.linalg.norm(B0) / 2

    if shape == "square":
        ys, zs = zip(*rand_square(N, (-d/2, 0), d))
    elif shape == "circle":
        ys, zs = zip(*rand_cluster(N, (0, 0), d/2))
    else:
        raise ValueError(f"'{shape}' is not a valid [shape] argument")

    rand_Sf = [run_particle(vx, (y, z), two_wires, t_bounds, S0((y, z)))
               for y, z in zip(ys, zs)]
    # average over all final spin vectors
    return np.average(rand_Sf, axis=0)


# shape = "square"
# Sf_shape = run_two_wires_shape_2D(
#     1000, 10, (10, 0), (-10, 0), 100, [-100, 100], shape)
# print(f"Final S ({shape}): {Sf_shape}")
