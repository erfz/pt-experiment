import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def B_wire(r, r0, I):
    """
    [r] is field observation point (3D vector)
    [r0] is position of infinite wire (2D vector)
    [I] is current (positive means in +z-direction)
    Gives result in nanotesla
    """
    # ignore z-component of r, since wire extends infinitely in z
    dist_vec = r[:-1] - r0
    z_hat = np.array([0, 0, 1])
    direction = np.cross(z_hat, dist_vec)
    B_hat = direction / np.linalg.norm(direction)
    mu2pi = 2e+2  # mu_0 / 2pi (approx), yields result in nanotesla
    dist = np.linalg.norm(dist_vec)
    return mu2pi * I / dist * B_hat


def B_tot(r, wires):
    """
    [r] is field observation point (3D vector)
    [wires] is array of (r0, I) pairs, one per infinite wire
    Gives result in nanotesla
    """
    result = np.zeros(3)
    for r0, I in wires:
        result += B_wire(r, r0, I)
    return result


def B_two_wires(r, d, I1, I2):
    """
    Convention: y = 0 is the midline between the two wires
    and (x, y) = (0, 0) is the midpoint between the two wires.
    [r] is field observation point (3D vector)
    [d] is the distance between wires
    [I1], [I2] is current of top and bottom wire respectively
    Gives result in nanotesla
    """
    top_wire_pos = np.array([0, d/2])
    wires = [(top_wire_pos, I1), (-top_wire_pos, I2)]
    return B_tot(r, wires)


def B_vladimirskii(t, Hx, H_dot):
    return [Hx, 0, t * H_dot]


def rhs(t, S, B, v):
    """
    [t] is time
    [S] is spin vector
    [B] is B-field function B(r {3D vec}, t) (must return results in nanotesla)
    """
    x = v * t  # t = 0 corresponds to origin
    # mu_n (neutron) / h-bar scaled so that c*S x B (in nT) is in h-bar/second
    c = 2 * -9.162e-2
    return np.cross(c*S, B([x, 0, 0], t))


def naive(f, t_bounds, y0, num_pts):
    """Converges slowly"""
    t0, tf = t_bounds
    y = np.array(y0)
    range, step = np.linspace(t0, tf, num_pts, retstep=True)
    for t in range:
        y += step * f(t, y)
    return y


def run_vladimirskii(Hx, H_dot):
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
    sol = solve_ivp(f_vladimirskii, [-100, 100], [0, 0, 1/2],
                    method="LSODA", rtol=rtol, atol=atol)
    Sf = [sol.y[i][-1] for i in range(3)]

    print(f"Number of f evals: {sol.nfev}")
    print(f"Number of time points: {len(sol.t)}")
    print(f"Final S: {Sf}")
    print(f"Corresponding realignment probability: {100 * (Sf[2] + 1/2)}%")


def run_two_wires(v, d, I1, I2):
    def f_two_wires(t, S): return rhs(
        t, S, lambda r, t: B_two_wires(r, d, I1, I2), v)

    rtol, atol = (1e-8, 1e-8)
    sol = solve_ivp(f_two_wires, [-100, 100], [0, 1/2, 0],
                    method="LSODA", rtol=rtol, atol=atol)
    Sf = [sol.y[i][-1] for i in range(3)]

    print(f"Number of f evals: {sol.nfev}")
    print(f"Number of time points: {len(sol.t)}")
    print(f"Final S: {Sf}")
    # print(f"Naive final S: {naive(f_two_wires, [-100, 100], [0, 1/2, 0], 100000)}")


run_two_wires(1000, 10, 10, -10)
