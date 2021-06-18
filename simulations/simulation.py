import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def B_wire(r, r0, I):
    """
    [r] is field observation point (2D vector)
    [r0] is position of infinite wire (2D vector)
    [I] is current (positive means in +z-direction)
    Gives result in nanotesla
    """
    dist_vec = r - r0
    z_hat = np.array([0, 0, 1])
    direction = np.cross(z_hat, dist_vec)
    B_hat = direction / np.linalg.norm(direction)
    mu2pi = 2e+2  # mu_0 / 2pi (approx), yields result in nanotesla
    dist = np.linalg.norm(dist_vec)
    return mu2pi * I / dist * B_hat


def B_tot(r, wires):
    """
    [r] is field observation point (2D vector)
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
    [r] is field observation point (2D vector)
    [d] is the distance between wires
    [I1], [I2] is current of top and bottom wire respectively
    Gives result in nanotesla
    """
    top_wire_pos = np.array([0, d/2])
    wires = [(top_wire_pos, I1), (-top_wire_pos, I2)]
    return B_tot(r, wires)


def func(v, d, I1, I2, t, S):
    """
    [t] is time
    [S] is spin vector
    [v] is speed
    [d], [I1], [I2] same as in B_two_wires
    """
    x = v * t  # t = 0 corresponds to origin
    # mu_n (neutron) / h-bar scaled so that c*S x B (in nT) is in h-bar/second
    c = 2 * -9.162e-2
    return np.cross(c*S, B_two_wires([x, 0], d, I1, I2))


# r = np.array([1, 4])
# o = np.array([1, 2])

# print(B_wire(r, o, 5))
# print(B_tot(r, [(o, 5)]))
# print(B_two_wires([0, 0], 10, 5, -5))

v = 1000
d = 10
I1 = 10
I2 = -10


def f(t, S): return func(v, d, I1, I2, t, S)


def naive(f, t_bounds, y0, num_pts):
    """Converges slowly"""
    t0, tf = t_bounds
    y = np.array(y0)
    range, step = np.linspace(t0, tf, num_pts, retstep=True)
    for t in range:
        y += step * f(t, y)
    return y


xs = np.linspace(-100, 100, 1000)
By = [B_two_wires([x, 0], d, I1, I2)[0] for x in xs]
plt.plot(xs, By)
# plt.show()

rtol, atol = (1e-8, 1e-8)
sol = solve_ivp(f, [-100, 100], [0, 1/2, 0],
                method="LSODA", rtol=rtol, atol=atol)
Sf = [sol.y[i][-1] for i in range(3)]

print(f"Number of f evals: {sol.nfev}")
print(f"Number of time points: {len(sol.t)}")
print(Sf)

# print(naive(f, [-100, 100], [0, 1/2, 0], 100000))
