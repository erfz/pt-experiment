# %%
from simulation import *
import numpy as np
import matplotlib.pyplot as plt

x_hat = np.array([1.0, 0, 0])
y_hat = np.array([0, 1.0, 0])


def sim_vlad_probability(S):
    return 50 * (S[2] + 1)


def theory_vlad_probability(Hx, H_dot):
    return np.exp(-np.pi * Hx * Hx / H_dot * -0.09162) * 100


def max_time_step(v_normal, domain_len):
    return domain_len / v_normal / 2


# %%
Sf, ts, spins = Particle(
    [1000, 0, 0],
    np.zeros(3),
    generate_two_wires(10, (10, 0), (-10, 0)),
    [-100, 100],
    [0, 1, 0],
).simulate_with_output()
print(f"Final S (two wires): {Sf}")
plt.plot(ts, list(zip(*spins)), label=("S_x", "S_y", "S_z"))
plt.xlabel("Time (s)")
plt.ylabel("Spin components")
plt.legend()
plt.show()

# %%
Hx = 10
H_dot = -41.5
Sf, ts, spins = Particle(
    np.empty(3), np.empty(3), [Vladimirskii(Hx, H_dot)], [-10, 10], [0, 0, 1]
).simulate_with_output()
print(f"Final S (Vladimirskii): {Sf}")
print(f"Corresponding realignment probability: {sim_vlad_probability(Sf)}%")
print(f"Theoretical realignment probability: {theory_vlad_probability(Hx, H_dot)}%")
plt.plot(ts, list(zip(*spins)), label=("S_x", "S_y", "S_z"))
plt.xlabel("Time (s)")
plt.ylabel("Spin components")
plt.legend()
plt.show()

# %%
shape = "line"
Sf = rand_shape_sim(
    1000, 10, generate_two_wires(10, (10, 0), (-10, 0)), 100, [-100, 100], shape
)
print(f"Final S ({shape}): {Sf}")

shape = "square"
Sf = rand_shape_sim(
    1000, 10, generate_two_wires(10, (10, 0), (-10, 0)), 100, [-100, 100], shape
)
print(f"Final S ({shape}): {Sf}")

# %%
def sim_time(Hx, H_dot):
    t = Hx / H_dot * -100
    return [-t, t]


Hx = 10
H_dots = np.linspace(-10, -200, 20)
sim_ps = [
    sim_vlad_probability(
        Particle(
            np.empty(3),
            np.empty(3),
            [Vladimirskii(Hx, H_dot)],
            sim_time(Hx, H_dot),
            [0, 0, 1],
        ).simulate()
    )
    for H_dot in H_dots
]

vlad_ps = [theory_vlad_probability(Hx, H_dot) for H_dot in H_dots]

plt.plot(H_dots, sim_ps, label="Simulated")
plt.plot(H_dots, vlad_ps, label="Vladimirskii prediction")
plt.xlabel("H_dot (nT/s)")
plt.ylabel("Probability of non-adiabatic realignment")
plt.legend()
plt.show()

# %%
Sf, ts, spins = Particle(
    [5, 0, 0],
    [0, 4.5 / 100, 0],
    generate_two_wires(5 / 100, (0.3, 0), (0.3, 0)),
    [-100, 100],
).simulate_with_output()
print(f"Final S (two wires, particle offset closer to top wire): {Sf}")
plt.plot(ts, list(zip(*spins)), label=("S_x", "S_y", "S_z"))
plt.xlabel("Time (s)")
plt.ylabel("Spin components")
plt.legend()
plt.show()

# %%
Sf, ts, spins = Particle(
    [100, 0, 0],
    [0, 4.5 / 100, 0],
    generate_two_wires(5 / 100, (0.3, 0), (0.3, 0)),
    [-100, 100],
).simulate_with_output()
print(f"Final S (two wires, particle offset closer to top wire, faster): {Sf}")
plt.plot(ts, list(zip(*spins)), label=("S_x", "S_y", "S_z"))
plt.xlabel("Time (s)")
plt.ylabel("Spin components")
plt.legend()
plt.show()

# %%
domain_len = 1e-5
B_metglas = 5e8  # corresponds to 5e8 nT = 0.5 T
vx = 100

metglas = Metglas(
    [0, -0.05, -50],
    -B_metglas * y_hat,
    x_hat,
    0.82,
    [domain_len, domain_len, 100],
    [10, 10000, 1],
)

Sf, ts, spins = Particle(
    [vx, 0, 0],
    [0, 0, 0],
    [metglas],
    [-0.00000025, 0.00000125],
    y_hat,
    max_time_step(vx, domain_len),
).simulate_with_output()
print(f"Number of t evals: {len(ts)}")
print(f"Final S (through Metglas): {Sf}")
plt.plot(ts, list(zip(*spins)), label=("S_x", "S_y", "S_z"))
plt.xlabel("Time (s)")
plt.ylabel("Spin components")
plt.legend()
plt.show()

shape = "square"
Sf = rand_shape_sim(
    vx,
    0.1,
    [metglas],
    1000,
    [-0.00000025, 0.00000125],
    shape,
    y_hat,
    max_time_step(vx, domain_len),
)
print(f"Final S (thru Metglas, {shape}): {Sf}")

# %%
domain_len = 1e-5
B_metglas = 5e8  # corresponds to 5e8 nT = 0.5 T
vx = 100


def metglas(sat):
    return Metglas(
        [0, -0.05, -50],
        -B_metglas * y_hat,
        x_hat,
        sat,
        [domain_len, domain_len, 100],
        [10, 10000, 1],
    )


sats = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 1]

spins_y = [
    rand_shape_sim(
        vx,
        0.1,
        [metglas(sat)],
        1000,
        [-0.00000025, 0.00000125],
        "square",
        y_hat,
        max_time_step(vx, domain_len),
    )[1]
    for sat in sats
]

plt.scatter(sats, spins_y)
plt.xlabel("Saturation")
plt.ylabel("S_y")
plt.show()

# %%
B_metglas = 5e8  # corresponds to 5e8 nT = 0.5 T
vx = 100
metglas_len = 0.0001  # 0.1 mm
metglas_width = 0.1  # 0.1 m


def metglas(num_layers):
    domain_len = metglas_len / num_layers
    return Metglas(
        [0, -0.05, -50],
        -B_metglas * y_hat,
        x_hat,
        0.8,
        [domain_len, domain_len, 100],
        [num_layers, int(metglas_width / domain_len), 1],
    )


Ns = [1, 2, 5, 8, 10, 12, 15, 20]

spins_y = [
    rand_shape_sim(
        vx,
        0.1,
        [metglas(N)],
        1000,
        [-0.00000025, 0.00000125],
        "square",
        y_hat,
        max_time_step(vx, metglas_len / N),
    )[1]
    for N in Ns
]

print(spins_y)

plt.scatter(Ns, spins_y)
plt.xlabel("Number of domains passed thru per particle")
plt.ylabel("S_y")
plt.show()

# %%
