# %%
from simulation import *
import numpy as np
import matplotlib.pyplot as plt


def sim_vlad_probability(S):
    return 50 * (S[2] + 1)


def theory_vlad_probability(Hx, H_dot):
    return np.exp(-np.pi * Hx * Hx / H_dot * -0.09162) * 100


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
Sf_rand_line = run_two_wires_line(1000, 10, (10, 0), (-10, 0), 100, [-100, 100])
print(f"Final S (rand line): {Sf_rand_line}")

shape = "square"
Sf_shape = run_two_wires_shape_2D(1000, 10, (10, 0), (-10, 0), 100, [-100, 100], shape)
print(f"Final S ({shape}): {Sf_shape}")

# %%
Hx = 10
H_dots = np.linspace(-10, -200, 20)
sim_ps = [
    sim_vlad_probability(
        Particle(
            np.empty(3),
            np.empty(3),
            [Vladimirskii(Hx, H_dot)],
            [-10, 10],
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
incoming_region = OverridingBox(
    [-100, -50, -50], [100, 100, 100], lambda r, t: 3e-6 * np.array([0, 1, 0])
)
superconductor = OverridingBox(
    [0, -50, -50], [8e-4, 100, 100], lambda r, t: np.zeros(3)
)
outgoing_region = OverridingBox(
    [8e-4, -50, -50], [100, 100, 100], lambda r, t: -3e-9 * np.array([0, 1, 0])
)
Sf, ts, spins = Particle(
    [100, 0, 0],
    [0, 0, 0],
    [incoming_region, superconductor, outgoing_region],
    [-1, 1],
    [0, 1, 0],
).simulate_with_output()
print(f"Final S (through superconductor): {Sf}")
plt.plot(ts, list(zip(*spins)), label=("S_x", "S_y", "S_z"))
plt.xlabel("Time (s)")
plt.ylabel("Spin components")
plt.legend()
plt.show()
