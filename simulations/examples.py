# %%
import matplotlib.pyplot as plt
import numpy as np

from simulation import *

x_hat = np.array([1.0, 0, 0])
y_hat = np.array([0, 1.0, 0])


def sim_vlad_probability(S):
    return 50 * (S[2] + 1)


def theory_vlad_probability(Hx, H_dot):
    return np.exp(-np.pi * Hx * Hx / H_dot * -0.09162) * 100


def max_time_step(v_normal, min_cell_len):
    return min_cell_len / v_normal / 2


def generate_metglas(sat, cell_length_range, B, thickness):
    return Metglas(
        (0, -0.1 / 2, -100 / 2),
        (thickness, 0.1, 100),
        -B * 1.0e9 * y_hat,
        x_hat,
        sat,
        cell_length_range,
    )


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
Sf, ts, spins = Particle(
    [vx := 100, 0, 0],
    [0, 0, 0],
    [
        generate_metglas(
            sat := 0.01,
            domain_thickness := (4e-6, 6e-6),
            B := 0.01,
            thickness := 25e-6,
        )
    ],
    [-0.00000005, 0.0000003],
    y_hat,
    max_time_step(vx, domain_thickness[0]),
).simulate_with_output()
print(f"Number of t evals: {len(ts)}")
print(f"Final S (through Metglas): {Sf}")
plt.plot(ts, list(zip(*spins)), label=("S_x", "S_y", "S_z"))
plt.title(
    f"""Particle passing thru Metglas:
    Saturation = {sat}
    Total thickness = {thickness} m
    Domain thickness = {tuple((round(x, 10) for x in domain_thickness))} m
    Field strength = {B} T"""
)
plt.xlabel("Time (s)")
plt.ylabel("Spin components")
plt.legend()
plt.show()

# %%
shape = "square"
Sf = rand_shape_sim(
    vx := 100,
    0.1,
    [generate_metglas(0.82, domain_thickness := (0.5 * 1e-5, 1.5 * 1e-5), 0.5, 1e-4)],
    50,
    [-0.00000025, 0.00000125],
    shape,
    y_hat,
    max_time_step(vx, domain_thickness[0]),
)
print(f"Final S (thru Metglas, {shape}): {Sf}")

# %%
sats = [
    0.01,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.98,
    1,
]
Bs = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0, 1.5]

spins_y = [
    [
        rand_shape_sim(
            vx := 100,
            0.1,
            [
                generate_metglas(
                    sat,
                    domain_thickness := (4e-6, 6e-6),
                    B,
                    thickness := 25e-6,
                )
            ],
            1000,
            [-0.00000005, 0.0000003],
            "square",
            y_hat,
            max_time_step(vx, domain_thickness[0]),
        )[1]
        for sat in sats
    ]
    for B in Bs
]

plt.plot(sats, list(zip(*spins_y)), label=[f"{B} T" for B in Bs])
plt.title(
    f"""Polarization vs Saturation:
    Speed = {vx} m/s
    Total thickness = {thickness} m
    Domain thickness = {tuple((round(x, 10) for x in domain_thickness))} m
    """
)
plt.xlabel("Saturation")
plt.ylabel("S_y")
plt.legend()
plt.show()

# %%
bases = [3e-6, 6e-6, 8e-6, 10e-6, 12e-6]
offset = 2e-6

spins_y = [
    rand_shape_sim(
        vx := 100,
        0.1,
        [
            generate_metglas(
                sat := 0.82,
                domain_thickness := (base - offset, base + offset),
                B := 0.5,
                thickness := 25e-6,
            )
        ],
        1000,
        [-0.00000005, 0.0000003],
        "square",
        y_hat,
        max_time_step(vx, domain_thickness[0]),
    )[1]
    for base in bases
]

plt.plot(bases, spins_y)
plt.title(
    f"""Polarization vs Base Domain Thickness:
    Base thickness offset = {offset}
    Saturation = {sat}
    Total thickness = {thickness} m
    Speed = {vx} m/s
    Field strength = {B} T"""
)
plt.xlabel(f"Base domain thickness (range = base +/- offset)")
plt.ylabel("S_y")
plt.show()

# %%
