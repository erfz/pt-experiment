from simulation import *

Sf_two_wires = run_particle(
    [1000, 0, 0],
    np.zeros(3),
    generate_two_wires(10, (10, 0), (-10, 0)),
    [-100, 100],
    [0, 1 / 2, 0],
)
print(f"Final S (two wires): {Sf_two_wires}")

Sf_vlad = run_particle(
    np.empty(3), np.empty(3), [Vladimirskii(10, -41.5)], [-100, 100], [0, 0, 1 / 2]
)
print(f"Final S (Vladimirskii): {Sf_vlad}")
print(f"-> Corresponding realignment probability: {100 * (Sf_vlad[2] + 1/2)}%")

Sf_rand_line = run_two_wires_line(1000, 10, (10, 0), (-10, 0), 100, [-100, 100])
print(f"Final S (rand line): {Sf_rand_line}")

shape = "square"
Sf_shape = run_two_wires_shape_2D(1000, 10, (10, 0), (-10, 0), 100, [-100, 100], shape)
print(f"Final S ({shape}): {Sf_shape}")
