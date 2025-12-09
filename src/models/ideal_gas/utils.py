import numpy as np
import pandas as pd


def calculate_maxwell_distribution(
    speeds: np.ndarray, temperature: float, particle_mass: float
) -> np.ndarray:
    k_b = 1.38e-23

    if temperature <= 0 or particle_mass <= 0:
        return np.zeros_like(speeds)

    sqrt_term = np.sqrt(particle_mass / (2 * np.pi * k_b * temperature))
    norm_factor = 4 * np.pi * (sqrt_term**3)
    exp_arg = -particle_mass * speeds**2 / (2 * k_b * temperature)

    return norm_factor * speeds**2 * np.exp(exp_arg)


def calculate_barometric_formula(
    heights: np.ndarray, scale_height: float
) -> np.ndarray:
    if scale_height <= 0:
        return np.zeros_like(heights)

    return (1.0 / scale_height) * np.exp(-heights / scale_height)


def calculate_speed_from_velocity(velocities: np.ndarray) -> np.ndarray:
    return np.linalg.norm(velocities, axis=1)


def create_statistics_dataframe(
    positions: np.ndarray,
    velocities: np.ndarray,
    particle_mass: float,
    times: np.ndarray,
) -> pd.DataFrame:
    num_steps = len(times)

    data = {
        "time": [],
        "kinetic_energy": [],
        "potential_energy": [],
        "total_energy": [],
        "temperature": [],
        "mean_height": [],
    }

    g = 9.81
    k_b = 1.38e-23

    for step in range(num_steps):
        pos = positions[step]
        vel = velocities[step]

        kinetic_energy = 0.5 * particle_mass * np.sum(vel**2)

        potential_energy = particle_mass * g * np.sum(pos[:, 2])

        kinetic_per_particle = kinetic_energy / len(pos)
        temperature = 2.0 * kinetic_per_particle / (3.0 * k_b)

        data["time"].append(times[step])
        data["kinetic_energy"].append(kinetic_energy)
        data["potential_energy"].append(potential_energy)
        data["total_energy"].append(kinetic_energy + potential_energy)
        data["temperature"].append(temperature)
        data["mean_height"].append(np.mean(pos[:, 2]))

    return pd.DataFrame(data)
