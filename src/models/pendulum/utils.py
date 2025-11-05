import numpy as np

MIN_ZERO_CROSSINGS = 2
MIN_DATA_POINTS_FOR_DECAY = 100
EXCELLENT_THRESHOLD = 1e-6
GOOD_THRESHOLD = 1e-4
ACCEPTABLE_THRESHOLD = 1e-2


def calculate_pendulum_metrics(
    phi: np.ndarray, omega: np.ndarray, t: np.ndarray
) -> dict:
    if len(phi) == 0:
        return {"error": "Нет данных для анализа"}

    max_amplitude = float(np.max(np.abs(phi)))

    zero_crossings = []
    for i in range(1, len(phi)):
        if phi[i - 1] * phi[i] < 0 and omega[i] < 0:
            t_cross = t[i - 1] - phi[i - 1] * (t[i] - t[i - 1]) / (
                phi[i] - phi[i - 1]
            )
            zero_crossings.append(t_cross)

    period = 0.0
    if len(zero_crossings) >= MIN_ZERO_CROSSINGS:
        periods = np.diff(zero_crossings)
        period = float(np.mean(periods))

    max_omega = float(np.max(np.abs(omega)))

    kinetic_max = float(np.max(0.5 * omega**2))
    potential_max = float(max_amplitude**2 / 2)

    amplitude_decay_rate = 0.0
    if len(phi) > MIN_DATA_POINTS_FOR_DECAY:
        peaks = [
            (t[i], phi[i])
            for i in range(1, len(phi) - 1)
            if phi[i - 1] < phi[i] > phi[i + 1] and phi[i] > 0
        ]

        if len(peaks) >= MIN_ZERO_CROSSINGS:
            peak_times = np.array([p[0] for p in peaks])
            peak_values = np.array([p[1] for p in peaks])

            if np.all(peak_values > 0):
                coeffs = np.polyfit(peak_times, np.log(peak_values), 1)
                amplitude_decay_rate = float(-coeffs[0])

    return {
        "max_amplitude_rad": max_amplitude,
        "max_amplitude_deg": float(np.degrees(max_amplitude)),
        "period_s": period,
        "max_angular_velocity": max_omega,
        "kinetic_energy_max": kinetic_max,
        "potential_energy_max": potential_max,
        "amplitude_decay_rate": amplitude_decay_rate,
        "simulation_time": float(t[-1]) if len(t) > 0 else 0.0,
        "data_points": len(t),
    }


def theoretical_period(
    length: float, g: float, phi0_rad: float, pendulum_type: str = "uniform_rod"
) -> float:
    if pendulum_type == "uniform_rod":
        omega0 = np.sqrt(3 * g / (2 * length))
    elif pendulum_type == "point_mass":
        omega0 = np.sqrt(g / length)
    else:
        omega0 = np.sqrt(3 * g / (2 * length))

    t0 = 2 * np.pi / omega0

    amplitude_correction = 1 + phi0_rad**2 / 16

    return t0 * amplitude_correction


def analyze_energy_conservation(
    kinetic: np.ndarray, potential: np.ndarray, total: np.ndarray
) -> dict:
    if len(total) == 0:
        return {"error": "Нет данных для анализа энергии"}

    e0 = total[0]
    e_final = total[-1]

    relative_change = (e_final - e0) / e0 if e0 != 0 else 0

    max_deviation = np.max(np.abs(total - e0)) / e0 if e0 != 0 else 0

    mean_energy = float(np.mean(total))

    energy_std = float(np.std(total))

    if max_deviation < EXCELLENT_THRESHOLD:
        quality = "excellent"
    elif max_deviation < GOOD_THRESHOLD:
        quality = "good"
    elif max_deviation < ACCEPTABLE_THRESHOLD:
        quality = "acceptable"
    else:
        quality = "poor"

    return {
        "initial_energy": float(e0),
        "final_energy": float(e_final),
        "relative_change": float(relative_change),
        "max_deviation": float(max_deviation),
        "mean_energy": mean_energy,
        "energy_std": energy_std,
        "conservation_quality": quality,
    }
