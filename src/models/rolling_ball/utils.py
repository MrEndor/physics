from typing import Any

import numpy as np


def calculate_ball_metrics(
    x: int, y: int, vx: int, vy: int, omega_z: list[float], t: int
) -> dict[str, Any]:
    final_position = (float(x[-1]), float(y[-1]))
    total_distance = np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

    v_magnitude = np.sqrt(vx**2 + vy**2)
    max_velocity = float(np.max(v_magnitude))
    final_velocity = float(v_magnitude[-1])

    final_angular_velocity = float(omega_z[-1])

    rolling_condition = np.abs(vx - omega_z * 0.1)
    is_rolling_final = float(rolling_condition[-1]) < 1e-3  # noqa: PLR2004

    return {
        "final_position": final_position,
        "total_distance_traveled": float(total_distance),
        "max_velocity": max_velocity,
        "final_velocity": final_velocity,
        "final_angular_velocity": final_angular_velocity,
        "is_rolling_at_end": is_rolling_final,
        "simulation_time": float(t[-1]),
    }
