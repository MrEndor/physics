import numpy as np
import pandas as pd


def calculate_flight_metrics(
    x: np.ndarray, y: np.ndarray, t: np.ndarray
) -> tuple[float, float, float]:
    valid_indices = np.where(y >= 0)[0]
    if len(valid_indices) == 0:
        return 0.0, 0.0, 0.0

    landing_index = valid_indices[-1]
    landing_point = x[landing_index]
    flight_time = t[landing_index]
    max_height = np.max(y[: landing_index + 1])

    return landing_point, flight_time, max_height


def create_flight_dataframe(
    x: np.ndarray, y: np.ndarray, t: np.ndarray, drag_model: str
) -> pd.DataFrame:
    if len(x) < 2 or len(y) < 2 or len(t) < 2:  # noqa: PLR2004
        return pd.DataFrame({
            "time": t,
            "x": x,
            "y": y,
            "velocity": 0,
            "acceleration": 0,
            "drag_model": drag_model,
        })

    velocity_x = np.gradient(x, t)
    velocity_y = np.gradient(y, t)
    velocity = np.sqrt(velocity_x**2 + velocity_y**2)
    acceleration = np.gradient(velocity, t)

    return pd.DataFrame({
        "time": t,
        "x": x,
        "y": y,
        "velocity": velocity,
        "acceleration": acceleration,
        "drag_model": drag_model,
    })
