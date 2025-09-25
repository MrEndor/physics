import numpy as np
import streamlit as st

from models.stone_flight.charts import (
    create_acceleration_chart,
    create_metrics_bar_chart,
    create_trajectory_chart,
    create_velocity_chart,
)
from models.stone_flight.config import DragModel
from models.stone_flight.objects import FlightParameters, StoneFlightModel
from models.stone_flight.sidebar import create_stone_flight_sidebar
from models.stone_flight.utils import (
    calculate_flight_metrics,
    create_flight_dataframe,
)


def page() -> None:
    st.set_page_config(page_title="М1. Полёт камня", layout="wide")
    st.title("М1. Полёт камня")
    st.write(
        "Моделирование траектории полёта камня с учётом сопротивления воздуха"
    )

    model = StoneFlightModel()
    params = create_stone_flight_sidebar()

    trajectories = []
    velocity_data = []
    acceleration_data = []
    metrics_list = []

    for drag_model in [
        DragModel.WITHOUT,
        DragModel.LINEAR,
        DragModel.QUADRATIC,
    ]:
        try:
            x, y, t = model.calculate_trajectory(params, drag_model)
            landing_point, flight_time, max_height = calculate_flight_metrics(
                x, y, t
            )

            trajectories.append((x, y, drag_model, landing_point))

            metrics_list.extend([
                (drag_model.value, landing_point, "Дальность (м)"),
                (drag_model.value, flight_time, "Время (с)"),
                (drag_model.value, max_height, "Макс. высота (м)"),
            ])

            df = create_flight_dataframe(x, y, t, drag_model.value)
            velocity_data.append((t, df["velocity"].values, drag_model))
            acceleration_data.append((t, df["acceleration"].values, drag_model))

        except Exception as e:
            st.error(f"Ошибка при расчете для {drag_model.value}: {e}")

    if not trajectories:
        st.error("Не удалось рассчитать ни одну траекторию")
        return

    display_metrics_table(metrics_list)
    display_trajectory_comparison(trajectories, params)
    display_dynamics_comparison(velocity_data, acceleration_data)


def display_metrics_table(
    metrics_data: list[tuple[str, float, str]],
) -> None:
    st.header("Сравнение метрик полета")

    chart = create_metrics_bar_chart(
        metrics_data,
    )

    st.altair_chart(chart, use_container_width=True)


def display_trajectory_comparison(
    trajectories: list[tuple[np.ndarray, np.ndarray, DragModel, float]],
    params: FlightParameters,
) -> None:
    st.header("Сравнение траекторий")

    chart = create_trajectory_chart(
        trajectories=trajectories,
        v0=params.v0,
        angle_deg=params.angle_deg,
    )

    st.altair_chart(chart, use_container_width=True)


def display_dynamics_comparison(
    velocity_data: list[tuple[np.ndarray, np.ndarray, DragModel]],
    acceleration_data: list[tuple[np.ndarray, np.ndarray, DragModel]],
) -> None:
    st.header("Сравнение динамики полета")

    velocity_chart = create_velocity_chart(
        velocity_data,
    )

    acceleration_chart = create_acceleration_chart(
        acceleration_data,
    )

    st.altair_chart(velocity_chart, use_container_width=True)
    st.altair_chart(acceleration_chart, use_container_width=True)
