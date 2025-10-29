from dataclasses import dataclass
from typing import Any

import altair as alt
import numpy as np
import pandas as pd

from models.billiard.config import PHYSICAL_CONSTANTS, CollisionMethod
from models.billiard.objects import BilliardParameters


@dataclass
class TrajectoryData:
    times: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    params: BilliardParameters
    method: CollisionMethod


def create_billiard_trajectory_chart(
    times: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    params: BilliardParameters,
    method: CollisionMethod,
) -> alt.Chart:
    data = TrajectoryData(times, x1, y1, x2, y2, params, method)
    return _build_trajectory_chart(data)


def _build_trajectory_chart(data: TrajectoryData) -> alt.Chart:
    table_data = pd.DataFrame([
        {"x": 0, "y": 0, "x2": PHYSICAL_CONSTANTS["table_length"], "y2": 0},
        {
            "x": PHYSICAL_CONSTANTS["table_length"],
            "y": 0,
            "x2": PHYSICAL_CONSTANTS["table_length"],
            "y2": PHYSICAL_CONSTANTS["table_width"],
        },
        {
            "x": PHYSICAL_CONSTANTS["table_length"],
            "y": PHYSICAL_CONSTANTS["table_width"],
            "x2": 0,
            "y2": PHYSICAL_CONSTANTS["table_width"],
        },
        {"x": 0, "y": PHYSICAL_CONSTANTS["table_width"], "x2": 0, "y2": 0},
    ])

    table_outline = (
        alt.Chart(table_data)
        .mark_rule(color="brown", strokeWidth=4)
        .encode(
            x=alt.X(
                "x:Q",
                scale=alt.Scale(domain=[0, PHYSICAL_CONSTANTS["table_length"]]),
            ),
            y=alt.Y(
                "y:Q",
                scale=alt.Scale(domain=[0, PHYSICAL_CONSTANTS["table_width"]]),
            ),
            x2="x2:Q",
            y2="y2:Q",
        )
        .properties(width=700, height=350)
    )

    table_fill = (
        alt.Chart(
            pd.DataFrame([
                {
                    "x": 0,
                    "y": 0,
                    "width": PHYSICAL_CONSTANTS["table_length"],
                    "height": PHYSICAL_CONSTANTS["table_width"],
                }
            ])
        )
        .mark_rect(color="darkgreen", opacity=0.2)
        .encode(
            x=alt.X("x:Q"),
            y=alt.Y("y:Q"),
        )
    )

    step = max(1, len(data.times) // 200)
    trajectory_data = []

    for i in range(0, len(data.times), step):
        t = data.times[i]
        trajectory_data.extend([
            {
                "time": t,
                "x": data.x1[i],
                "y": data.y1[i],
                "ball": "Биток",
                "sequence": i,
            },
            {
                "time": t,
                "x": data.x2[i],
                "y": data.y2[i],
                "ball": "Прицельный",
                "sequence": i,
            },
        ])

    df_trajectory = pd.DataFrame(trajectory_data)

    trajectory_lines = (
        alt.Chart(df_trajectory)
        .mark_line(strokeWidth=3, opacity=0.8)
        .encode(
            x=alt.X("x:Q", title="Расстояние по длине стола (м)"),
            y=alt.Y("y:Q", title="Расстояние по ширине стола (м)"),
            color=alt.Color(
                "ball:N",
                title="Шар",
                scale=alt.Scale(
                    domain=["Биток", "Прицельный"], range=["#000000", "#FF4444"]
                ),
            ),
            order=alt.Order("sequence:O"),
            tooltip=["ball:N", "time:Q", "x:Q", "y:Q"],
        )
    )

    initial_positions = pd.DataFrame([
        {
            "x": data.params.cue_x,
            "y": data.params.cue_y,
            "ball": "Биток",
            "type": "Начальная позиция",
        },
        {
            "x": data.params.target_x,
            "y": data.params.target_y,
            "ball": "Прицельный",
            "type": "Начальная позиция",
        },
    ])

    initial_chart = (
        alt.Chart(initial_positions)
        .mark_circle(size=400, opacity=0.9, stroke="black", strokeWidth=3)
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color(
                "ball:N",
                scale=alt.Scale(
                    domain=["Биток", "Прицельный"], range=["#FFFFFF", "#FF4444"]
                ),
                legend=None,
            ),
            tooltip=["ball:N", "type:N", "x:Q", "y:Q"],
        )
    )

    final_positions = pd.DataFrame([
        {
            "x": data.x1[-1],
            "y": data.y1[-1],
            "ball": "Биток",
            "type": "Конечная позиция",
        },
        {
            "x": data.x2[-1],
            "y": data.y2[-1],
            "ball": "Прицельный",
            "type": "Конечная позиция",
        },
    ])

    final_chart = (
        alt.Chart(final_positions)
        .mark_square(size=300, opacity=0.9, stroke="black", strokeWidth=3)
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color(
                "ball:N",
                scale=alt.Scale(
                    domain=["Биток", "Прицельный"], range=["#CCCCCC", "#CC2222"]
                ),
                legend=None,
            ),
            tooltip=["ball:N", "type:N", "x:Q", "y:Q"],
        )
    )

    chart: alt.Chart = (
        alt.layer(
            table_fill,
            table_outline,
            trajectory_lines,
            initial_chart,
            final_chart,
        )
        .resolve_scale(color="independent")
        .properties(
            title=alt.TitleParams(
                text=[
                    f"Траектории движения шаров (метод: {data.method.value})",
                    "Кружки - начальные позиции, квадраты - конечные позиции",
                ],
                fontSize=14,
                anchor="start",
            )
        )
    )

    return chart


def create_detailed_collision_analysis_chart(
    analysis: dict[str, Any],
) -> alt.Chart:
    if "time_series" not in analysis:
        chart: alt.Chart = alt.Chart().mark_text(text="Нет данных для анализа")
        return chart

    times = analysis["time_series"]["times"]
    ke = analysis["time_series"]["kinetic_energy"]
    momentum = analysis["time_series"]["momentum"]

    energy_data = pd.DataFrame({"time": times, "energy": ke})

    energy_chart = (
        alt.Chart(energy_data)
        .mark_line(strokeWidth=3, color="red")
        .encode(
            x=alt.X("time:Q", title="Время (с)"),
            y=alt.Y("energy:Q", title="Кинетическая энергия (Дж)"),
            tooltip=["time:Q", "energy:Q"],
        )
        .properties(
            title="Сохранение кинетической энергии", width=600, height=200
        )
    )

    momentum_data = pd.DataFrame({"time": times, "momentum": momentum})

    momentum_chart = (
        alt.Chart(momentum_data)
        .mark_line(strokeWidth=3, color="blue")
        .encode(
            x=alt.X("time:Q", title="Время (с)"),
            y=alt.Y("momentum:Q", title="Импульс (кг⋅м/с)"),
            tooltip=["time:Q", "momentum:Q"],
        )
        .properties(title="Сохранение импульса", width=600, height=200)
    )

    vchart: alt.Chart = alt.vconcat(energy_chart, momentum_chart).resolve_scale(
        y="independent"
    )

    return vchart


def create_velocity_analysis_chart(analysis: dict[str, Any]) -> alt.Chart:
    if "time_series" not in analysis:
        chart: alt.Chart = alt.Chart().mark_text(
            text="Нет данных для анализа скоростей"
        )
        return chart

    times = analysis["time_series"]["times"]
    v_cue = analysis["time_series"]["velocities"]["cue"]
    v_target = analysis["time_series"]["velocities"]["target"]
    a_cue = analysis["time_series"]["accelerations"]["cue"]
    a_target = analysis["time_series"]["accelerations"]["target"]

    velocity_data = []
    for i, t in enumerate(times):
        velocity_data.extend([
            {"time": t, "value": v_cue[i], "ball": "Биток"},
            {"time": t, "value": v_target[i], "ball": "Прицельный"},
        ])

    df_vel = pd.DataFrame(velocity_data)

    velocity_chart = (
        alt.Chart(df_vel)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("time:Q", title="Время (с)"),
            y=alt.Y("value:Q", title="Скорость (м/с)"),
            color=alt.Color(
                "ball:N",
                title="Шар",
                scale=alt.Scale(
                    domain=["Биток", "Прицельный"], range=["#0066CC", "#CC0000"]
                ),
            ),
            tooltip=["ball:N", "time:Q", "value:Q"],
        )
        .properties(title="Изменение скоростей шаров", width=600, height=200)
    )

    accel_data = []
    for i, t in enumerate(times):
        accel_data.extend([
            {"time": t, "value": a_cue[i], "ball": "Биток"},
            {"time": t, "value": a_target[i], "ball": "Прицельный"},
        ])

    df_accel = pd.DataFrame(accel_data)

    acceleration_chart = (
        alt.Chart(df_accel)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("time:Q", title="Время (с)"),
            y=alt.Y("value:Q", title="Ускорение (м/с²)"),
            color=alt.Color(
                "ball:N",
                scale=alt.Scale(
                    domain=["Биток", "Прицельный"], range=["#0066CC", "#CC0000"]
                ),
                legend=None,
            ),
            tooltip=["ball:N", "time:Q", "value:Q"],
        )
        .properties(title="Ускорения шаров", width=600, height=200)
    )

    vchart: alt.Chart = alt.vconcat(
        velocity_chart, acceleration_chart
    ).resolve_scale(y="independent")

    return vchart
