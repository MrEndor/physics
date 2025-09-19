import altair as alt
import numpy as np
import pandas as pd

from models.stone_flight.config import DRAG_MODEL_COLORS, DragModel


def create_trajectory_chart(
    trajectories: list[tuple[np.ndarray, np.ndarray, DragModel, float]],
    v0: float,
    angle_deg: float,
) -> alt.LayerChart:
    data_frames = []
    landing_points = []

    for x, y, drag_model, landing_point in trajectories:
        df = pd.DataFrame({
            "x": x,
            "y": y,
            "model": drag_model.value,
            "model_type": "real",
            "color": DRAG_MODEL_COLORS[drag_model],
        })
        data_frames.append(df)
        landing_points.append((landing_point, drag_model.value))

    df_combined = pd.concat(data_frames)

    landing_df = pd.DataFrame([
        {
            "x": point,
            "y": 0,
            "model": model_name,
            "label": f"Падение: {model_name}",
        }
        for point, model_name in landing_points
    ])

    base_chart = alt.Chart(df_combined).encode(
        x=alt.X("x:Q", title="Расстояние, м", scale=alt.Scale(zero=False)),
        y=alt.Y("y:Q", title="Высота, м", scale=alt.Scale(zero=False)),
    )

    lines = base_chart.mark_line().encode(
        color=alt.Color(
            "model:N",
            title="Модель сопротивления",
            scale=alt.Scale(
                domain=[model.value for model in DRAG_MODEL_COLORS],
                range=list(DRAG_MODEL_COLORS.values()),
            ),
            legend=alt.Legend(title="Модели"),
        ),
    )

    points = (
        alt.Chart(landing_df)
        .mark_point(size=80, filled=True)
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color(
                "model:N",
                scale=alt.Scale(
                    domain=[m.value for m in DRAG_MODEL_COLORS],
                    range=list(DRAG_MODEL_COLORS.values()),
                ),
                legend=None,
            ),
            tooltip=["label:N"],
        )
    )

    chart: alt.LayerChart = (
        (lines + points)
        .properties(
            title=f"Сравнение траекторий (v₀={v0} м/с, θ={angle_deg}°)",
        )
        .interactive()
    )

    return chart


def create_velocity_chart(
    dynamics_data: list[tuple[np.ndarray, np.ndarray, DragModel]],
) -> alt.Chart:
    data_frames = []

    for time, velocity, drag_model in dynamics_data:
        df = pd.DataFrame({
            "time": time,
            "velocity": velocity,
            "model": drag_model.value,
            "color": DRAG_MODEL_COLORS[drag_model],
        })
        data_frames.append(df)

    df_combined = pd.concat(data_frames, ignore_index=True)

    chart: alt.Chart = (
        alt.Chart(df_combined)
        .mark_line()
        .encode(
            x=alt.X("time:Q", title="Время, с"),
            y=alt.Y("velocity:Q", title="Скорость, м/с"),
            color=alt.Color(
                "model:N",
                title="Модель сопротивления",
                scale=alt.Scale(
                    domain=[m.value for m in DRAG_MODEL_COLORS],
                    range=list(DRAG_MODEL_COLORS.values()),
                ),
            ),
        )
        .properties(title="Скорость камня во времени")
    )

    return chart


def create_acceleration_chart(
    dynamics_data: list[tuple[np.ndarray, np.ndarray, DragModel]],
) -> alt.Chart:
    data_frames = []

    for time, acceleration, drag_model in dynamics_data:
        df = pd.DataFrame({
            "time": time,
            "acceleration": acceleration,
            "model": drag_model.value,
            "color": DRAG_MODEL_COLORS[drag_model],
        })
        data_frames.append(df)

    df_combined = pd.concat(data_frames)

    chart: alt.Chart = (
        alt.Chart(df_combined)
        .mark_line()
        .encode(
            x=alt.X("time:Q", title="Время, с"),
            y=alt.Y("acceleration:Q", title="Ускорение, м/с²"),
            color=alt.Color(
                "model:N",
                title="Модель сопротивления",
                scale=alt.Scale(
                    domain=[m.value for m in DRAG_MODEL_COLORS],
                    range=list(DRAG_MODEL_COLORS.values()),
                ),
            ),
        )
        .properties(title="Ускорение камня во времени")
    )

    return chart


def create_metrics_bar_chart(
    metrics_data: list[tuple[str, float, str]],
    chart_width: int = 1000,
) -> alt.FacetChart:
    df = pd.DataFrame(metrics_data, columns=["model", "value", "metric"])

    base = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("model:N", title="Модель", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("value:Q", title="Значение"),
            color=alt.Color("model:N", legend=None),
        )
        .properties(width=chart_width // 3 - 10, height=300)
    )

    chart: alt.FacetChart = base.facet(
        column=alt.Column("metric:N", title=None)
    ).properties(title="Сравнение метрик полета")

    return chart
