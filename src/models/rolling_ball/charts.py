import altair as alt
import pandas as pd


def create_position_chart(x: float, y: float, t: float) -> alt.Chart:
    df = pd.DataFrame({
        "x": x,
        "y": y,
        "t": t,
    })

    return (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=2)
        .add_params(alt.selection_interval(bind="scales"))
        .encode(
            x=alt.X("x:Q", title="Позиция X (м)"),
            y=alt.Y("y:Q", title="Позиция Y (м)"),
            color=alt.Color(
                "t:Q", title="Время (с)", scale=alt.Scale(scheme="viridis")
            ),
            tooltip=["x:Q", "y:Q", "t:Q"],
        )
        .properties(title="Траектория движения шара", width=400, height=300)
    )


def create_velocity_chart(
    vx: float, vy: float, omega_z: float, t: float
) -> alt.Chart:
    df = pd.DataFrame({
        "t": t,
        "Скорость X": vx,
        "Скорость Y": vy,
        "Угловая скорость": omega_z,
    })

    df_melted = df.melt("t", var_name="Тип скорости", value_name="Значение")

    return (
        alt.Chart(df_melted)
        .mark_line(strokeWidth=2)
        .add_params(alt.selection_interval(bind="scales"))
        .encode(
            x=alt.X("t:Q", title="Время (с)"),
            y=alt.Y("Значение:Q", title="Скорость"),
            color=alt.Color("Тип скорости:N"),
            tooltip=["t:Q", "Тип скорости:N", "Значение:Q"],
        )
        .properties(
            title="Изменение скоростей во времени", width=400, height=300
        )
    )


def create_energy_chart(model, params, vx, vy, omega_z, t):
    kinetic_translational = 0.5 * params.mass * (vx**2 + vy**2)
    kinetic_rotational = (
        0.5 * (2 / 5 * params.mass * params.radius**2) * omega_z**2
    )
    total_kinetic = kinetic_translational + kinetic_rotational

    df = pd.DataFrame({
        "t": t,
        "Поступательная кинетическая": kinetic_translational,
        "Вращательная кинетическая": kinetic_rotational,
        "Полная кинетическая": total_kinetic,
    })

    df_melted = df.melt("t", var_name="Тип энергии", value_name="Энергия (Дж)")

    return (
        alt.Chart(df_melted)
        .mark_line(strokeWidth=2)
        .add_params(alt.selection_interval(bind="scales"))
        .encode(
            x=alt.X("t:Q", title="Время (с)"),
            y=alt.Y("Энергия (Дж):Q", title="Энергия (Дж)"),
            color=alt.Color("Тип энергии:N"),
            tooltip=["t:Q", "Тип энергии:N", "Энергия (Дж):Q"],
        )
        .properties(title="Анализ энергии системы", width=400, height=300)
    )
