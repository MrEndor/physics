import altair as alt
import numpy as np
import pandas as pd


def create_phase_portrait_chart(vx, omega_z, t):
    df = pd.DataFrame({
        "vx": vx,
        "omega_z": omega_z,
        "t": t,
    })

    chart = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=2)
        .add_params(alt.selection_interval(bind="scales"))
        .encode(
            x=alt.X("vx:Q", title="Линейная скорость vx (м/с)"),
            y=alt.Y("omega_z:Q", title="Угловая скорость ω (рад/с)"),
            color=alt.Color(
                "t:Q", title="Время (с)", scale=alt.Scale(scheme="viridis")
            ),
            tooltip=["vx:Q", "omega_z:Q", "t:Q"],
        )
        .properties(
            title="Фазовый портрет движения шара", width=400, height=300
        )
    )

    rolling_condition_data = pd.DataFrame({
        "vx_roll": np.linspace(df["vx"].min(), df["vx"].max(), 100),
    })
    rolling_condition_data["omega_roll"] = (
        rolling_condition_data["vx_roll"] / 0.1
    )

    rolling_line = (
        alt.Chart(rolling_condition_data)
        .mark_line(color="red", strokeDash=[5, 5], strokeWidth=2)
        .encode(x="vx_roll:Q", y="omega_roll:Q")
    )

    return chart + rolling_line


def create_friction_force_chart(model, params, vx, vy, omega_z, t):
    friction_forces = []
    modes = []

    for i in range(len(t)):
        fx, fy, mode = model.calculate_friction_force(
            vx[i], vy[i], omega_z[i], params
        )
        friction_magnitude = np.sqrt(fx**2 + fy**2)
        friction_forces.append(friction_magnitude)
        modes.append(mode)

    df = pd.DataFrame({
        "t": t,
        "friction_force": friction_forces,
        "mode": modes,
    })

    return (
        alt.Chart(df)
        .mark_line(strokeWidth=2)
        .add_params(alt.selection_interval(bind="scales"))
        .encode(
            x=alt.X("t:Q", title="Время (с)"),
            y=alt.Y("friction_force:Q", title="Сила трения (Н)"),
            color=alt.Color("mode:N", title="Режим движения"),
            tooltip=["t:Q", "friction_force:Q", "mode:N"],
        )
        .properties(
            title="Изменение силы трения во времени", width=500, height=300
        )
    )


def create_rolling_condition_chart(vx, omega_z, params, t):
    radius = params.radius
    rolling_violation = np.abs(vx - omega_z * radius)

    df = pd.DataFrame({
        "t": t,
        "rolling_violation": rolling_violation,
        "log_violation": np.log10(rolling_violation + 1e-10),
    })

    chart1 = (
        alt.Chart(df)
        .mark_line(strokeWidth=2, color="blue")
        .encode(
            x=alt.X("t:Q", title="Время (с)"),
            y=alt.Y("rolling_violation:Q", title="|v - ωR| (м/с)"),
            tooltip=["t:Q", "rolling_violation:Q"],
        )
        .properties(title="Нарушение условия качения", width=400, height=250)
    )

    chart2 = (
        alt.Chart(df)
        .mark_line(strokeWidth=2, color="red")
        .encode(
            x=alt.X("t:Q", title="Время (с)"),
            y=alt.Y("log_violation:Q", title="log₁₀|v - ωR|"),
            tooltip=["t:Q", "log_violation:Q"],
        )
        .properties(
            title="Нарушение условия качения (лог. шкала)",
            width=400,
            height=250,
        )
    )

    return alt.vconcat(chart1, chart2)


def create_acceleration_components_chart(model, params, vx, vy, omega_z, t):
    dt = t[1] - t[0] if len(t) > 1 else params.dt

    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    alpha = np.gradient(omega_z, dt)

    df = pd.DataFrame({
        "t": t,
        "Ускорение X": ax,
        "Ускорение Y": ay,
        "Угловое ускорение": alpha,
    })

    df_melted = df.melt("t", var_name="Тип ускорения", value_name="Ускорение")

    return (
        alt.Chart(df_melted)
        .mark_line(strokeWidth=2)
        .add_params(alt.selection_interval(bind="scales"))
        .encode(
            x=alt.X("t:Q", title="Время (с)"),
            y=alt.Y("Ускорение:Q", title="Ускорение"),
            color=alt.Color("Тип ускорения:N"),
            tooltip=["t:Q", "Тип ускорения:N", "Ускорение:Q"],
        )
        .properties(
            title="Компоненты ускорения во времени", width=500, height=300
        )
    )


def create_energy_balance_chart(model, params, x, y, vx, vy, omega_z, t):
    mass = params.mass
    radius = params.radius

    kinetic_trans = 0.5 * mass * (vx**2 + vy**2)
    moment_inertia = 2 / 5 * mass * radius**2
    kinetic_rot = 0.5 * moment_inertia * omega_z**2

    height = x * np.sin(np.radians(params.theta_degrees))
    potential = mass * model.g * height

    total_mechanical = kinetic_trans + kinetic_rot + potential

    energy_loss = total_mechanical[0] - total_mechanical

    df = pd.DataFrame({
        "t": t,
        "Поступательная кинетическая": kinetic_trans,
        "Вращательная кинетическая": kinetic_rot,
        "Потенциальная": potential,
        "Полная механическая": total_mechanical,
        "Потери энергии": energy_loss,
    })

    df_melted = df.melt("t", var_name="Тип энергии", value_name="Энергия (Дж)")

    return (
        alt.Chart(df_melted)
        .mark_line(strokeWidth=2)
        .add_params(alt.selection_interval(bind="scales"))
        .encode(
            x=alt.X("t:Q", title="Время (с)"),
            y=alt.Y("Энергия (Дж):Q", title="Энергия (Дж)"),
            color=alt.Color(
                "Тип энергии:N", scale=alt.Scale(scheme="category10")
            ),
            tooltip=["t:Q", "Тип энергии:N", "Энергия (Дж):Q"],
        )
        .properties(
            title="Детальный баланс энергии системы", width=600, height=400
        )
    )


def create_trajectory_with_vectors(x, y, vx, vy, t, step=10):
    trajectory_df = pd.DataFrame({"x": x, "y": y, "t": t})

    trajectory = (
        alt.Chart(trajectory_df)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("x:Q", title="Позиция X (м)"),
            y=alt.Y("y:Q", title="Позиция Y (м)"),
            color=alt.Color(
                "t:Q", title="Время (с)", scale=alt.Scale(scheme="viridis")
            ),
            tooltip=["x:Q", "y:Q", "t:Q"],
        )
    )

    vector_indices = range(0, len(x), step)
    vectors_df = pd.DataFrame({
        "x_start": [x[i] for i in vector_indices],
        "y_start": [y[i] for i in vector_indices],
        "x_end": [x[i] + 0.1 * vx[i] for i in vector_indices],
        "y_end": [y[i] + 0.1 * vy[i] for i in vector_indices],
        "t_vector": [t[i] for i in vector_indices],
    })

    vectors = (
        alt.Chart(vectors_df)
        .mark_rule(strokeWidth=2, color="red")
        .encode(
            x="x_start:Q",
            y="y_start:Q",
            x2="x_end:Q",
            y2="y_end:Q",
            tooltip=["t_vector:Q"],
        )
    )

    arrows = (
        alt.Chart(vectors_df)
        .mark_point(shape="triangle-right", size=50, color="red", filled=True)
        .encode(x="x_end:Q", y="y_end:Q", tooltip=["t_vector:Q"])
    )

    return (
        (trajectory + vectors + arrows)
        .properties(
            title="Траектория движения с векторами скорости",
            width=500,
            height=400,
        )
        .resolve_scale(color="independent")
    )


def create_comparative_analysis_chart(results_dict):
    comparison_data = []

    for param_name, (x, _y, vx, vy, _omega_z, t) in results_dict.items():
        final_distance = x[-1]
        max_velocity = np.max(np.sqrt(vx**2 + vy**2))
        final_velocity = np.sqrt(vx[-1] ** 2 + vy[-1] ** 2)
        total_time = t[-1]

        comparison_data.extend([
            {
                "Параметр": param_name,
                "Метрика": "Конечное расстояние (м)",
                "Значение": final_distance,
            },
            {
                "Параметр": param_name,
                "Метрика": "Макс. скорость (м/с)",
                "Значение": max_velocity,
            },
            {
                "Параметр": param_name,
                "Метрика": "Конечная скорость (м/с)",
                "Значение": final_velocity,
            },
            {
                "Параметр": param_name,
                "Метрика": "Время симуляции (с)",
                "Значение": total_time,
            },
        ])

    df = pd.DataFrame(comparison_data)

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Параметр:N", title="Вариант параметров"),
            y=alt.Y("Значение:Q", title="Значение метрики"),
            color=alt.Color("Параметр:N"),
            column=alt.Column("Метрика:N", title=""),
            tooltip=["Параметр:N", "Метрика:N", "Значение:Q"],
        )
        .properties(
            width=150, height=200, title="Сравнительный анализ результатов"
        )
    )
