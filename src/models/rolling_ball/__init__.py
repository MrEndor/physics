import streamlit as st

from models.rolling_ball.advanced_charts import (
    create_acceleration_components_chart,
    create_energy_balance_chart,
    create_friction_force_chart,
    create_phase_portrait_chart,
    create_rolling_condition_chart,
    create_trajectory_with_vectors,
)
from models.rolling_ball.charts import (
    create_energy_chart,
    create_position_chart,
    create_velocity_chart,
)
from models.rolling_ball.objects import BallModel
from models.rolling_ball.sidebar import create_ball_sidebar


def page() -> None:
    st.set_page_config(page_title="М4. Шар на столе", layout="wide")
    st.title("М4. Шар на столе")
    st.write(
        "Моделирование движения шара по"
        " шероховатой поверхности с учётом вращения"
    )

    model = BallModel()

    analysis_type = st.sidebar.radio(
        "Тип анализа:",
        ["Основной анализ", "Расширенная визуализация"],
    )
    params = create_ball_sidebar()

    try:
        if analysis_type == "Основной анализ":
            display_basic_analysis(model, params)
        elif analysis_type == "Расширенная визуализация":
            display_advanced_analysis(model, params)

    except Exception as e:
        st.error(f"Ошибка при моделировании: {e}")


def display_basic_analysis(model, params):
    x, y, vx, vy, omega_z, t = model.simulate_motion(params)

    col1, col2 = st.columns(2)

    with col1:
        st.header("Траектория движения")
        trajectory_chart = create_position_chart(x, y, t)
        st.altair_chart(trajectory_chart, use_container_width=True)

    with col2:
        st.header("Скорости")
        velocity_chart = create_velocity_chart(vx, vy, omega_z, t)
        st.altair_chart(velocity_chart, use_container_width=True)

    st.header("Анализ энергии")
    energy_chart = create_energy_chart(model, params, vx, vy, omega_z, t)
    st.altair_chart(energy_chart, use_container_width=True)


def display_advanced_analysis(model, params):
    x, y, vx, vy, omega_z, t = model.simulate_motion(params)

    col1, col2 = st.columns(2)

    with col1:
        st.header("Траектория с векторами скорости")
        trajectory_vectors = create_trajectory_with_vectors(x, y, vx, vy, t)
        st.altair_chart(trajectory_vectors, use_container_width=True)

    with col2:
        st.header("Фазовый портрет")
        phase_portrait = create_phase_portrait_chart(vx, omega_z, t)
        st.altair_chart(phase_portrait, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.header("Сила трения")
        st.altair_chart(
            create_friction_force_chart(model, params, vx, vy, omega_z, t),
            use_container_width=True,
        )

    with col4:
        st.header("Условие качения")
        st.altair_chart(
            create_rolling_condition_chart(vx, omega_z, params, t),
            use_container_width=True,
        )

    col5, col6 = st.columns(2)

    with col5:
        st.header("Компоненты ускорения")
        st.altair_chart(
            create_acceleration_components_chart(
                model, params, vx, vy, omega_z, t
            ),
            use_container_width=True,
        )

    with col6:
        st.header("Детальный баланс энергии")
        st.altair_chart(
            create_energy_balance_chart(
                model, params, x, y, vx, vy, omega_z, t
            ),
            use_container_width=True,
        )
