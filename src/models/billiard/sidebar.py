import math
from typing import Any

import streamlit as st

from models.billiard.config import PHYSICAL_CONSTANTS, CollisionMethod
from models.billiard.objects import BilliardParameters


def create_billiard_sidebar() -> BilliardParameters:
    st.sidebar.header("Параметры моделирования")

    st.sidebar.subheader("Метод расчета")
    method = st.sidebar.selectbox(
        "Выберите метод:",
        options=[method.value for method in CollisionMethod],
        index=0,
        help="Выбор между аналитическим и численным подходом",
    )

    st.sidebar.subheader("Биток (белый шар)")

    cue_params = _create_ball_inputs("cue", 0.5, 0.7, 3.0, 0.0)

    st.sidebar.subheader("Прицельный шар")

    target_params = _create_ball_inputs("target", 1.8, 0.7, 0.0, 0.0)

    st.sidebar.subheader("Физические параметры")

    ball_mass = st.sidebar.slider(
        "Масса шара (кг):",
        min_value=0.1,
        max_value=0.5,
        value=PHYSICAL_CONSTANTS["ball_mass"],
        step=0.01,
        help="Стандартная масса бильярдного шара ≈ 0.17 кг",
    )

    ball_radius = st.sidebar.slider(
        "Радиус шара (м):",
        min_value=0.02,
        max_value=0.05,
        value=PHYSICAL_CONSTANTS["ball_radius"],
        step=0.001,
        format="%.3f",
        help="Стандартный радиус бильярдного шара ≈ 0.0286 м",
    )

    friction_coeff = st.sidebar.slider(
        "Коэффициент трения:",
        min_value=0.0,
        max_value=0.1,
        value=PHYSICAL_CONSTANTS["friction_coeff"],
        step=0.001,
        format="%.3f",
        help="Коэффициент трения скольжения",
    )

    st.sidebar.subheader("Параметры расчета")

    t_max = st.sidebar.slider(
        "Время моделирования (с):",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
    )

    k_spring = 1e6
    if method == "Силовая модель (Гук)":
        st.sidebar.subheader("Параметры модели Гука")
        k_spring = st.sidebar.number_input(
            "Коэффициент жесткости (Н/м):",
            min_value=1e4,
            max_value=1e7,
            value=1e6,
            step=1e4,
            format="%.0e",
            help="Чем больше k, тем жестче столкновения",
        )

    dt = _calculate_optimal_timestep(
        method,
        ball_radius,
        k_spring if method == "Силовая модель (Гук)" else None,
    )

    st.sidebar.subheader("Информация о столе")
    st.sidebar.info(
        f"**Размеры стола:**\n"
        f"Длина: {PHYSICAL_CONSTANTS['table_length']:.2f} м\n"
        f"Ширина: {PHYSICAL_CONSTANTS['table_width']:.2f} м\n"
    )

    return BilliardParameters(
        cue_x=cue_params["x"],
        cue_y=cue_params["y"],
        cue_vx=cue_params["vx"],
        cue_vy=cue_params["vy"],
        target_x=target_params["x"],
        target_y=target_params["y"],
        target_vx=target_params["vx"],
        target_vy=target_params["vy"],
        t_max=t_max,
        dt=dt,
        ball_mass=ball_mass,
        ball_radius=ball_radius,
        k_spring=k_spring,
        friction_coeff=friction_coeff,
    )


def _create_ball_inputs(
    ball_type: str,
    default_x: float,
    default_y: float,
    default_vx: float,
    default_vy: float,
) -> dict[str, Any]:
    col1, col2 = st.sidebar.columns(2)

    with col1:
        x = st.number_input(
            "X позиция (м):",
            min_value=0.1,
            max_value=PHYSICAL_CONSTANTS["table_length"] - 0.1,
            value=default_x,
            step=0.01,
            key=f"{ball_type}_x",
        )
        vx = st.number_input(
            "Скорость X (м/с):",
            min_value=-10.0,
            max_value=10.0,
            value=default_vx,
            step=0.01,
            key=f"{ball_type}_vx",
        )

    with col2:
        y = st.number_input(
            "Y позиция (м):",
            min_value=0.1,
            max_value=PHYSICAL_CONSTANTS["table_width"] - 0.1,
            value=default_y,
            step=0.01,
            key=f"{ball_type}_y",
        )
        vy = st.number_input(
            "Скорость Y (м/с):",
            min_value=-10.0,
            max_value=10.0,
            value=default_vy,
            step=0.01,
            key=f"{ball_type}_vy",
        )

    return {"x": x, "y": y, "vx": vx, "vy": vy}


def _calculate_optimal_timestep(
    method_name: str, ball_radius: float, k_spring: float | None = None
) -> float:
    if method_name == "Законы сохранения":
        return 0.001

    if k_spring is not None:
        mass = PHYSICAL_CONSTANTS["ball_mass"]
        omega = (k_spring / mass) ** 0.5
        period = 2 * math.pi / omega
        optimal_dt: float = period / 100
        return max(min(optimal_dt, 0.0001), 1e-6)

    return 0.0001


def get_selected_method() -> CollisionMethod:
    method_name = None

    for key, value in st.session_state.items():
        if "selectbox" in str(key).lower() and value in [
            m.value for m in CollisionMethod
        ]:
            method_name = value
            break

    if not method_name:
        method_name = "Законы сохранения"

    for method in CollisionMethod:
        if method.value == method_name:
            return method

    return CollisionMethod.CONSERVATION_LAWS
