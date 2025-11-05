import streamlit as st

from models.pendulum.objects import PendulumParameters


def create_pendulum_sidebar() -> PendulumParameters:
    st.sidebar.header("Параметры маятника")

    length = st.sidebar.number_input(
        "Длина L (м)", min_value=0.1, max_value=10.0, value=1.0, step=0.1
    )

    mass = st.sidebar.number_input(
        "Масса m (кг)", min_value=0.1, max_value=10.0, value=1.0, step=0.1
    )

    pendulum_type = st.sidebar.selectbox(
        "Тип маятника",
        options=["uniform_rod", "point_mass"],
        format_func=lambda x: {
            "uniform_rod": "Однородный стержень",
            "point_mass": "Точечная масса",
        }[x],
        index=0,
    )

    st.sidebar.header("Физические параметры")

    friction_coeff = st.sidebar.number_input(
        "Коэффициент трения γ (с⁻¹)",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
    )

    st.sidebar.header("Начальные условия")

    phi0_deg = st.sidebar.number_input(
        "Начальный угол φ₀ (градусы)",
        min_value=-180.0,
        max_value=180.0,
        value=30.0,
        step=5.0,
    )

    omega0 = st.sidebar.number_input(
        "Начальная угловая скорость ω₀ (рад/с)",
        min_value=-10.0,
        max_value=10.0,
        value=0.0,
        step=0.5,
    )

    st.sidebar.header("Параметры симуляции")

    t_max = st.sidebar.number_input(
        "Время симуляции (с)",
        min_value=5.0,
        max_value=100.0,
        value=20.0,
        step=5.0,
    )

    return PendulumParameters(
        length=length,
        mass=mass,
        pendulum_type=pendulum_type,
        friction_coeff=friction_coeff,
        phi0_deg=phi0_deg,
        omega0=omega0,
        t_max=t_max,
        dt=0.01,
    )
