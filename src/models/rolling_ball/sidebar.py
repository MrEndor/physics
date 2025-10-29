import streamlit as st

from models.rolling_ball.objects import BallParameters


def create_ball_sidebar() -> BallParameters:
    st.sidebar.header("Параметры шара")

    mass = st.sidebar.number_input(
        "Масса шара (кг)", min_value=0.1, max_value=10.0, value=1.0, step=0.1
    )

    radius = st.sidebar.number_input(
        "Радиус шара (м)", min_value=0.01, max_value=1.0, value=0.1, step=0.01
    )

    mu_friction = st.sidebar.number_input(
        "Коэффициент трения", min_value=0.0, max_value=1.0, value=0.3, step=0.05
    )

    st.sidebar.header("Поверхность")

    theta_degrees = st.sidebar.number_input(
        "Угол наклона (градусы)",
        min_value=0.0,
        max_value=45.0,
        value=0.0,
        step=1.0,
    )

    st.sidebar.header("Начальные условия")

    x0 = st.sidebar.number_input("Начальная позиция X (м)", value=0.0, step=0.1)
    y0 = st.sidebar.number_input("Начальная позиция Y (м)", value=0.0, step=0.1)

    vx0 = st.sidebar.number_input(
        "Начальная скорость X (м/с)",
        min_value=-10.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
    )

    vy0 = st.sidebar.number_input(
        "Начальная скорость Y (м/с)",
        min_value=-10.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
    )

    omega_z0 = st.sidebar.number_input(
        "Начальная угловая скорость (рад/с)",
        min_value=-20.0,
        max_value=20.0,
        value=0.0,
        step=0.5,
    )

    st.sidebar.header("Параметры симуляции")

    t_max = st.sidebar.number_input(
        "Время симуляции (с)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
    )

    return BallParameters(
        mass=mass,
        radius=radius,
        mu_friction=mu_friction,
        theta_degrees=theta_degrees,
        x0=x0,
        y0=y0,
        vx0=vx0,
        vy0=vy0,
        omega_z0=omega_z0,
        t_max=t_max,
        dt=0.01,
    )
