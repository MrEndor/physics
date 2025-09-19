import streamlit as st

from models.stone_flight.objects import FlightParameters


def create_stone_flight_sidebar() -> FlightParameters:
    with st.sidebar:
        st.header("Параметры броска")

        v0 = st.slider(
            "Начальная скорость (м/с)",
            min_value=5.0,
            max_value=100.0,
            value=30.0,
            step=1.0,
            help="Скорость броска камня",
            key="slider_v0",
        )

        angle_deg = st.slider(
            "Угол броска (°)",
            min_value=0.0,
            max_value=90.0,
            value=45.0,
            step=1.0,
            help="Угол к горизонту",
            key="slider_angle_deg",
        )

        mass = st.slider(
            "Масса камня (кг)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Масса брошенного камня",
            key="slider_mass",
        )

        radius = st.slider(
            "Радиус камня (м)",
            min_value=0.01,
            max_value=1.0,
            value=0.05,
            step=0.01,
            help="Радиус сферического камня",
            key="slider_radius",
        )

        cd_linear = st.slider(
            "Коэффициент линейного сопротивления",
            min_value=0.01,
            max_value=1.0,
            value=0.05,
            step=0.01,
            key="slider_cd_linear",
        )

        cd_quadratic = st.slider(
            "Коэффициент квадратичного сопротивления",
            min_value=0.01,
            max_value=1.0,
            value=0.05,
            step=0.01,
            key="slider_cd_quadratic",
        )

    return FlightParameters(
        v0=v0,
        angle_deg=angle_deg,
        mass=mass,
        radius=radius,
        cd_linear=cd_linear,
        cd_quadratic=cd_quadratic,
    )
