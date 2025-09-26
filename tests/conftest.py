import pytest

from models.stone_flight.objects import FlightParameters, StoneFlightModel


@pytest.fixture
def flight_model() -> StoneFlightModel:
    return StoneFlightModel()


@pytest.fixture
def basic_params() -> FlightParameters:
    return FlightParameters(
        v0=30.0,
        angle_deg=45.0,
        mass=1.0,
        radius=0.05,
        t_max=20.0,
        dt=0.01,
        cd_linear=0.1,
        cd_quadratic=0.1,
    )


@pytest.fixture
def no_drag_params() -> FlightParameters:
    return FlightParameters(
        v0=30.0,
        angle_deg=45.0,
        mass=1.0,
        radius=0.05,
        t_max=20.0,
        dt=0.01,
        cd_linear=0.0,
        cd_quadratic=0.0,
    )
