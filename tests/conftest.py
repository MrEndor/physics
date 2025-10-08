import pytest

from models.billiard.objects import BilliardParameters, BilliardSimulation
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


@pytest.fixture
def billiard_simulation() -> BilliardSimulation:
    return BilliardSimulation()


@pytest.fixture
def collision_params() -> BilliardParameters:
    return BilliardParameters(
        cue_x=0.5,
        cue_y=0.5,
        cue_vx=1.0,
        cue_vy=0.0,
        target_x=0.8,
        target_y=0.5,
        target_vx=0.0,
        target_vy=0.0,
        t_max=2.0,
        dt=0.001,
        friction_coeff=0.0,
    )


@pytest.fixture
def no_collision_params() -> BilliardParameters:
    return BilliardParameters(
        cue_x=0.5,
        cue_y=0.5,
        cue_vx=0.5,
        cue_vy=0.0,
        target_x=2.0,
        target_y=1.0,
        target_vx=0.0,
        target_vy=0.0,
        t_max=3.0,
        dt=0.01,
    )
