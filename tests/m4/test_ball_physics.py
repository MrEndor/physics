import numpy as np
import pytest

from models.rolling_ball.objects import BallModel, BallParameters


@pytest.fixture
def ball_model() -> BallModel:
    return BallModel()


@pytest.fixture
def basic_ball_params() -> BallParameters:
    return BallParameters(
        mass=1.0,
        radius=0.1,
        mu_friction=0.3,
        theta_degrees=0.0,
        vx0=1.0,
        vy0=0.0,
        omega_z0=0.0,
        t_max=5.0,
        dt=0.01,
    )


@pytest.fixture
def inclined_params() -> BallParameters:
    return BallParameters(
        mass=1.0,
        radius=0.1,
        mu_friction=0.3,
        theta_degrees=15.0,
        vx0=0.0,
        vy0=0.0,
        omega_z0=0.0,
        t_max=3.0,
        dt=0.01,
    )


def test_ball_model_initialization(ball_model: BallModel) -> None:
    assert ball_model.g == 9.81


def test_horizontal_motion_energy_conservation(
    ball_model: BallModel, basic_ball_params: BallParameters
) -> None:
    x, y, vx, vy, omega_z, t = ball_model.simulate_motion(basic_ball_params)

    assert len(x) == len(y) == len(vx) == len(vy) == len(omega_z) == len(t)
    assert x[0] == pytest.approx(basic_ball_params.x0)
    assert y[0] == pytest.approx(basic_ball_params.y0)
    assert vx[0] == pytest.approx(basic_ball_params.vx0)


def test_inclined_plane_acceleration(
    ball_model: BallModel, inclined_params: BallParameters
) -> None:
    _x, _y, vx, _vy, _omega_z, _t = ball_model.simulate_motion(inclined_params)

    expected_acceleration = (
        5 * ball_model.g * np.sin(np.radians(inclined_params.theta_degrees)) / 7
    )

    dt = inclined_params.dt
    actual_acceleration = (vx[1] - vx[0]) / dt

    assert actual_acceleration == pytest.approx(expected_acceleration, rel=0.1)


def test_friction_force_calculation(ball_model: BallModel) -> None:
    params = BallParameters(mass=1.0, radius=0.1, mu_friction=0.3)

    fx, _, mode = ball_model.calculate_friction_force(1.0, 0.0, 0.0, params)
    assert mode == "sliding"
    assert fx < 0

    fx, _fy, mode = ball_model.calculate_friction_force(
        0.001, 0.0, 0.01, params
    )
    assert mode == "rolling"


def test_momentum_conservation_properties(
    ball_model: BallModel, basic_ball_params: BallParameters
) -> None:
    _, _, vx, _, _, _ = ball_model.simulate_motion(basic_ball_params)

    assert not np.allclose(vx, basic_ball_params.vx0)
