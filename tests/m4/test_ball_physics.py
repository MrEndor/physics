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


@pytest.fixture
def horizontal_params_case_a() -> BallParameters:
    return BallParameters(
        mass=0.5,
        radius=0.1,
        mu_friction=0.3,
        theta_degrees=0.0,
        x0=0.0,
        y0=0.0,
        vx0=5.0,
        vy0=0.0,
        omega_z0=30.0,
        t_max=1.0,
        dt=0.001,
        epsilon=1e-4,
    )


@pytest.fixture
def horizontal_params_case_b() -> BallParameters:
    return BallParameters(
        mass=0.5,
        radius=0.1,
        mu_friction=0.3,
        theta_degrees=0.0,
        x0=0.0,
        y0=0.0,
        vx0=2.0,
        vy0=0.0,
        omega_z0=50.0,
        t_max=1.0,
        dt=0.001,
        epsilon=1e-4,
    )


@pytest.fixture
def horizontal_params_pure_rolling() -> BallParameters:
    v0 = 3.0
    r = 0.1
    omega0 = v0 / r

    return BallParameters(
        mass=0.5,
        radius=r,
        mu_friction=0.3,
        theta_degrees=0.0,
        x0=0.0,
        y0=0.0,
        vx0=v0,
        vy0=0.0,
        omega_z0=omega0,
        t_max=1.0,
        dt=0.001,
        epsilon=1e-4,
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


def test_energy_conservation_case_a(
    ball_model: BallModel, horizontal_params_case_a: BallParameters
) -> None:
    params = horizontal_params_case_a
    _, _, vx, vy, omega_z, _ = ball_model.simulate_motion(params)

    i = (2 / 5) * params.mass * params.radius**2
    energy_trans = 0.5 * params.mass * (vx**2 + vy**2)
    energy_rot = 0.5 * i * omega_z**2
    energy_total = energy_trans + energy_rot

    e0 = energy_total[0]
    ef = energy_total[-1]

    assert ef < e0

    energy_diffs = np.diff(energy_total)
    assert np.all(energy_diffs <= 1e-6)

    energy_loss = e0 - ef
    assert 0 < energy_loss < e0


def test_rolling_condition_convergence_case_a(
    ball_model: BallModel, horizontal_params_case_a: BallParameters
) -> None:
    params = horizontal_params_case_a
    _, _, vx, _, omega_z, _ = ball_model.simulate_motion(params)

    v_rel = vx - omega_z * params.radius

    assert v_rel[0] > 0

    assert v_rel[-1] < v_rel[0]

    assert abs(v_rel[-1]) < 0.01


def test_velocity_evolution_case_a(
    ball_model: BallModel, horizontal_params_case_a: BallParameters
) -> None:
    params = horizontal_params_case_a
    _, _, vx, _, omega_z, _ = ball_model.simulate_motion(params)

    assert vx[-1] < vx[0]
    assert omega_z[-1] > omega_z[0]

    dt = params.dt
    dvx_dt_initial = (vx[1] - vx[0]) / dt
    domega_dt_initial = (omega_z[1] - omega_z[0]) / dt

    expected_dvx_dt = -params.mu_friction * ball_model.g
    expected_domega_dt = (
        5 * params.mu_friction * ball_model.g / (2 * params.radius)
    )

    assert abs(dvx_dt_initial - expected_dvx_dt) / abs(expected_dvx_dt) < 0.1
    assert (
        abs(domega_dt_initial - expected_domega_dt) / abs(expected_domega_dt)
        < 0.1
    )


def test_stable_velocity_formula_case_a(
    ball_model: BallModel, horizontal_params_case_a: BallParameters
) -> None:
    params = horizontal_params_case_a
    _, _, vx, _, omega_z, _ = ball_model.simulate_motion(params)

    v_stable_theory = (5 * params.vx0 + 2 * params.omega_z0 * params.radius) / 7
    omega_stable_theory = v_stable_theory / params.radius

    v_stable_actual = vx[-1]
    omega_stable_actual = omega_z[-1]

    assert abs(v_stable_actual - v_stable_theory) / v_stable_theory < 0.05
    assert (
        abs(omega_stable_actual - omega_stable_theory) / omega_stable_theory
        < 0.05
    )


def test_energy_redistribution_case_a(
    ball_model: BallModel, horizontal_params_case_a: BallParameters
) -> None:
    params = horizontal_params_case_a
    _, _, vx, _, omega_z, _ = ball_model.simulate_motion(params)

    i = (2 / 5) * params.mass * params.radius**2

    e_trans_0 = 0.5 * params.mass * params.vx0**2
    e_rot_0 = 0.5 * i * params.omega_z0**2

    e_trans_f = 0.5 * params.mass * vx[-1] ** 2
    e_rot_f = 0.5 * i * omega_z[-1] ** 2

    delta_e_trans = e_trans_f - e_trans_0
    delta_e_rot = e_rot_f - e_rot_0
    delta_e_total = delta_e_trans + delta_e_rot

    assert delta_e_trans < 0
    assert delta_e_rot > 0
    assert delta_e_total < 0
    assert abs(delta_e_trans) > abs(delta_e_rot)


def test_velocity_evolution_case_b(
    ball_model: BallModel, horizontal_params_case_b: BallParameters
) -> None:
    params = horizontal_params_case_b
    _, _, vx, _, omega_z, _ = ball_model.simulate_motion(params)

    assert vx[-1] > vx[0]

    assert omega_z[-1] < omega_z[0]

    v_rel_0 = vx[0] - omega_z[0] * params.radius
    assert v_rel_0 < 0


def test_pure_rolling_case_c(
    ball_model: BallModel, horizontal_params_pure_rolling: BallParameters
) -> None:
    params = horizontal_params_pure_rolling
    _, _, vx, _, omega_z, _ = ball_model.simulate_motion(params)

    v_rel = vx - omega_z * params.radius
    assert np.all(np.abs(v_rel) < 0.01)

    vx_std = np.std(vx)
    omega_std = np.std(omega_z)
    assert vx_std < 0.01
    assert omega_std < 0.1

    i = (2 / 5) * params.mass * params.radius**2
    e_total = 0.5 * params.mass * vx**2 + 0.5 * i * omega_z**2
    energy_variation = (e_total[-1] - e_total[0]) / e_total[0]
    assert abs(energy_variation) < 0.01


def test_time_to_rolling(
    ball_model: BallModel, horizontal_params_case_a: BallParameters
) -> None:
    params = horizontal_params_case_a
    _, _, vx, _, omega_z, t = ball_model.simulate_motion(params)

    v_rel_0 = params.vx0 - params.omega_z0 * params.radius
    t_star_theory = 2 * v_rel_0 / (7 * params.mu_friction * ball_model.g)

    v_rel = np.abs(vx - omega_z * params.radius)
    indices_rolling = np.where(v_rel < params.epsilon)[0]

    if len(indices_rolling) > 0:
        t_star_actual = t[indices_rolling[0]]

        assert abs(t_star_actual - t_star_theory) / t_star_theory < 0.1


def test_friction_sign_case_a(
    ball_model: BallModel, horizontal_params_case_b: BallParameters
) -> None:
    vx = 5.0
    vy = 0.0
    omega_z = 0.0

    friction_x, friction_y, mode = ball_model.calculate_friction_force(
        vx, vy, omega_z, horizontal_params_case_b
    )

    assert mode == "sliding"
    assert friction_x < 0

    expected_magnitude = (
        horizontal_params_case_b.mu_friction
        * horizontal_params_case_b.mass
        * ball_model.g
    )
    actual_magnitude = np.sqrt(friction_x**2 + friction_y**2)
    assert abs(actual_magnitude - expected_magnitude) < 0.01


def test_friction_zero_for_rolling(
    ball_model: BallModel, basic_ball_params: BallParameters
) -> None:
    vx = 1.0
    omega_z = vx / basic_ball_params.radius
    vy = 0.0

    friction_x, friction_y, mode = ball_model.calculate_friction_force(
        vx,
        vy,
        omega_z,
        basic_ball_params,
    )

    assert mode == "rolling"
    assert friction_x == 0.0
    assert friction_y == 0.0
