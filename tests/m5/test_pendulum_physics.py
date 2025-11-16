import numpy as np
import pytest

from models.pendulum.objects import PendulumModel, PendulumParameters


@pytest.fixture
def pendulum_model():
    return PendulumModel()


@pytest.fixture
def basic_params():
    return PendulumParameters(
        length=1.0,
        mass=1.0,
        pendulum_type="uniform_rod",
        friction_coeff=0.0,
        phi0_deg=30.0,
        omega0=0.0,
        t_max=10.0,
        dt=0.01,
    )


@pytest.fixture
def small_amplitude_params():
    return PendulumParameters(
        length=1.0,
        mass=1.0,
        pendulum_type="uniform_rod",
        friction_coeff=0.0,
        phi0_deg=5.0,
        omega0=0.0,
        t_max=20.0,
        dt=0.005,
    )


def test_pendulum_model_initialization(pendulum_model):
    assert pendulum_model.g == 9.81


def test_pendulum_properties_uniform_rod(pendulum_model):
    params = PendulumParameters(
        length=1.0, mass=1.0, pendulum_type="uniform_rod"
    )

    i_o, cm_distance, omega0 = pendulum_model._get_pendulum_properties(params)

    expected_i_o = 1.0 * 1.0**2 / 3
    expected_cm_distance = 1.0 / 2
    expected_omega0 = np.sqrt(1.0 * 9.81 * expected_cm_distance / expected_i_o)

    assert pytest.approx(expected_i_o) == i_o
    assert cm_distance == pytest.approx(expected_cm_distance)
    assert omega0 == pytest.approx(expected_omega0)


def test_pendulum_properties_point_mass(pendulum_model):
    params = PendulumParameters(
        length=1.0, mass=1.0, pendulum_type="point_mass"
    )

    i_o, cm_distance, omega0 = pendulum_model._get_pendulum_properties(params)

    expected_i_o = 1.0 * 1.0**2
    expected_cm_distance = 1.0
    expected_omega0 = np.sqrt(1.0 * 9.81 * expected_cm_distance / expected_i_o)

    assert pytest.approx(expected_i_o) == i_o
    assert cm_distance == pytest.approx(expected_cm_distance)
    assert omega0 == pytest.approx(expected_omega0)


def test_small_amplitude_oscillations(pendulum_model, small_amplitude_params):
    phi, omega, t = pendulum_model.simulate_motion(small_amplitude_params)

    assert len(phi) == len(omega) == len(t)
    assert len(phi) > 0

    assert phi[0] == pytest.approx(
        np.radians(small_amplitude_params.phi0_deg), abs=1e-6
    )
    assert omega[0] == pytest.approx(small_amplitude_params.omega0, abs=1e-6)

    _i_o, _cm_distance, omega0_calc = pendulum_model._get_pendulum_properties(
        small_amplitude_params
    )
    theoretical_period = 2 * np.pi / omega0_calc

    period = pendulum_model.calculate_period(phi, t)

    assert period > 0, "Не удалось определить период колебаний"
    assert period == pytest.approx(theoretical_period, rel=0.005)


def test_energy_conservation_no_friction(pendulum_model, basic_params):
    phi, omega, _t = pendulum_model.simulate_motion(basic_params)

    _kinetic, _potential, total = pendulum_model.calculate_energy(
        phi, omega, basic_params
    )

    initial_energy = total[0]
    final_energy = total[-1]

    relative_change = abs(final_energy - initial_energy) / initial_energy

    assert relative_change < 1e-3


def test_damped_oscillations(pendulum_model):
    params = PendulumParameters(
        length=1.0,
        mass=1.0,
        pendulum_type="uniform_rod",
        friction_coeff=0.2,
        phi0_deg=30.0,
        omega0=0.0,
        t_max=20.0,
        dt=0.01,
    )

    phi, _omega, _t = pendulum_model.simulate_motion(params)

    max_amplitudes = [
        phi[i]
        for i in range(1, len(phi) - 1)
        if phi[i - 1] < phi[i] > phi[i + 1] and phi[i] > 0
    ]

    assert len(max_amplitudes) >= 3

    for i in range(1, len(max_amplitudes)):
        assert max_amplitudes[i] < max_amplitudes[i - 1]


def test_period_amplitude_dependence(pendulum_model):
    base_params = PendulumParameters(
        length=1.0,
        mass=1.0,
        pendulum_type="uniform_rod",
        friction_coeff=0.0,
        t_max=25.0,
        dt=0.005,
    )

    amplitudes = [5, 30, 60]
    periods = []

    for amp in amplitudes:
        params = PendulumParameters(**{**base_params.__dict__, "phi0_deg": amp})
        phi, _omega, t = pendulum_model.simulate_motion(params)
        period = pendulum_model.calculate_period(phi, t)

        if period > 0:
            periods.append(period)
        else:
            periods.append(0)

    valid_periods = [p for p in periods if p > 0]
    if len(valid_periods) >= 2:
        assert max(valid_periods) > min(valid_periods), (
            "Период должен зависеть от амплитуды"
        )
    else:
        pytest.skip("Не удалось определить достаточно периодов для анализа")


def test_equations_of_motion(pendulum_model):
    params = PendulumParameters()

    state_equilibrium = np.array([0.0, 0.0])
    derivatives = pendulum_model._equations_of_motion(
        0, state_equilibrium, params
    )

    assert derivatives[0] == 0.0
    assert derivatives[1] == 0.0

    state_displaced = np.array([np.radians(30), 0.0])
    derivatives = pendulum_model._equations_of_motion(
        0, state_displaced, params
    )

    assert derivatives[0] == 0.0

    assert derivatives[1] < 0


def test_energy_calculation(pendulum_model):
    params = PendulumParameters(
        length=1.0, mass=1.0, pendulum_type="uniform_rod"
    )

    phi = np.array([np.radians(30)])
    omega = np.array([0.0])

    kinetic, potential, total = pendulum_model.calculate_energy(
        phi, omega, params
    )

    assert kinetic[0] == 0.0
    assert potential[0] > 0.0
    assert total[0] == potential[0]

    phi = np.array([0.0])
    omega = np.array([2.0])

    kinetic, potential, total = pendulum_model.calculate_energy(
        phi, omega, params
    )

    assert kinetic[0] > 0.0
    assert potential[0] == 0.0
    assert total[0] == kinetic[0]
