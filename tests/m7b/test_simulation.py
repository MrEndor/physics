import numpy as np
from scipy.integrate import trapezoid

from models.ideal_gas.config import SimulationConfig
from models.ideal_gas.objects import EquilibriumAnalyzer, IdealGasSimulation
from models.ideal_gas.utils import (
    calculate_barometric_formula,
    calculate_maxwell_distribution,
    calculate_speed_from_velocity,
)


def test_simulation_output_shapes(basic_config):
    sim = IdealGasSimulation(basic_config)
    positions, velocities, times = sim.run()

    num_steps = int(basic_config.simulation_time / basic_config.dt)

    assert positions.shape == (num_steps, basic_config.num_particles, 3)
    assert velocities.shape == (num_steps, basic_config.num_particles, 3)
    assert times.shape == (num_steps,)


def test_particles_stay_in_bounds(basic_config):
    sim = IdealGasSimulation(basic_config)
    positions, _, _ = sim.run()

    final_positions = positions[-1]

    assert np.all(final_positions[:, 2] >= 0), "Some particles went below floor"
    assert np.all(final_positions[:, 2] <= basic_config.container_height), (
        "Some particles went above ceiling"
    )

    r = np.sqrt(final_positions[:, 0] ** 2 + final_positions[:, 1] ** 2)
    assert np.all(r <= basic_config.container_radius * 1.01), (
        "Some particles outside cylinder"
    )


def test_time_progression(basic_config):
    sim = IdealGasSimulation(basic_config)
    _, _, times = sim.run()

    assert np.all(np.diff(times) > 0), "Time should increase monotonically"
    assert times[0] == 0.0, "Time should start at 0"
    assert times[-1] <= basic_config.simulation_time, (
        "Final time should not exceed simulation_time"
    )


def test_total_energy_conserved(basic_config):
    sim = IdealGasSimulation(basic_config)
    positions, velocities, _ = sim.run()

    total_energies = []
    g = basic_config.g
    m = basic_config.particle_mass

    for pos_step, vel_step in zip(positions, velocities, strict=False):
        kinetic_energy = 0.5 * m * np.sum(vel_step**2)
        potential_energy = m * g * np.sum(pos_step[:, 2])
        total_energies.append(kinetic_energy + potential_energy)

    total_energies = np.array(total_energies)

    initial_energy = total_energies[0]

    rel_energy_error = np.max(np.abs(total_energies - initial_energy)) / np.abs(
        initial_energy
    )
    assert rel_energy_error < 0.05, (
        f"Energy conservation error too large: {rel_energy_error:.2%}"
    )


def test_kinetic_energy_decreases_initially(basic_config):
    sim = IdealGasSimulation(basic_config)
    _, velocities, _ = sim.run()

    m = basic_config.particle_mass
    kinetic_energies = []
    for vel_step in velocities:
        kinetic_energy = 0.5 * m * np.sum(vel_step**2)
        kinetic_energies.append(kinetic_energy)

    kinetic_energies = np.array(kinetic_energies)

    assert kinetic_energies[0] > kinetic_energies[-1], (
        "KE should decrease (particles rise against gravity)"
    )


def test_potential_energy_increases(basic_config):
    sim = IdealGasSimulation(basic_config)
    positions, _, _ = sim.run()

    g = basic_config.g
    m = basic_config.particle_mass

    potential_energies = []
    for pos_step in positions:
        potential_energy = m * g * np.sum(pos_step[:, 2])
        potential_energies.append(potential_energy)

    potential_energies = np.array(potential_energies)

    assert potential_energies[-1] > potential_energies[0], (
        "PE should increase (particles rise)"
    )


def test_mechanical_energy_conservation_no_collisions():
    config = SimulationConfig(
        num_particles=10,
        container_height=5.0,
        container_radius=2.0,
        initial_velocity=5.0,
        simulation_time=0.5,
        dt=0.01,
    )

    sim = IdealGasSimulation(config)
    positions, velocities, _ = sim.run()

    g = config.g
    m = config.particle_mass

    total_energies = []
    for pos_step, vel_step in zip(positions, velocities, strict=False):
        kinetic_energy = 0.5 * m * np.sum(vel_step**2)
        potential_energy = m * g * np.sum(pos_step[:, 2])
        total_energies.append(kinetic_energy + potential_energy)

    total_energies = np.array(total_energies)
    initial_energy = total_energies[0]

    rel_energy_error = np.max(np.abs(total_energies - initial_energy)) / np.abs(
        initial_energy
    )
    assert rel_energy_error < 0.02, (
        f"Energy drift too large: {rel_energy_error:.2%}"
    )


def test_temperature_calculation(basic_config):
    sim = IdealGasSimulation(basic_config)
    _, velocities, _ = sim.run()

    analyzer = EquilibriumAnalyzer(basic_config)
    temperature = analyzer.calculate_temperature(velocities[-1])

    assert temperature > 0, "Temperature should be positive"
    assert temperature < 10000, "Temperature seems unreasonably high"


def test_pressure_calculation(basic_config):
    sim = IdealGasSimulation(basic_config)
    positions, velocities, _ = sim.run()

    analyzer = EquilibriumAnalyzer(basic_config)
    pressure = analyzer.calculate_pressure(positions[-1], velocities[-1], 0)

    assert pressure > 0, "Pressure should be positive"


def test_maxwell_distribution_normalization():
    speeds = np.linspace(0, 1000, 1000)
    maxwell = calculate_maxwell_distribution(speeds, 300, 6.63e-26)

    integral = trapezoid(maxwell, speeds)
    assert 0.9 < integral < 1.1, (
        f"Maxwell integral should be ~1, got {integral}"
    )


def test_barometric_formula_normalization():
    heights = np.linspace(0, 5, 100)
    barometric = calculate_barometric_formula(heights, 1.0)

    integral = trapezoid(barometric, heights)
    assert 0.9 < integral < 1.1, (
        f"Barometric integral should be ~1, got {integral}"
    )


def test_speed_calculation():
    velocities = np.array([
        [3, 4, 0],  # speed = 5
        [0, 0, 5],  # speed = 5
        [1, 1, 1],  # speed = sqrt(3)
    ])

    speeds = calculate_speed_from_velocity(velocities)

    assert np.isclose(speeds[0], 5.0), "Speed calculation for [3,4,0]"
    assert np.isclose(speeds[1], 5.0), "Speed calculation for [0,0,5]"
    assert np.isclose(speeds[2], np.sqrt(3)), "Speed calculation for [1,1,1]"


def test_simulation_doesnt_diverge(basic_config):
    sim = IdealGasSimulation(basic_config)
    positions, velocities, _ = sim.run()

    assert np.all(np.isfinite(positions)), "Positions contain non-finite values"
    assert np.all(np.isfinite(velocities)), (
        "Velocities contain non-finite values"
    )

    assert np.all(np.abs(velocities) < 1000), (
        "Velocities are unreasonably large"
    )


def test_no_negative_kinetic_energy(basic_config):
    sim = IdealGasSimulation(basic_config)
    _, velocities, _ = sim.run()

    for vel_step in velocities:
        kinetic_energy = np.sum(vel_step**2)
        assert kinetic_energy >= 0, "Kinetic energy cannot be negative"


def test_gravity_pulls_downward(basic_config):
    config = SimulationConfig(
        num_particles=1,
        container_height=10.0,
        container_radius=5.0,
        initial_velocity=20.0,
        initial_height=0.0,
        simulation_time=1.0,
        dt=0.01,
    )

    sim = IdealGasSimulation(config)
    positions, velocities, _ = sim.run()

    initial_v_z = velocities[0, 0, 2]
    final_v_z = velocities[-1, 0, 2]

    if initial_v_z <= 0:
        velocity_changed = abs(final_v_z - initial_v_z) > 0.1

        assert velocity_changed, "Gravity should affect velocity"
        return

    velocity_decreased = False
    for i in range(1, min(10, len(velocities))):
        current_v_z = velocities[i, 0, 2]
        if current_v_z < initial_v_z:
            velocity_decreased = True
            break

    assert velocity_decreased, "Gravity should reduce upward velocity"

    initial_z = positions[0, 0, 2]
    max_z = np.max(positions[:, 0, 2])
    assert max_z > initial_z, (
        "Particle should rise due to initial upward velocity"
    )
