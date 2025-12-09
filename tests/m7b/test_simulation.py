"""Проверки для моделирования идеального газа."""

import numpy as np
import pytest
from scipy.integrate import trapezoid
from models.ideal_gas.config import SimulationConfig
from models.ideal_gas.objects import IdealGasSimulation, EquilibriumAnalyzer
from models.ideal_gas.utils import (
    calculate_maxwell_distribution,
    calculate_barometric_formula,
    calculate_speed_from_velocity,
)


class TestSimulationInitialization:
    """Tests for simulation initialization."""
    
    def test_config_validation_negative_particles(self):
        """Test that negative particle count raises error."""
        with pytest.raises(ValueError):
            SimulationConfig(
                num_particles=-10,
                container_height=2.0,
                container_radius=1.0,
                initial_velocity=10.0,
                simulation_time=1.0,
                dt=0.01,
            )
    
    def test_config_validation_invalid_dt(self):
        """Test that invalid dt raises error."""
        with pytest.raises(ValueError):
            SimulationConfig(
                num_particles=100,
                container_height=2.0,
                container_radius=1.0,
                initial_velocity=10.0,
                simulation_time=1.0,
                dt=0.05,  # Требуется <= 0.01
            )
    
    def test_config_validation_height_bounds(self):
        """Test that initial height cannot exceed container height."""
        with pytest.raises(ValueError):
            SimulationConfig(
                num_particles=100,
                container_height=2.0,
                container_radius=1.0,
                initial_velocity=10.0,
                simulation_time=1.0,
                dt=0.01,
                initial_height=3.0,  # > container_height
            )


class TestSimulationRun:
    """Tests for simulation execution."""
    
    def test_simulation_output_shapes(self, basic_config):
        """Проверить, что выводы имеют правильные формы."""
        sim = IdealGasSimulation(basic_config)
        positions, velocities, times = sim.run()
        
        num_steps = int(basic_config.simulation_time / basic_config.dt)
        
        assert positions.shape == (num_steps, basic_config.num_particles, 3)
        assert velocities.shape == (num_steps, basic_config.num_particles, 3)
        assert times.shape == (num_steps,)
    
    def test_particles_stay_in_bounds(self, basic_config):
        """Test that particles stay within container bounds."""
        sim = IdealGasSimulation(basic_config)
        positions, _, _ = sim.run()
        
        final_positions = positions[-1]
        
        # Проверить высоту
        assert np.all(final_positions[:, 2] >= 0), "Some particles went below floor"
        assert np.all(final_positions[:, 2] <= basic_config.container_height), "Some particles went above ceiling"
        
        # Проверить латеральные между цилиндрами
        r = np.sqrt(final_positions[:, 0]**2 + final_positions[:, 1]**2)
        assert np.all(r <= basic_config.container_radius * 1.01), "Some particles outside cylinder"
    
    def test_time_progression(self, basic_config):
        """Test that time increases monotonically."""
        sim = IdealGasSimulation(basic_config)
        _, _, times = sim.run()
        
        assert np.all(np.diff(times) > 0), "Time should increase monotonically"
        assert times[0] == 0.0, "Time should start at 0"
        assert times[-1] <= basic_config.simulation_time, "Final time should not exceed simulation_time"


class TestPhysicsConservation:
    """Tests for physical conservation laws."""
    
    def test_total_energy_is_roughly_conserved(self, basic_config):
        """Test that total energy is roughly conserved (isolated system)."""
        basic_config.simulation_time = 1.0
        basic_config.dt = 0.01
        
        sim = IdealGasSimulation(basic_config)
        positions, velocities, _ = sim.run()
        
        # Полная энергия
        total_energies = []
        g = basic_config.g
        m = basic_config.particle_mass
        
        for pos_step, vel_step in zip(positions, velocities):
            KE = 0.5 * m * np.sum(vel_step**2)
            PE = m * g * np.sum(pos_step[:, 2])
            total_energies.append(KE + PE)
        
        total_energies = np.array(total_energies)
        
        # Проверить, что энергия не растёт сигнификантно
        energy_variation = (np.max(total_energies) - np.min(total_energies)) / np.mean(np.abs(total_energies))
        assert energy_variation < 0.5, f"Energy variation too large: {energy_variation:.2%}"
    
    def test_particles_eventually_distribute(self, basic_config):
        """Test that particles distribute in the container over time."""
        basic_config.simulation_time = 2.0
        basic_config.dt = 0.02
        
        sim = IdealGasSimulation(basic_config)
        positions, _, _ = sim.run()
        
        # Получить число молекул в нижней галявине
        initial_lower_half = np.sum(positions[0, :, 2] < basic_config.container_height / 2)
        final_lower_half = np.sum(positions[-1, :, 2] < basic_config.container_height / 2)
        
        # Ожидаем, что молекулы рассеялись вверх
        assert final_lower_half < initial_lower_half * 0.9, "Particles should distribute upward over time"


class TestEquilibriumAnalyzer:
    """Tests for equilibrium analysis."""
    
    def test_temperature_calculation(self, basic_config):
        """Test temperature calculation from velocities."""
        sim = IdealGasSimulation(basic_config)
        _, velocities, _ = sim.run()
        
        analyzer = EquilibriumAnalyzer(basic_config)
        T = analyzer.calculate_temperature(velocities[-1])
        
        # Температура должна быть положительной
        assert T > 0, "Temperature should be positive"
        # Ордер величины: несколько сот К
        assert T < 10000, "Temperature seems unreasonably high"
    
    def test_pressure_calculation(self, basic_config):
        """Test pressure calculation."""
        sim = IdealGasSimulation(basic_config)
        positions, velocities, _ = sim.run()
        
        analyzer = EquilibriumAnalyzer(basic_config)
        P = analyzer.calculate_pressure(positions[-1], velocities[-1], 0)
        
        # Давление должно быть положительным
        assert P > 0, "Pressure should be positive"


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_maxwell_distribution_normalization(self):
        """Проверить нормализацию Максвелла."""
        speeds = np.linspace(0, 1000, 1000)
        maxwell = calculate_maxwell_distribution(speeds, 300, 6.63e-26)
        
        # Приближенная проверка интеграла с scipy.integrate.trapezoid
        integral = trapezoid(maxwell, speeds)
        assert 0.9 < integral < 1.1, f"Maxwell integral should be ~1, got {integral}"
    
    def test_barometric_formula_normalization(self):
        """Проверить нормализацию барометрической формулы."""
        heights = np.linspace(0, 5, 100)
        barometric = calculate_barometric_formula(heights, 1.0)
        
        # Приближенная проверка интеграла с scipy.integrate.trapezoid
        integral = trapezoid(barometric, heights)
        assert 0.9 < integral < 1.1, f"Barometric integral should be ~1, got {integral}"
    
    def test_speed_calculation(self):
        """Проверить начисление скоростей."""
        velocities = np.array([
            [3, 4, 0],  # speed = 5
            [0, 0, 5],  # speed = 5
            [1, 1, 1],  # speed = sqrt(3)
        ])
        
        speeds = calculate_speed_from_velocity(velocities)
        
        assert np.isclose(speeds[0], 5.0), "Speed calculation for [3,4,0]"
        assert np.isclose(speeds[1], 5.0), "Speed calculation for [0,0,5]"
        assert np.isclose(speeds[2], np.sqrt(3)), "Speed calculation for [1,1,1]"


class TestNumericalStability:
    """Tests for numerical stability."""
    
    def test_simulation_doesnt_diverge(self, basic_config):
        """Test that simulation results don't diverge to infinity."""
        sim = IdealGasSimulation(basic_config)
        positions, velocities, _ = sim.run()
        
        assert np.all(np.isfinite(positions)), "Positions contain non-finite values"
        assert np.all(np.isfinite(velocities)), "Velocities contain non-finite values"
        
        # Проверить, что скорости не становятся сигнификантно большими
        assert np.all(np.abs(velocities) < 1000), "Velocities are unreasonably large"
    
    def test_no_negative_kinetic_energy(self, basic_config):
        """Test that kinetic energy never becomes negative."""
        sim = IdealGasSimulation(basic_config)
        _, velocities, _ = sim.run()
        
        for vel_step in velocities:
            KE = np.sum(vel_step**2)
            assert KE >= 0, "Kinetic energy cannot be negative"
