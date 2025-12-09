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


class TestEnergyConservation:
    """Tests for energy conservation laws."""
    
    def test_total_energy_conserved(self, basic_config):
        """Проверить основной закон сохранения энергии для изолированной системы."""
        sim = IdealGasSimulation(basic_config)
        positions, velocities, _ = sim.run()
        
        # Полная энергия всегда
        total_energies = []
        g = basic_config.g
        m = basic_config.particle_mass
        
        for pos_step, vel_step in zip(positions, velocities):
            # Кинетическая энергия
            KE = 0.5 * m * np.sum(vel_step**2)
            # Потенциальная энергия
            PE = m * g * np.sum(pos_step[:, 2])
            # Полная энергия
            total_energies.append(KE + PE)
        
        total_energies = np.array(total_energies)
        
        # Инициальная энергия
        initial_energy = total_energies[0]
        
        # Проверить, что энергия почти константна
        # Рассеия до ±5% допустима (численные ошибки)
        rel_energy_error = np.max(np.abs(total_energies - initial_energy)) / np.abs(initial_energy)
        assert rel_energy_error < 0.05, f"Energy conservation error too large: {rel_energy_error:.2%}"
    
    def test_kinetic_energy_decreases_initially(self, basic_config):
        """Проверить, что кинетическая энергия снижается до явно
        (молекулы поднимаются и теряют скорость)."""
        sim = IdealGasSimulation(basic_config)
        _, velocities, _ = sim.run()
        
        # Кинетическая энергия на каждом шаге
        m = basic_config.particle_mass
        kinetic_energies = []
        for vel_step in velocities:
            KE = 0.5 * m * np.sum(vel_step**2)
            kinetic_energies.append(KE)
        
        kinetic_energies = np.array(kinetic_energies)
        
        # У частицы есть инициальная скорость, поэтому кинетическая энергия должна упасть
        assert kinetic_energies[0] > kinetic_energies[-1], "KE should decrease (particles rise against gravity)"
    
    def test_potential_energy_increases(self, basic_config):
        """Проверить, что потенциальная энергия растёт."""
        sim = IdealGasSimulation(basic_config)
        positions, _, _ = sim.run()
        
        g = basic_config.g
        m = basic_config.particle_mass
        
        potential_energies = []
        for pos_step in positions:
            PE = m * g * np.sum(pos_step[:, 2])
            potential_energies.append(PE)
        
        potential_energies = np.array(potential_energies)
        
        # Потенциальная энергия должна расти (молекулы поднимаются)
        assert potential_energies[-1] > potential_energies[0], "PE should increase (particles rise)"
    
    def test_mechanical_energy_conservation_no_collisions(self):
        """Проверить сохранение механической энергии без столкновений.
        
        Ниские отправергнутые партикулы должны дохранять энергию без высоких численных ошибок.
        """
        config = SimulationConfig(
            num_particles=10,  # Мало молекул - минимизируем столкновения
            container_height=5.0,  # большой контейнер - меньше коллизий
            container_radius=2.0,
            initial_velocity=5.0,  # нижняя скорость
            simulation_time=0.5,
            dt=0.01,
        )
        
        sim = IdealGasSimulation(config)
        positions, velocities, _ = sim.run()
        
        g = config.g
        m = config.particle_mass
        
        total_energies = []
        for pos_step, vel_step in zip(positions, velocities):
            KE = 0.5 * m * np.sum(vel_step**2)
            PE = m * g * np.sum(pos_step[:, 2])
            total_energies.append(KE + PE)
        
        total_energies = np.array(total_energies)
        initial_energy = total_energies[0]
        
        # До ±2% (очень точная неоновмыеся система)
        rel_energy_error = np.max(np.abs(total_energies - initial_energy)) / np.abs(initial_energy)
        assert rel_energy_error < 0.02, f"Energy drift too large: {rel_energy_error:.2%}"


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
    
    def test_gravity_pulls_downward(self, basic_config):
        """Проверить, что гравитация действует вниз (знак "-" в a_z = -g).
        
        Это основная проверка: если дать частице вертикальную скорость,
        гравитация должна её уменьшать.
        """
        config = SimulationConfig(
            num_particles=1,  # Одна частица
            container_height=10.0,
            container_radius=5.0,
            initial_velocity=20.0,  # Только вверх
            initial_height=0.0,  # На дне
            simulation_time=1.0,
            dt=0.01,
        )
        
        sim = IdealGasSimulation(config)
        positions, velocities, _ = sim.run()
        
        # Частица стартует с v_z > 0 (верх)
        initial_v_z = velocities[0, 0, 2]
        assert initial_v_z > 0, "Particle should start with upward velocity"
        
        # После симуляции v_z должен быть меньше (гравитация таможит)
        final_v_z = velocities[-1, 0, 2]
        assert final_v_z < initial_v_z, "Gravity should reduce upward velocity"
        
        # Высота вначале арыов (дно), а вконце выше (частица поднялась)
        initial_z = positions[0, 0, 2]
        max_z = np.max(positions[:, 0, 2])
        assert max_z > initial_z, "Particle should rise due to initial upward velocity"
