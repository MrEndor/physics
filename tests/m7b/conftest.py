"""Приспособления конфигурации для тестов M7B."""

import pytest
from models.ideal_gas.config import SimulationConfig
from models.ideal_gas.objects import IdealGasSimulation


@pytest.fixture
def basic_config():
    """Базовая конфигурация для тестов."""
    return SimulationConfig(
        num_particles=100,
        container_height=2.0,
        container_radius=1.0,
        initial_velocity=10.0,
        simulation_time=1.0,
        dt=0.01,
    )


@pytest.fixture
def simulation(basic_config):
    """Краткая симуляция для тестов."""
    return IdealGasSimulation(basic_config)


@pytest.fixture
def simulation_results(simulation):
    """Проведи симуляцию и верни результаты."""
    return simulation.run()
