"""Конфигурация настроек для симуляции идеального газа."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationConfig:
    """Параметры симуляции идеального газа.
    
    Молекуларные параметры:
        num_particles: Количество молекул
        particle_mass: Масса одной молекулы (кг), по умолчанию Ar
        particle_diameter: Эффективный диаметр молекулы (м)
    
    Геометрия сосуда:
        container_height: Высота сосуда (м)
        container_radius: Радиус цилиндра (м)
    
    Начальные условия:
        initial_velocity: Начальная скорость молекул (м/с)
        initial_height: Начальная высота (м)
    
    Параметры времени:
        simulation_time: Тотальное время симуляции (с)
        dt: Шаг интегрирования (с)
    
    Параметры термостата:
        use_thermostat: Освание / теплоаккумуляция стенками
        target_temperature: Целевая температура (К)
    """
    
    # Молекулярные параметры
    num_particles: int
    particle_mass: float = 6.63e-26  # масса аргона (кг)
    particle_diameter: float = 3.4e-10  # эффективный диаметр (м)
    
    # Геометрия сосуда
    container_height: float
    container_radius: float
    
    # Начальные условия
    initial_velocity: float
    initial_height: float = 0.1  # Настолько высоко (м)
    
    # Параметры времени
    simulation_time: float
    dt: float
    
    # Параметры термостата
    use_thermostat: bool = False
    target_temperature: Optional[float] = None
    
    # Вычисляемые и стойкие параметры
    g: float = 9.81  # ауюстрение свободного падения (м/с^2)
    k_B: float = 1.38e-23  # постоянная Больцмана (Дж/К)
    
    def __post_init__(self):
        """Проверка много допустимых состояний."""
        if self.num_particles <= 0:
            raise ValueError("Количество молекул должно быть положительным")
        if self.dt <= 0 or self.dt > 0.01:
            raise ValueError("Шаг времени должен быть менюше 0.01")
        if self.initial_height > self.container_height:
            raise ValueError("Начальная высота не может быть больше высоты сосуда")
