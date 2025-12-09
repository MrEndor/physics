"""Конфигурация настроек для симуляции идеального газа."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationConfig:
    """Параметры симуляции идеального газа.
    
    Молекулярные параметры:
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
        use_thermostat: Основание / теплоаккумуляция стенками
        target_temperature: Целевая температура (К)
    """
    
    # Молекулярные параметры
    num_particles: int
    # НОВО: условная масса для университетского моделя
    # Это не реальная аргона, а высокая сильная барометрическая экспонента
    particle_mass: float = 6.63e-23  # условная масса (100× тяжелее для наглядности)
    particle_diameter: float = 3.4e-10  # эффективный диаметр (м)
    
    # Геометрия сосуда
    container_height: float
    container_radius: float
    
    # Начальные условия
    initial_velocity: float
    initial_height: float = 0.1  # На этакой высоте (м)
    
    # Параметры времени
    simulation_time: float
    dt: float
    
    # Параметры термостата
    use_thermostat: bool = False
    target_temperature: Optional[float] = None
    
    # Вычисляемые и постоянные параметры
    g: float = 9.81  # ускорение свободного падения (м/с²)
    k_B: float = 1.38e-23  # постоянная Больцмана (Дж/К)
    
    def __post_init__(self):
        """Проверка находящихся допустимых состояния."""
        if self.num_particles <= 0:
            raise ValueError("Количество молекул должно быть положительным")
        if self.dt <= 0 or self.dt > 0.01:
            raise ValueError("Шаг времени должен быть <= 0.01")
        if self.initial_height > self.container_height:
            raise ValueError("Начальная высота не может быть больше высоты сосуда")
