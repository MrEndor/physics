"""Утилиты для обработки данных исмуляции."""

import numpy as np
import pandas as pd
from typing import Tuple


def calculate_maxwell_distribution(speeds: np.ndarray, temperature: float, particle_mass: float) -> np.ndarray:
    """Распределение Максвелла по модулю скорости.
    
    f(v) = 4π v^2 (m / (2π k_B T))^(3/2) exp(-m v^2 / (2 k_B T))
    
    Args:
        speeds: Проенство скоростей
        temperature: Температура (K)
        particle_mass: Масса партикулы (kg)
    
    Returns:
        Нормализованное распределение
    """
    k_B = 1.38e-23
    
    # Исключительные случаи
    if temperature <= 0 or particle_mass <= 0:
        return np.zeros_like(speeds)
    
    # Префакторы
    sqrt_term = np.sqrt(particle_mass / (2 * np.pi * k_B * temperature))
    norm_factor = 4 * np.pi * (sqrt_term ** 3)
    exp_arg = -particle_mass * speeds**2 / (2 * k_B * temperature)
    
    distribution = norm_factor * speeds**2 * np.exp(exp_arg)
    
    return distribution


def calculate_barometric_formula(heights: np.ndarray, scale_height: float) -> np.ndarray:
    """Барометрическая формула для распределения но высоте.
    
    n(z) = n(0) exp(-z / H0)
    где H0 = k_B T / (m g)
    
    Args:
        heights: Простланство высот (m)
        scale_height: Масштаб высоты H0 (m)
    
    Returns:
        Нормализованное распределение
    """
    if scale_height <= 0:
        return np.zeros_like(heights)
    
    # Нормализованное распределение
    distribution = (1.0 / scale_height) * np.exp(-heights / scale_height)
    
    return distribution


def calculate_speed_from_velocity(velocities: np.ndarray) -> np.ndarray:
    """Перечислить ни векторы в них модули."""
    return np.linalg.norm(velocities, axis=1)


def create_statistics_dataframe(positions: np.ndarray, velocities: np.ndarray,
                               particle_mass: float, times: np.ndarray) -> pd.DataFrame:
    """Особые данные ис жистории симуляции."""
    num_steps = len(times)
    
    data = {
        'time': [],
        'kinetic_energy': [],
        'potential_energy': [],
        'total_energy': [],
        'temperature': [],
        'mean_height': [],
    }
    
    g = 9.81
    k_B = 1.38e-23
    
    for step in range(num_steps):
        pos = positions[step]
        vel = velocities[step]
        
        # Кинетическая энергия
        KE = 0.5 * particle_mass * np.sum(vel**2)
        
        # Потенциальная энергия
        PE = particle_mass * g * np.sum(pos[:, 2])
        
        # Температура
        kinetic_per_particle = KE / len(pos)
        T = 2.0 * kinetic_per_particle / (3.0 * k_B)
        
        data['time'].append(times[step])
        data['kinetic_energy'].append(KE)
        data['potential_energy'].append(PE)
        data['total_energy'].append(KE + PE)
        data['temperature'].append(T)
        data['mean_height'].append(np.mean(pos[:, 2]))
    
    return pd.DataFrame(data)
