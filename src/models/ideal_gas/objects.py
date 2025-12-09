"""Основные классы для моделирования идеального газа."""

import numpy as np
from typing import Tuple
from models.ideal_gas.config import SimulationConfig


class IdealGasSimulation:
    """Моделирование идеального газа методом молекулярной динамики.
    
    Основные методы:
        - run(): запуск симуляции
        - _initialize_particles(): инициализация молекул
        - _velocity_verlet_step(): шаг интегрирования
        - _handle_collisions(): обработка столкновений
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.num_steps = int(config.simulation_time / config.dt)
        
        # Объекты для столкновений
        self.collision_pairs_checked = 0
        self.collisions_detected = 0
        
    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Запустить симуляцию и вернуть историю позиций и скоростей.
        
        Returns:
            positions_history: История позиций (num_steps, num_particles, 3)
            velocities_history: История скоростей (num_steps, num_particles, 3)
            times: Вектор времен
        """
        # Инициализация
        positions = np.zeros((self.config.num_particles, 3))
        velocities = np.zeros((self.config.num_particles, 3))
        self._initialize_particles(positions, velocities)
        
        # Оси для сохранения истории
        positions_history = []
        velocities_history = []
        times = []
        
        # Основной цикл
        for step in range(self.num_steps):
            t = step * self.config.dt
            times.append(t)
            
            positions_history.append(positions.copy())
            velocities_history.append(velocities.copy())
            
            # Интегрирование уравнений движения
            self._velocity_verlet_step(positions, velocities)
            
            # Обработка столкновений
            self._handle_collisions(positions, velocities)
            
            # Управление термостатом (по необходимости)
            if self.config.use_thermostat and self.config.target_temperature is not None:
                self._apply_thermostat(velocities)
        
        return np.array(positions_history), np.array(velocities_history), np.array(times)
    
    def _initialize_particles(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """Начинальные позиции и скорости.
        
        Все молекулы располагаются вблизи дна с рандомными скоростями.
        """
        # Позиции: все вблизи дна в тонком слое
        for i in range(self.config.num_particles):
            x = np.random.uniform(-self.config.container_radius * 0.9, 
                                 self.config.container_radius * 0.9)
            y = np.random.uniform(-self.config.container_radius * 0.9, 
                                 self.config.container_radius * 0.9)
            
            # Проверка: внутри цилиндра
            while np.sqrt(x**2 + y**2) > self.config.container_radius * 0.9:
                x = np.random.uniform(-self.config.container_radius * 0.9, 
                                     self.config.container_radius * 0.9)
                y = np.random.uniform(-self.config.container_radius * 0.9, 
                                     self.config.container_radius * 0.9)
            
            z = np.random.uniform(0, self.config.initial_height)
            positions[i] = [x, y, z]
        
        # Скорости: одинаковые по модулю, случайные направления
        for i in range(self.config.num_particles):
            # Случайное направление
            phi = np.random.uniform(0, 2 * np.pi)
            theta = np.arccos(np.random.uniform(-1, 1))
            
            vx = self.config.initial_velocity * np.sin(theta) * np.cos(phi)
            vy = self.config.initial_velocity * np.sin(theta) * np.sin(phi)
            vz = self.config.initial_velocity * np.cos(theta)
            
            velocities[i] = [vx, vy, vz]
    
    def _velocity_verlet_step(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """Однот методом Velocity Verlet.
        
        r(t + dt) = r(t) + v(t) * dt + 0.5 * a(t) * dt^2
        v(t + dt) = v(t) + 0.5 * (a(t) + a(t + dt)) * dt
        
        Но так как юсиловая функция константна (g), это принимает простую форму.
        """
        dt = self.config.dt
        g = self.config.g
        
        # Обновление позиций
        positions[:, 0] += velocities[:, 0] * dt  # x
        positions[:, 1] += velocities[:, 1] * dt  # y
        positions[:, 2] += velocities[:, 2] * dt - 0.5 * g * dt**2  # z with gravity
        
        # Обновление вертикальной компоненты скорости
        velocities[:, 2] -= g * dt  # v_z -= g * dt
    
    def _handle_collisions(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """Обработка столкновений со стенками и между молекулами."""
        # Столкновения со стенками
        self._handle_wall_collisions(positions, velocities)
        
        # Межмолекулярные столкновения
        self._handle_intermolecular_collisions(positions, velocities)
    
    def _handle_wall_collisions(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """Отражение от стенок сосуда."""
        # Боковые стенки (цилиндр)
        r = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        collided = r > self.config.container_radius
        
        if np.any(collided):
            # Нормальные компоненты вектора скорости
            cos_phi = positions[collided, 0] / r[collided]
            sin_phi = positions[collided, 1] / r[collided]
            
            v_r = velocities[collided, 0] * cos_phi + velocities[collided, 1] * sin_phi
            v_t = -velocities[collided, 0] * sin_phi + velocities[collided, 1] * cos_phi
            
            # Отражение радиальной компоненты
            v_r *= -1
            
            # Новые компоненты скорости
            velocities[collided, 0] = v_r * cos_phi - v_t * sin_phi
            velocities[collided, 1] = v_r * sin_phi + v_t * cos_phi
            
            # Перепозиционирование частиц
            positions[collided, 0] *= self.config.container_radius / r[collided]
            positions[collided, 1] *= self.config.container_radius / r[collided]
        
        # Дно (высота = 0)
        collided_floor = positions[:, 2] < 0
        velocities[collided_floor, 2] *= -1
        positions[collided_floor, 2] = 0
        
        # Крыша (высота = H)
        collided_ceiling = positions[:, 2] > self.config.container_height
        velocities[collided_ceiling, 2] *= -1
        positions[collided_ceiling, 2] = self.config.container_height
    
    def _handle_intermolecular_collisions(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """Обработка упругих столкновений между молекулами."""
        d_collision = self.config.particle_diameter
        
        for i in range(self.config.num_particles):
            for j in range(i + 1, self.config.num_particles):
                # Расстояние между частицами
                r = positions[j] - positions[i]
                dist = np.linalg.norm(r)
                
                # Проверка столкновения
                if dist < d_collision:
                    self.collisions_detected += 1
                    self._resolve_collision(i, j, positions, velocities, r)
            
            self.collision_pairs_checked += self.config.num_particles - i - 1
    
    def _resolve_collision(self, i: int, j: int, positions: np.ndarray, 
                          velocities: np.ndarray, r: np.ndarray) -> None:
        """Решение упругого столкновения молекул.
        
        По библиотеке: МОМЕНТУМ и КИНЕТИЧЕСКАЯ ЭНЕРГИЯ сохраняются.
        """
        dist = np.linalg.norm(r)
        if dist == 0:
            return
        
        # Единичный вектор надлинии центров
        n = r / dist
        
        # Относительная скорость
        v_rel = velocities[i] - velocities[j]
        v_rel_n = np.dot(v_rel, n)
        
        # Проверить, что частицы сближаются
        if v_rel_n >= 0:
            return
        
        # Для равных масс: опорные компоненты скоростей обменываются
        delta_v = v_rel_n * n
        
        velocities[i] -= delta_v
        velocities[j] += delta_v
        
        # Грубая распределение: несколько раз сюда-туда
        overlap = self.config.particle_diameter - dist
        if overlap > 0:
            shift = (overlap / 2 + 1e-6) * n
            positions[i] -= shift
            positions[j] += shift
    
    def _apply_thermostat(self, velocities: np.ndarray) -> None:
        """Применение баростата (Ммскалирование объемноя скорость)."""
        if not self.config.target_temperature:
            return
        
        # текущая температура
        kinetic_energy = 0.5 * self.config.particle_mass * np.sum(velocities**2) / self.config.num_particles
        current_temp = 2.0 * kinetic_energy / (3.0 * self.config.k_B)
        
        # Пантофактор для переарчализации
        if current_temp > 0:
            scale_factor = np.sqrt(self.config.target_temperature / current_temp)
            velocities *= scale_factor


class EquilibriumAnalyzer:
    """Анализ равновесных распределений."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def calculate_temperature(self, velocities: np.ndarray) -> float:
        """Рассчен температуры из среднего до нижнего центра."""
        kinetic_energy = 0.5 * self.config.particle_mass * np.sum(velocities**2) / self.config.num_particles
        return 2.0 * kinetic_energy / (3.0 * self.config.k_B)
    
    def calculate_pressure(self, positions: np.ndarray, velocities: np.ndarray, 
                          collisions_count: int) -> float:
        """Наивная оценка давления."""
        # P = n * k_B * T
        n_density = self.config.num_particles / (np.pi * self.config.container_radius**2 * self.config.container_height)
        T = self.calculate_temperature(velocities)
        return n_density * self.config.k_B * T
