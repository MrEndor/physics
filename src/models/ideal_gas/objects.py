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
    
    Оптимизации:
        - Нумпи векторизация для быстрых вычислений
        - Cell list для сокращения столкновений O(N) вместо O(N^2)
        - Оптимизированная проверка столкновений
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.num_steps = int(config.simulation_time / config.dt)
        
        # Объекты для столкновений
        self.collision_pairs_checked = 0
        self.collisions_detected = 0
        
        # Cell list для акселерации столкновений
        self.cell_size = config.particle_diameter * 2.5
        self.cells_per_axis = int(config.container_radius / self.cell_size) + 1
        self.cells = {}
    
    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Запустить симуляцию и вернуть историю позиций и скоростей.
        
        Returns:
            positions_history: История позиций (num_steps, num_particles, 3)
            velocities_history: История скоростей (num_steps, num_particles, 3)
            times: Вектор времен
        """
        # Инициализация
        positions = np.zeros((self.config.num_particles, 3), dtype=np.float32)
        velocities = np.zeros((self.config.num_particles, 3), dtype=np.float32)
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
        # Оптимизированная векторизированная инициализация
        N = self.config.num_particles
        
        # Генерирую случайные позиции в цилиндре
        theta = np.random.uniform(0, 2 * np.pi, N)
        r = np.sqrt(np.random.uniform(0, 1, N)) * self.config.container_radius * 0.9
        
        positions[:, 0] = r * np.cos(theta)
        positions[:, 1] = r * np.sin(theta)
        positions[:, 2] = np.random.uniform(0, self.config.initial_height, N)
        
        # Генерирую скорости со случайными направлениями
        phi = np.random.uniform(0, 2 * np.pi, N)
        cos_theta = np.random.uniform(-1, 1, N)
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        velocities[:, 0] = self.config.initial_velocity * sin_theta * np.cos(phi)
        velocities[:, 1] = self.config.initial_velocity * sin_theta * np.sin(phi)
        velocities[:, 2] = self.config.initial_velocity * cos_theta
    
    def _velocity_verlet_step(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """Однот методом Velocity Verlet (векторизированный).
        
        r(t + dt) = r(t) + v(t) * dt + 0.5 * a(t) * dt^2
        v(t + dt) = v(t) + 0.5 * (a(t) + a(t + dt)) * dt
        """
        dt = self.config.dt
        g = self.config.g
        
        # Обновление позиций (векторизированно)
        positions[:, 0] += velocities[:, 0] * dt
        positions[:, 1] += velocities[:, 1] * dt
        positions[:, 2] += velocities[:, 2] * dt - 0.5 * g * dt**2
        
        # Обновление вертикальной компоненты скорости (векторизированно)
        velocities[:, 2] -= g * dt
    
    def _handle_collisions(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """Обработка столкновений со стенками и между молекулами."""
        # Столкновения со стенками
        self._handle_wall_collisions(positions, velocities)
        
        # Межмолекулярные столкновения с cell list оптимизацией
        self._handle_intermolecular_collisions_optimized(positions, velocities)
    
    def _handle_wall_collisions(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """Отражение от стенок сосуда (векторизированно)."""
        # Боковые стенки (цилиндр)
        r = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        collided = r > self.config.container_radius
        
        if np.any(collided):
            # Нормальные компоненты вектора скорости (векторизированно)
            cos_phi = np.divide(positions[collided, 0], r[collided], where=r[collided]!=0, out=np.zeros_like(positions[collided, 0]))
            sin_phi = np.divide(positions[collided, 1], r[collided], where=r[collided]!=0, out=np.zeros_like(positions[collided, 1]))
            
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
        
        # Дно и крыша (векторизированно)
        collided_floor = positions[:, 2] < 0
        velocities[collided_floor, 2] *= -1
        positions[collided_floor, 2] = 0
        
        collided_ceiling = positions[:, 2] > self.config.container_height
        velocities[collided_ceiling, 2] *= -1
        positions[collided_ceiling, 2] = self.config.container_height
    
    def _handle_intermolecular_collisions_optimized(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """Обработка столкновений с cell list оптимизацией O(N) вместо O(N^2)."""
        # Очистить налицы мячей
        self.cells.clear()
        
        # Рассортировать частицы по мячеям
        for i, pos in enumerate(positions):
            # Определить индекс ячейки
            ix = int((pos[0] + self.config.container_radius) / self.cell_size)
            iy = int((pos[1] + self.config.container_radius) / self.cell_size)
            iz = int(pos[2] / self.cell_size)
            
            # Пограничить индексы
            ix = max(0, min(ix, 2 * self.cells_per_axis - 1))
            iy = max(0, min(iy, 2 * self.cells_per_axis - 1))
            iz = max(0, min(iz, self.cells_per_axis))
            
            cell_key = (ix, iy, iz)
            if cell_key not in self.cells:
                self.cells[cell_key] = []
            self.cells[cell_key].append(i)
        
        # Проверить столкновения лишь в соседних ячейках
        d_collision = self.config.particle_diameter
        checked_pairs = set()
        
        for cell_key, particles in self.cells.items():
            ix, iy, iz = cell_key
            
            # Проверить в текущей ячейке
            for i in range(len(particles)):
                for j in range(i + 1, len(particles)):
                    p1, p2 = particles[i], particles[j]
                    pair = (min(p1, p2), max(p1, p2))
                    if pair not in checked_pairs:
                        checked_pairs.add(pair)
                        self._check_and_resolve_collision(p1, p2, positions, velocities, d_collision)
            
            # Проверить соседние ячейки
            for dix in [-1, 0, 1]:
                for diy in [-1, 0, 1]:
                    for diz in [-1, 0, 1]:
                        if dix == 0 and diy == 0 and diz == 0:
                            continue
                        
                        neighbor_key = (ix + dix, iy + diy, iz + diz)
                        if neighbor_key in self.cells:
                            for p1 in particles:
                                for p2 in self.cells[neighbor_key]:
                                    if p1 < p2:
                                        pair = (p1, p2)
                                        if pair not in checked_pairs:
                                            checked_pairs.add(pair)
                                            self._check_and_resolve_collision(p1, p2, positions, velocities, d_collision)
    
    def _check_and_resolve_collision(self, i: int, j: int, positions: np.ndarray, 
                                    velocities: np.ndarray, d_collision: float) -> None:
        """Проверить и решить столкновение одного пары молекул."""
        r = positions[j] - positions[i]
        dist_sq = np.sum(r**2)
        
        if dist_sq < d_collision**2:
            dist = np.sqrt(dist_sq)
            if dist > 0:
                self.collisions_detected += 1
                self._resolve_collision_inline(i, j, positions, velocities, r, dist)
    
    def _resolve_collision_inline(self, i: int, j: int, positions: np.ndarray, 
                                 velocities: np.ndarray, r: np.ndarray, dist: float) -> None:
        """Решение упругого столкновения (выполнено встроенные)."""
        # Единичный вектор
        n = r / dist
        
        # Относительная скорость
        v_rel = velocities[i] - velocities[j]
        v_rel_n = np.dot(v_rel, n)
        
        # Проверь, что частицы сближаются
        if v_rel_n >= 0:
            return
        
        # Обмен скоростей
        delta_v = v_rel_n * n
        velocities[i] -= delta_v
        velocities[j] += delta_v
        
        # Отсылка
        overlap = self.config.particle_diameter - dist
        if overlap > 0:
            shift = (overlap / 2 + 1e-6) * n
            positions[i] -= shift
            positions[j] += shift
    
    def _apply_thermostat(self, velocities: np.ndarray) -> None:
        """Применение баростата."""
        if not self.config.target_temperature:
            return
        
        kinetic_energy = 0.5 * self.config.particle_mass * np.sum(velocities**2) / self.config.num_particles
        current_temp = 2.0 * kinetic_energy / (3.0 * self.config.k_B)
        
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
        n_density = self.config.num_particles / (np.pi * self.config.container_radius**2 * self.config.container_height)
        T = self.calculate_temperature(velocities)
        return n_density * self.config.k_B * T
