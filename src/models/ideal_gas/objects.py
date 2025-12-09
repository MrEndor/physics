import numpy as np

from models.ideal_gas.config import SimulationConfig


class IdealGasSimulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.num_steps = int(config.simulation_time / config.dt)

        self.collision_pairs_checked = 0
        self.collisions_detected = 0

        self.cell_size = config.particle_diameter * 2.5
        self.cells_per_axis = int(config.container_radius / self.cell_size) + 1
        self.cells = {}

    def run(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.zeros((self.config.num_particles, 3), dtype=np.float32)
        velocities = np.zeros((self.config.num_particles, 3), dtype=np.float32)
        self._initialize_particles(positions, velocities)

        positions_history = []
        velocities_history = []
        times = []

        for step in range(self.num_steps):
            t = step * self.config.dt
            times.append(t)

            positions_history.append(positions.copy())
            velocities_history.append(velocities.copy())

            self._velocity_verlet_step(positions, velocities)

            collided_with_walls = self._handle_collisions(positions, velocities)

            if (
                self.config.use_thermostat
                and self.config.target_temperature is not None
            ):
                self._apply_thermostat_at_wall(velocities, collided_with_walls)

        return (
            np.array(positions_history),
            np.array(velocities_history),
            np.array(times),
        )

    def _initialize_particles(
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> None:
        num_particles = self.config.num_particles

        theta = np.random.uniform(0, 2 * np.pi, num_particles)
        radius_factor = self.config.container_radius * 0.9
        r = np.sqrt(np.random.uniform(0, 1, num_particles)) * radius_factor

        positions[:, 0] = r * np.cos(theta)
        positions[:, 1] = r * np.sin(theta)
        positions[:, 2] = np.random.uniform(
            0, self.config.initial_height, num_particles
        )

        phi = np.random.uniform(0, 2 * np.pi, num_particles)
        cos_theta = np.random.uniform(-1, 1, num_particles)
        sin_theta = np.sqrt(1 - cos_theta**2)

        v0 = self.config.initial_velocity
        velocities[:, 0] = v0 * sin_theta * np.cos(phi)
        velocities[:, 1] = v0 * sin_theta * np.sin(phi)
        velocities[:, 2] = v0 * cos_theta

    def _velocity_verlet_step(
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> None:
        dt = self.config.dt
        g = self.config.g

        positions[:, 0] += velocities[:, 0] * dt
        positions[:, 1] += velocities[:, 1] * dt

        positions[:, 2] += velocities[:, 2] * dt - 0.5 * g * dt**2

        velocities[:, 2] -= g * dt

    def _handle_collisions(
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> np.ndarray:
        collided_with_walls = self._handle_wall_collisions(
            positions, velocities
        )

        self._handle_intermolecular_collisions_optimized(positions, velocities)

        return collided_with_walls

    def _handle_wall_collisions(
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> np.ndarray:
        collided_indices = []

        r = np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)
        collided = r > self.config.container_radius

        if np.any(collided):
            collided_indices.extend(np.where(collided)[0])
            cos_phi = np.divide(
                positions[collided, 0],
                r[collided],
                where=r[collided] != 0,
                out=np.zeros_like(positions[collided, 0]),
            )
            sin_phi = np.divide(
                positions[collided, 1],
                r[collided],
                where=r[collided] != 0,
                out=np.zeros_like(positions[collided, 1]),
            )

            v_r = (
                velocities[collided, 0] * cos_phi
                + velocities[collided, 1] * sin_phi
            )
            v_t = (
                -velocities[collided, 0] * sin_phi
                + velocities[collided, 1] * cos_phi
            )

            v_r *= -1

            velocities[collided, 0] = v_r * cos_phi - v_t * sin_phi
            velocities[collided, 1] = v_r * sin_phi + v_t * cos_phi

            positions[collided, 0] *= self.config.container_radius / r[collided]
            positions[collided, 1] *= self.config.container_radius / r[collided]

        collided_floor = positions[:, 2] < 0
        if np.any(collided_floor):
            collided_indices.extend(np.where(collided_floor)[0])
            velocities[collided_floor, 2] *= -1
            positions[collided_floor, 2] = 0

        collided_ceiling = positions[:, 2] > self.config.container_height
        if np.any(collided_ceiling):
            collided_indices.extend(np.where(collided_ceiling)[0])
            velocities[collided_ceiling, 2] *= -1
            positions[collided_ceiling, 2] = self.config.container_height

        return np.unique(collided_indices)

    def _handle_intermolecular_collisions_optimized(  # noqa: PLR0912
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> None:
        self.cells.clear()

        for i, pos in enumerate(positions):
            ix = int((pos[0] + self.config.container_radius) / self.cell_size)
            iy = int((pos[1] + self.config.container_radius) / self.cell_size)
            iz = int(pos[2] / self.cell_size)

            ix = max(0, min(ix, 2 * self.cells_per_axis - 1))
            iy = max(0, min(iy, 2 * self.cells_per_axis - 1))
            iz = max(0, min(iz, self.cells_per_axis))

            cell_key = (ix, iy, iz)
            if cell_key not in self.cells:
                self.cells[cell_key] = []
            self.cells[cell_key].append(i)

        d_collision = self.config.particle_diameter
        checked_pairs = set()

        for cell_key, particles in self.cells.items():  # noqa: PLR1702
            ix, iy, iz = cell_key

            for i in range(len(particles)):
                for j in range(i + 1, len(particles)):
                    p1, p2 = particles[i], particles[j]
                    pair = (min(p1, p2), max(p1, p2))
                    if pair not in checked_pairs:
                        checked_pairs.add(pair)
                        self._check_and_resolve_collision(
                            p1, p2, positions, velocities, d_collision
                        )

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
                                            self._check_and_resolve_collision(
                                                p1,
                                                p2,
                                                positions,
                                                velocities,
                                                d_collision,
                                            )

    def _check_and_resolve_collision(
        self,
        i: int,
        j: int,
        positions: np.ndarray,
        velocities: np.ndarray,
        d_collision: float,
    ) -> None:
        r = positions[j] - positions[i]
        dist_sq = np.sum(r**2)

        if dist_sq < d_collision**2:
            dist = np.sqrt(dist_sq)
            if dist > 0:
                self.collisions_detected += 1
                self._resolve_collision_inline(
                    i, j, positions, velocities, r, dist
                )

    def _resolve_collision_inline(
        self,
        i: int,
        j: int,
        positions: np.ndarray,
        velocities: np.ndarray,
        r: np.ndarray,
        dist: float,
    ) -> None:
        n = r / dist

        v_rel = velocities[i] - velocities[j]
        v_rel_n = np.dot(v_rel, n)

        if v_rel_n >= 0:
            return

        delta_v = v_rel_n * n
        velocities[i] -= delta_v
        velocities[j] += delta_v

        overlap = self.config.particle_diameter - dist
        if overlap > 0:
            shift = (overlap / 2 + 1e-6) * n
            positions[i] -= shift
            positions[j] += shift

    def _apply_thermostat_at_wall(
        self, velocities: np.ndarray, collided_indices: np.ndarray
    ) -> None:
        if not self.config.target_temperature or len(collided_indices) == 0:
            return

        sigma = np.sqrt(
            self.config.k_B
            * self.config.target_temperature
            / self.config.particle_mass
        )
        velocities[collided_indices] = np.random.normal(
            0, sigma, size=(len(collided_indices), 3)
        )


class EquilibriumAnalyzer:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def calculate_temperature(self, velocities: np.ndarray) -> float:
        kinetic_energy = (
            0.5
            * self.config.particle_mass
            * np.sum(velocities**2)
            / self.config.num_particles
        )
        return 2.0 * kinetic_energy / (3.0 * self.config.k_B)

    def calculate_pressure(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        collisions_count: int,
    ) -> float:
        n_density = self.config.num_particles / (
            np.pi
            * self.config.container_radius**2
            * self.config.container_height
        )
        temperature = self.calculate_temperature(velocities)
        return n_density * self.config.k_B * temperature
