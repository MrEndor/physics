from dataclasses import dataclass

MAX_DT = 0.01


@dataclass
class SimulationConfig:
    num_particles: int

    container_height: float
    container_radius: float

    initial_velocity: float

    simulation_time: float
    dt: float

    use_thermostat: bool = False
    target_temperature: float | None = None

    g: float = 9.81
    k_B: float = 1.38e-23  # noqa: N815

    initial_height: float = 0.1
    particle_mass: float = 6.63e-26
    particle_diameter: float = 3.4e-10

    def __post_init__(self):
        if self.num_particles <= 0:
            raise ValueError("Количество молекул должно быть положительным")
        if self.dt <= 0 or self.dt > MAX_DT:
            raise ValueError(f"Шаг времени должен быть <= {MAX_DT}")
        if self.initial_height > self.container_height:
            raise ValueError(
                "Начальная высота не может быть больше высоты сосуда"
            )
