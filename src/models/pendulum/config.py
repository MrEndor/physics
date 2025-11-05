PHYSICAL_CONSTANTS = {
    "g": 9.81,
}

DEFAULT_PENDULUM_TYPES = {
    "point_mass": {
        "description": "Точечная масса на стержне",
        "I_ratio": 1.0,
        "l_ratio": 1.0,
    },
    "uniform_rod": {
        "description": "Однородный стержень",
        "I_ratio": 1.0 / 3.0,
        "l_ratio": 0.5,
    },
    "thin_disk": {
        "description": "Тонкий диск на стержне",
        "I_ratio": 1.5,
        "l_ratio": 1.0,
    },
}
