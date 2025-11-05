import altair as alt
import numpy as np
import pandas as pd

SMALL_AMPLITUDE_DEG = 5


def create_oscillation_chart(phi: np.ndarray, t: np.ndarray) -> alt.Chart:
    df = pd.DataFrame({"t": t, "phi_deg": np.degrees(phi), "phi_rad": phi})

    return (
        alt.Chart(df)
        .mark_line(strokeWidth=2, color="blue")
        .add_params(alt.selection_interval(bind="scales"))
        .encode(
            x=alt.X("t:Q", title="Время t (с)"),
            y=alt.Y("phi_deg:Q", title="Угол φ (градусы)"),
            tooltip=["t:Q", "phi_deg:Q", "phi_rad:Q"],
        )
        .properties(title="Колебания маятника", width=600, height=300)
    )


def create_phase_portrait(phi: np.ndarray, omega: np.ndarray) -> alt.Chart:
    df = pd.DataFrame({"phi_deg": np.degrees(phi), "omega": omega})

    return (
        alt.Chart(df)
        .mark_line(
            strokeWidth=2,
            color="red",
            point=alt.OverlayMarkDef(size=20, opacity=0.3),
        )
        .add_params(alt.selection_interval(bind="scales"))
        .encode(
            x=alt.X("phi_deg:Q", title="Угол φ (градусы)"),
            y=alt.Y("omega:Q", title="Угловая скорость ω (рад/с)"),
            tooltip=["phi_deg:Q", "omega:Q"],
        )
        .properties(title="Фазовый портрет", width=400, height=400)
    )


def create_energy_chart(
    kinetic: np.ndarray, potential: np.ndarray, total: np.ndarray, t: np.ndarray
) -> alt.Chart:
    df = pd.DataFrame({
        "t": t,
        "Кинетическая": kinetic,
        "Потенциальная": potential,
        "Полная": total,
    })

    df_melted = df.melt("t", var_name="Тип энергии", value_name="Энергия (Дж)")

    return (
        alt.Chart(df_melted)
        .mark_line(strokeWidth=2)
        .add_params(alt.selection_interval(bind="scales"))
        .encode(
            x=alt.X("t:Q", title="Время t (с)"),
            y=alt.Y("Энергия (Дж):Q", title="Энергия (Дж)"),
            color=alt.Color(
                "Тип энергии:N", scale=alt.Scale(scheme="category10")
            ),
            tooltip=["t:Q", "Тип энергии:N", "Энергия (Дж):Q"],
        )
        .properties(title="Энергия системы", width=600, height=300)
    )


def create_amplitude_analysis_chart(amplitudes: dict) -> alt.Chart:
    data = []
    for amp, period in amplitudes.items():
        period_ratio = (
            period / amplitudes[SMALL_AMPLITUDE_DEG]
            if SMALL_AMPLITUDE_DEG in amplitudes
            else 1.0
        )
        data.append({
            "Амплитуда (град)": amp,
            "Период (с)": period,
            "Период/T₀": period_ratio,
        })

    df = pd.DataFrame(data)

    chart1 = (
        alt.Chart(df)
        .mark_circle(size=100, opacity=0.8)
        .encode(
            x=alt.X(
                "Амплитуда (град):Q", title="Начальная амплитуда φ₀ (градусы)"
            ),
            y=alt.Y("Период (с):Q", title="Период T (с)"),
            color=alt.value("blue"),
            tooltip=["Амплитуда (град):Q", "Период (с):Q", "Период/T₀:Q"],
        )
        .properties(
            title="Зависимость периода от амплитуды", width=400, height=300
        )
    )

    if SMALL_AMPLITUDE_DEG in amplitudes:
        t0 = amplitudes[SMALL_AMPLITUDE_DEG]
        theory_data = []
        for amp in range(SMALL_AMPLITUDE_DEG, 121, SMALL_AMPLITUDE_DEG):
            phi_rad = np.radians(amp)
            t_theory = t0 * (1 + phi_rad**2 / 16)
            theory_data.append({"Амплитуда (град)": amp, "Теория": t_theory})

        theory_df = pd.DataFrame(theory_data)
        theory_line = (
            alt.Chart(theory_df)
            .mark_line(color="red", strokeDash=[5, 5], strokeWidth=2)
            .encode(
                x="Амплитуда (град):Q",
                y=alt.Y("Теория:Q", title="Период T (с)"),
            )
        )

        chart1 += theory_line

    return chart1


def create_damping_analysis_chart(damping_data: dict) -> alt.Chart:
    all_data = []

    for gamma, (phi, t) in damping_data.items():
        phi_deg = np.degrees(phi)
        all_data.extend([
            {
                "t": t[i],
                "phi": phi_deg[i],
                "gamma": f"γ = {gamma} с⁻¹",
            }
            for i in range(len(t))
        ])

    df = pd.DataFrame(all_data)

    return (
        alt.Chart(df)
        .mark_line(strokeWidth=2)
        .add_params(alt.selection_interval(bind="scales"))
        .encode(
            x=alt.X("t:Q", title="Время t (с)"),
            y=alt.Y("phi:Q", title="Угол φ (градусы)"),
            color=alt.Color("gamma:N", title="Коэффициент трения"),
            tooltip=["t:Q", "phi:Q", "gamma:N"],
        )
        .properties(
            title="Затухающие колебания при разном трении",
            width=600,
            height=400,
        )
    )
