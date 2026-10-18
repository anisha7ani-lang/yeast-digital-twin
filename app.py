import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import minimize

st.set_page_config(
    page_title="Yeast Digital Twin",
    page_icon="🧪",
    layout="wide"
)

# --------------------------------------------------
# 1. MODEL PARAMETERS
# Illustrative values for a software prototype only
# --------------------------------------------------
PARAMS = {
    "mu_max": 0.30,
    "Ks": 1.0,
    "Yxs": 0.50,
    "Yps": 0.40,
    "Yco2s": 0.40,
    "feed_sugar": 100.0
}

INITIAL_STATE = [20.0, 1.0, 0.0, 0.5, 0.0]
# State order: glucose, biomass, ethanol estimate, volume, cumulative CO2 estimate


# --------------------------------------------------
# 2. FERMENTATION MODEL
# --------------------------------------------------
def fermentation_model(t, y, feed_rate_L_h, p):
    S, X, ethanol, V, co2 = y

    V = max(V, 0.01)
    S = max(S, 0.0)
    X = max(X, 0.0)

    mu = p["mu_max"] * S / (p["Ks"] + S)
    qs = mu * X / p["Yxs"]
    D = feed_rate_L_h / V

    dS = D * (p["feed_sugar"] - S) - qs
    dX = mu * X - D * X
    dEthanol = p["Yps"] * qs - D * ethanol
    dV = feed_rate_L_h
    dCO2 = p["Yco2s"] * qs * V

    return [dS, dX, dEthanol, dV, dCO2]


def simulate_constant_feed(feed_ml_h, duration_h, initial_state=None):
    if initial_state is None:
        initial_state = INITIAL_STATE.copy()

    feed_L_h = feed_ml_h / 1000.0
    times = np.linspace(0, duration_h, int(duration_h * 4) + 1)

    solution = solve_ivp(
        lambda t, y: fermentation_model(t, y, feed_L_h, PARAMS),
        [0, duration_h],
        initial_state,
        t_eval=times,
        max_step=0.25
    )

    df = pd.DataFrame({
        "time_h": solution.t,
        "glucose_g_L": solution.y[0],
        "biomass_g_L": solution.y[1],
        "ethanol_est_g_L": solution.y[2],
        "volume_L": solution.y[3],
        "co2_est_cumulative_g": solution.y[4]
    })

    return df


# --------------------------------------------------
# 3. MPC-STYLE FEED OPTIMIZER
# Optimizes four hourly feed-rate decisions
# --------------------------------------------------
def run_mpc():
    horizon_h = 4
    max_feed = 10.0
    max_volume = 0.75
    max_glucose = 30.0

    def advance_one_hour(state, feed_ml_h):
        sol = solve_ivp(
            lambda t, y: fermentation_model(
                t, y, feed_ml_h / 1000.0, PARAMS
            ),
            [0, 1],
            state,
            t_eval=[1],
            max_step=0.1
        )
        return sol.y[:, -1]

    def objective(feed_sequence):
        state = np.array(INITIAL_STATE, dtype=float)

        for feed in feed_sequence:
            state = advance_one_hour(state, feed)

        glucose, biomass, ethanol, volume, co2 = state
        ethanol_mass = ethanol * volume

        penalty = (
            1000 * max(0, volume - max_volume) ** 2
            + 100 * max(0, glucose - max_glucose) ** 2
        )

        # Small penalty discourages unnecessarily high feeding.
        return -ethanol_mass + penalty + 0.001 * np.sum(feed_sequence ** 2)

    result = minimize(
        objective,
        x0=np.full(horizon_h, 2.0),
        method="L-BFGS-B",
        bounds=[(0.0, max_feed)] * horizon_h
    )

    return result.x, result.success


# --------------------------------------------------
# 4. RLS DEMONSTRATION USING SYNTHETIC OBSERVATIONS
# --------------------------------------------------
def run_rls_demo():
    demo = simulate_constant_feed(5.0, 24)
    rng = np.random.default_rng(42)

    demo["synthetic_observed_biomass_g_L"] = np.maximum(
        demo["biomass_g_L"] + rng.normal(0, 0.03, len(demo)),
        0.001
    )

    theta = 0.15
    covariance = 10.0
    forgetting_factor = 0.98
    estimates = []

    for i in range(1, len(demo)):
        dt = demo.loc[i, "time_h"] - demo.loc[i - 1, "time_h"]

        x_prev = demo.loc[i - 1, "synthetic_observed_biomass_g_L"]
        x_now = demo.loc[i, "synthetic_observed_biomass_g_L"]
        glucose_prev = max(demo.loc[i - 1, "glucose_g_L"], 0.001)
        volume_prev = max(demo.loc[i - 1, "volume_L"], 0.01)

        dilution = 0.005 / volume_prev  # 5 mL/h demo feed
        apparent_growth = np.log(x_now / x_prev) / dt + dilution
        phi = glucose_prev / (PARAMS["Ks"] + glucose_prev)

        gain = (
            covariance * phi
            / (forgetting_factor + phi * covariance * phi)
        )

        theta = np.clip(
            theta + gain * (apparent_growth - phi * theta),
            0.001,
            1.0
        )

        covariance = (
            covariance - gain * phi * covariance
        ) / forgetting_factor

        estimates.append({
            "time_h": demo.loc[i, "time_h"],
            "mu_max_estimated": theta
        })

    return demo, pd.DataFrame(estimates)


# --------------------------------------------------
# 5. DASHBOARD USER INTERFACE
# --------------------------------------------------
st.title("🧪 Yeast Fermentation Digital Twin")
st.caption(
    "Software prototype | SIMULATION MODE — no physical sensors or pump connected"
)

st.warning(
    "All displayed values are model-generated estimates, not experimental "
    "measurements. Validate the model and safety limits before any hardware control."
)

with st.sidebar:
    st.header("Simulation settings")
    feed_rate = st.slider(
        "Constant feed rate (mL/h)", 0.0, 10.0, 5.0, 0.5
    )
    duration = st.slider(
        "Simulation duration (hours)", 4, 48, 24, 4
    )
    st.markdown("---")
    st.write("Illustrative model parameters")
    st.write(f"Maximum growth rate: {PARAMS['mu_max']} 1/h")
    st.write(f"Feed sugar concentration: {PARAMS['feed_sugar']} g/L")
    st.write(f"Initial volume: {INITIAL_STATE[3]} L")
