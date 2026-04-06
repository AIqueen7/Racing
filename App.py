# path: app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# -------------------------------
# 1. MODELS
# -------------------------------

class RacingVAE(nn.Module):
    def __init__(self, input_dim=12, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class ThermalLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class TelemetryPredictor(nn.Module):
    """Short horizon prediction (future state simulation)"""
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(6, 64, batch_first=True)
        self.fc = nn.Linear(64, 6)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])


# -------------------------------
# 2. FEATURE ENGINEERING
# -------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Slip ratio approximation
    wheel_cols = ['Wheel_Speed_FL', 'Wheel_Speed_FR', 'Wheel_Speed_RL', 'Wheel_Speed_RR']
    df['vehicle_speed'] = df[wheel_cols].mean(axis=1)

    for col in wheel_cols:
        df[f'slip_{col}'] = (df[col] - df['vehicle_speed']) / (df['vehicle_speed'] + 1e-3)

    # Longitudinal load proxy
    df['long_acc'] = df['Brake_Pressure'].diff().fillna(0)

    # Tire energy proxy
    df['tire_energy'] = df['Brake_Pressure'] * df['vehicle_speed']

    return df


# -------------------------------
# 3. OPTIMIZATION ENGINE
# -------------------------------

def recommend_setup(z):
    rec = []

    if z[0] > 1.0:
        rec.append("Reduce front stiffness or increase rear aero balance")

    if z[1] < -1.0:
        rec.append("Increase tire pressure or reduce camber")

    if not rec:
        rec.append("Setup within optimal manifold window")

    return rec


# -------------------------------
# 4. ANOMALY DETECTION
# -------------------------------

def detect_anomalies(df):
    anomalies = {}
    for col in ['Brake_Pressure', 'Damper_Pos']:
        z = (df[col] - df[col].mean()) / df[col].std()
        anomalies[col] = (np.abs(z) > 3).sum()
    return anomalies


# -------------------------------
# 5. STREAMLIT UI
# -------------------------------

st.set_page_config(layout="wide")
st.title("🏁 AI Racing Digital Twin")

with st.sidebar:
    st.header("Chassis Parameters")
    hp = st.number_input("HP", 500, 3000, 1200)
    kg = st.number_input("Mass (kg)", 500, 2500, 850)
    tire_rate = st.number_input("Tire Rate", 100, 500, 280)
    aero = st.slider("Aero Balance %", 30.0, 70.0, 42.0)

    uploaded = st.file_uploader("Upload Telemetry CSV", type=["csv"])

# -------------------------------
# 6. MODEL INIT
# -------------------------------

vae = RacingVAE()
lstm = ThermalLSTM()
predictor = TelemetryPredictor()

# -------------------------------
# 7. BASE INPUT VECTOR
# -------------------------------

input_vec = torch.tensor([[
    hp/3000, kg/2500, tire_rate/500, aero/100,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
]], dtype=torch.float32)

with torch.no_grad():
    z = vae(input_vec).numpy()[0]

# -------------------------------
# 8. TABS
# -------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Manifold",
    "Thermal",
    "Telemetry",
    "Optimization"
])

# -------------------------------
# MANIFOLD
# -------------------------------

with tab1:
    st.subheader("Latent Setup Space")

    fig, ax = plt.subplots()
    grid = np.linspace(-3, 3, 50)
    gx, gy = np.meshgrid(grid, grid)
    ax.contourf(gx, gy, np.exp(-(gx**2 + gy**2)))

    ax.scatter(z[0], z[1], c='red', s=200)
    st.pyplot(fig)

# -------------------------------
# TELEMETRY INGESTION
# -------------------------------

if uploaded:
    df = pd.read_csv(uploaded)

    required_cols = [
        'Wheel_Speed_FL', 'Wheel_Speed_FR',
        'Wheel_Speed_RL', 'Wheel_Speed_RR',
        'Brake_Pressure', 'Damper_Pos'
    ]

    if not all(col in df.columns for col in required_cols):
        st.error("Missing required telemetry columns")
        st.stop()

    df = compute_features(df)

    # ---------------------------
    # THERMAL MODEL
    # ---------------------------

    with tab2:
        st.subheader("Thermal Prediction")

        thermal_input = torch.tensor(
            df[['Brake_Pressure', 'vehicle_speed', 'tire_energy']].values[-50:].reshape(1, 50, 3),
            dtype=torch.float32
        )

        with torch.no_grad():
            temp = lstm(thermal_input).item()

        st.metric("Predicted Carcass Temp", f"{temp:.2f} °C")

    # ---------------------------
    # TELEMETRY ANALYSIS
    # ---------------------------

    with tab3:
        st.subheader("Telemetry Features")

        fig, ax = plt.subplots()
        ax.plot(df['vehicle_speed'], label="Speed")
        ax.plot(df['Brake_Pressure'], label="Brake")
        ax.legend()
        st.pyplot(fig)

        anomalies = detect_anomalies(df)
        st.write("Anomalies:", anomalies)

    # ---------------------------
    # OPTIMIZATION
    # ---------------------------

    with tab4:
        st.subheader("Setup Recommendations")

        recs = recommend_setup(z)
        for r in recs:
            st.write("•", r)

        # future state simulation
        sim_input = torch.tensor(
            df[['vehicle_speed', 'Brake_Pressure', 'Damper_Pos',
                'tire_energy', 'long_acc', 'slip_Wheel_Speed_FL']].values[-30:].reshape(1, 30, 6),
            dtype=torch.float32
        )

        with torch.no_grad():
            future = predictor(sim_input).numpy()

        st.write("Predicted next state:", future.tolist())

else:
    st.warning("Upload telemetry to activate full digital twin")
