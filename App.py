# path: digital_twin_app.py

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# =====================================================
# 1. MODELS
# =====================================================

class RacingVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(12, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 2)
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
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(6, 64, batch_first=True)
        self.fc = nn.Linear(64, 6)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        action = self.actor(x)
        value = self.critic(x)
        return action, value


# =====================================================
# 2. RL ENVIRONMENT
# =====================================================

class RacingEnv:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.max_step = len(df) - 1
        self.reset()

    def reset(self):
        self.step_idx = 0
        return self._get_state()

    def _get_state(self):
        row = self.df.iloc[self.step_idx]
        return np.array([
            row['vehicle_speed'],
            row['Brake_Pressure'],
            row['tire_energy'],
            row['long_acc'],
            row['slip_Wheel_Speed_FL']
        ], dtype=np.float32)

    def step(self, action):
        throttle, brake = action

        row = self.df.iloc[self.step_idx]

        speed_reward = row['vehicle_speed']
        stability_penalty = abs(row['slip_Wheel_Speed_FL']) * 50
        thermal_penalty = row['tire_energy'] * 0.001

        reward = speed_reward - stability_penalty - thermal_penalty

        self.step_idx += 1
        done = self.step_idx >= self.max_step

        return self._get_state(), reward, done, {}


class PPOAgent:
    def __init__(self):
        self.model = ActorCritic()
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        self.gamma = 0.99

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action, _ = self.model(state)
        return action.detach().numpy()

    def train(self, trajectories):
        states = torch.tensor([t[0] for t in trajectories], dtype=torch.float32)
        actions = torch.tensor([t[1] for t in trajectories], dtype=torch.float32)
        rewards = [t[2] for t in trajectories]

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)

        pred_actions, values = self.model(states)
        values = values.squeeze()

        advantage = returns - values.detach()

        loss_actor = ((pred_actions - actions)**2 * advantage.unsqueeze(1)).mean()
        loss_critic = (returns - values).pow(2).mean()

        loss = loss_actor + 0.5 * loss_critic

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# =====================================================
# 3. FEATURE ENGINEERING
# =====================================================

def compute_features(df):
    df = df.copy()

    wheels = ['Wheel_Speed_FL', 'Wheel_Speed_FR', 'Wheel_Speed_RL', 'Wheel_Speed_RR']
    df['vehicle_speed'] = df[wheels].mean(axis=1)

    for col in wheels:
        df[f'slip_{col}'] = (df[col] - df['vehicle_speed']) / (df['vehicle_speed'] + 1e-3)

    df['long_acc'] = df['Brake_Pressure'].diff().fillna(0)
    df['tire_energy'] = df['Brake_Pressure'] * df['vehicle_speed']

    return df


def detect_anomalies(df):
    anomalies = {}
    for col in ['Brake_Pressure', 'Damper_Pos']:
        z = (df[col] - df[col].mean()) / df[col].std()
        anomalies[col] = int((np.abs(z) > 3).sum())
    return anomalies


def recommend_setup(z):
    rec = []
    if z[0] > 1:
        rec.append("Reduce front stiffness")
    if z[1] < -1:
        rec.append("Increase tire pressure")
    if not rec:
        rec.append("Setup optimal")
    return rec


# =====================================================
# 4. STREAMLIT UI
# =====================================================

st.set_page_config(layout="wide")
st.title("🏁 AI Racing Digital Twin (RL-Enhanced)")

with st.sidebar:
    st.header("Chassis Setup")
    hp = st.number_input("HP", 500, 3000, 1200)
    kg = st.number_input("Mass", 500, 2500, 850)
    tire = st.number_input("Tire Rate", 100, 500, 280)
    aero = st.slider("Aero %", 30.0, 70.0, 42.0)

    file = st.file_uploader("Upload Telemetry CSV", type=["csv"])

# =====================================================
# 5. BASE MODEL INFERENCE
# =====================================================

vae = RacingVAE()
lstm = ThermalLSTM()
predictor = TelemetryPredictor()

input_vec = torch.tensor([[hp/3000, kg/2500, tire/500, aero/100] + [0.5]*8], dtype=torch.float32)

with torch.no_grad():
    z = vae(input_vec).numpy()[0]

# =====================================================
# 6. TABS
# =====================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Manifold",
    "Thermal",
    "Telemetry",
    "Optimization",
    "RL Agent"
])

# -------------------------------
# MANIFOLD
# -------------------------------

with tab1:
    fig, ax = plt.subplots()
    grid = np.linspace(-3, 3, 50)
    gx, gy = np.meshgrid(grid, grid)
    ax.contourf(gx, gy, np.exp(-(gx**2 + gy**2)))
    ax.scatter(z[0], z[1], c='red', s=200)
    st.pyplot(fig)

# -------------------------------
# TELEMETRY PIPELINE
# -------------------------------

if file:
    df = pd.read_csv(file)

    required = [
        'Wheel_Speed_FL','Wheel_Speed_FR',
        'Wheel_Speed_RL','Wheel_Speed_RR',
        'Brake_Pressure','Damper_Pos'
    ]

    if not all(c in df.columns for c in required):
        st.error("Missing telemetry columns")
        st.stop()

    df = compute_features(df)

    # Thermal
    with tab2:
        inp = torch.tensor(
            df[['Brake_Pressure','vehicle_speed','tire_energy']].values[-50:].reshape(1,50,3),
            dtype=torch.float32
        )
        with torch.no_grad():
            temp = lstm(inp).item()
        st.metric("Carcass Temp", f"{temp:.2f} °C")

    # Telemetry
    with tab3:
        fig, ax = plt.subplots()
        ax.plot(df['vehicle_speed'])
        ax.plot(df['Brake_Pressure'])
        st.pyplot(fig)

        st.write("Anomalies:", detect_anomalies(df))

    # Optimization
    with tab4:
        for r in recommend_setup(z):
            st.write("•", r)

        sim = torch.tensor(
            df[['vehicle_speed','Brake_Pressure','Damper_Pos',
                'tire_energy','long_acc','slip_Wheel_Speed_FL']].values[-30:].reshape(1,30,6),
            dtype=torch.float32
        )

        with torch.no_grad():
            future = predictor(sim).numpy()

        st.write("Future State:", future.tolist())

    # RL AGENT
    with tab5:
        env = RacingEnv(df)
        agent = PPOAgent()

        state = env.reset()
        traj = []

        for _ in range(200):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            traj.append((state, action, reward))
            state = next_state

            if done:
                break

        agent.train(traj)

        optimal = agent.select_action(env.reset())

        st.subheader("Optimal Control")
        st.write({
            "Throttle": float(optimal[0]),
            "Brake": float(optimal[1])
        })

else:
    st.warning("Upload telemetry to activate full system")
