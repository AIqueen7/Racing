# path: optimized_digital_twin_v2.py

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# =====================================================
# TRACK + PHYSICS
# =====================================================

class Track:
    def __init__(self):
        self.segments = [
            {"type": "straight", "length": 200},
            {"type": "corner", "radius": 50},
            {"type": "straight", "length": 300},
            {"type": "corner", "radius": 30},
            {"type": "straight", "length": 250},
        ]

    def get_segment(self, idx):
        return self.segments[idx % len(self.segments)]


def max_corner_speed(radius, mu=1.2):
    return np.sqrt(mu * 9.81 * radius)


# =====================================================
# ENVIRONMENT
# =====================================================

class AdvancedRacingEnv:
    def __init__(self):
        self.track = Track()
        self.reset()

    def reset(self):
        self.speed = 5.0  # FIX: avoid dead start
        self.position = 0.0
        self.segment_idx = 0
        self.time = 0.0
        return self._get_state()

    def _get_state(self):
        seg = self.track.get_segment(self.segment_idx)
        curvature = 0.0 if seg["type"] == "straight" else 1.0 / seg["radius"]

        return np.array([
            self.speed,
            curvature,
            self.time,
            self.segment_idx,
            self.speed * curvature
        ], dtype=np.float32)

    def step(self, action):
        throttle_raw, brake_raw = action

        # FIX: proper control scaling
        throttle = (throttle_raw + 1) / 2  # [-1,1] → [0,1]
        brake = max(0, brake_raw)

        accel = throttle * 6 - brake * 10 - 0.02 * self.speed**2
        self.speed = max(0, self.speed + accel * 0.1)

        seg = self.track.get_segment(self.segment_idx)

        grip_penalty = 0.0
        if seg["type"] == "corner":
            vmax = max_corner_speed(seg["radius"])
            if self.speed > vmax:
                grip_penalty = (self.speed - vmax)**2
                self.speed *= 0.9

        self.position += self.speed * 0.1
        self.time += 0.1

        if seg["type"] == "straight" and self.position > seg["length"]:
            self.position = 0
            self.segment_idx += 1
        elif seg["type"] == "corner" and self.position > seg["radius"] * np.pi / 2:
            self.position = 0
            self.segment_idx += 1

        done = self.segment_idx >= len(self.track.segments)

        # FIX: early termination
        if self.speed < 1 and self.time > 5:
            done = True

        # FIX: proper reward
        reward = self.speed * 0.1 - grip_penalty * 0.01

        return self._get_state(), reward, done, {}


# =====================================================
# RL MODEL
# =====================================================

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


class PPOAgent:
    def __init__(self):
        self.model = ActorCritic()
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        self.gamma = 0.99

    def select_action(self, state, explore=True):
        if state is None:
            return np.array([0.0, 0.0], dtype=np.float32)
    
        state = np.array(state, dtype=np.float32)
    
        # FIX: ensure correct shape
        if state.ndim == 1:
            state = state.reshape(1, -1)
    
        state_tensor = torch.tensor(state, dtype=torch.float32)
    
        action, _ = self.model(state_tensor)
    
        action = action.detach().numpy()[0]  # FIX: remove batch dim
    
        if explore:
            noise = np.random.normal(0, 0.2, size=action.shape)
            action = action + noise
    
        return np.clip(action, -1, 1)
        action, _ = self.model(state)

        action = action.detach().numpy()

        # FIX: exploration
        if explore:
            noise = np.random.normal(0, 0.2, size=action.shape)
            action = action + noise

        return np.clip(action, -1, 1)

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
# CACHE MODEL
# =====================================================

@st.cache_resource
def load_agent():
    agent = PPOAgent()
    if os.path.exists("ppo_model.pth"):
        agent.model.load_state_dict(torch.load("ppo_model.pth"))
    return agent


# =====================================================
# UI
# =====================================================

st.set_page_config(layout="wide")
st.title("🏁 AI Racing Digital Twin (RL Fixed)")

agent = load_agent()

if "agent" not in st.session_state:
    st.session_state.agent = agent

agent = st.session_state.agent

# =====================================================
# TRAIN BUTTON
# =====================================================

if st.button("Train RL Agent"):
    env = AdvancedRacingEnv()

    for episode in range(5):
        state = env.reset()
        traj = []

        for _ in range(50):
            action = agent.select_action(state, explore=True)
            next_state, reward, done, _ = env.step(action)

            traj.append((state, action, reward))
            state = next_state

            if done:
                break

        agent.train(traj)

    torch.save(agent.model.state_dict(), "ppo_model.pth")
    st.success("Training complete")

# =====================================================
# INFERENCE (NO EXPLORATION)
# =====================================================

env = AdvancedRacingEnv()
state = env.reset()

trajectory = []
for _ in range(100):
    action = agent.select_action(state, explore=False)
    next_state, reward, done, _ = env.step(action)

    trajectory.append((state, action))
    state = next_state

    if done:
        break

# =====================================================
# VISUALIZATION
# =====================================================

speeds = [s[0][0] for s in trajectory]
throttle_vals = [(a[1][0] + 1)/2 for a in trajectory]
brake_vals = [max(0, a[1][1]) for a in trajectory]

fig, ax = plt.subplots()
ax.plot(speeds, label="Speed")
ax.plot(throttle_vals, label="Throttle")
ax.plot(brake_vals, label="Brake")
ax.legend()
st.pyplot(fig)

st.subheader("Final Control Output")
st.write({
    "Throttle": float(throttle_vals[-1]),
    "Brake": float(brake_vals[-1])
})

st.subheader("Debug Actions (first 10)")
st.write(throttle_vals[:10])
