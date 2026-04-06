# path: rl_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# -------------------------------
# 1. ENVIRONMENT
# -------------------------------

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

        # reward shaping
        speed_reward = row['vehicle_speed']
        stability_penalty = abs(row['slip_Wheel_Speed_FL']) * 50
        thermal_penalty = row['tire_energy'] * 0.001

        reward = speed_reward - stability_penalty - thermal_penalty

        self.step_idx += 1
        done = self.step_idx >= self.max_step

        return self._get_state(), reward, done, {}

# -------------------------------
# 2. PPO NETWORK
# -------------------------------

class ActorCritic(nn.Module):
    def __init__(self, state_dim=5, action_dim=2):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        action = self.actor(x)
        value = self.critic(x)
        return action, value

# -------------------------------
# 3. PPO TRAINER
# -------------------------------

class PPOAgent:
    def __init__(self):
        self.model = ActorCritic()
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        self.gamma = 0.99
        self.eps_clip = 0.2

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action, _ = self.model(state)
        return action.detach().numpy()

    def compute_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32)

    def train(self, trajectories):
        states = torch.tensor(np.array([t[0] for t in trajectories]), dtype=torch.float32)
        actions = torch.tensor(np.array([t[1] for t in trajectories]), dtype=torch.float32)
        rewards = [t[2] for t in trajectories]

        returns = self.compute_returns(rewards)

        pred_actions, values = self.model(states)
        values = values.squeeze()

        advantages = returns - values.detach()

        loss_actor = ((pred_actions - actions)**2 * advantages.unsqueeze(1)).mean()
        loss_critic = (returns - values).pow(2).mean()

        loss = loss_actor + 0.5 * loss_critic

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()