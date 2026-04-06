# path: advanced_rl_env.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================
# 1. TRACK MODEL
# =====================================================

class Track:
    """Simple track: straights + corners"""
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


# =====================================================
# 2. PACEJKA-LITE MODEL
# =====================================================

def pacejka_lite(slip_angle, mu=1.2):
    """Simplified lateral force curve"""
    B, C, D = 10, 1.9, mu
    return D * np.sin(C * np.arctan(B * slip_angle))


def max_corner_speed(radius, mu=1.2):
    g = 9.81
    return np.sqrt(mu * g * radius)


# =====================================================
# 3. ADVANCED ENVIRONMENT
# =====================================================

class AdvancedRacingEnv:
    def __init__(self):
        self.track = Track()
        self.reset()

    def reset(self):
        self.speed = 0.0
        self.position = 0.0
        self.segment_idx = 0
        self.time = 0.0
        return self._get_state()

    def _get_state(self):
        seg = self.track.get_segment(self.segment_idx)

        curvature = 0.0
        if seg["type"] == "corner":
            curvature = 1.0 / seg["radius"]

        return np.array([
            self.speed,
            curvature,
            self.time,
            self.segment_idx,
            self.speed * curvature
        ], dtype=np.float32)

    def step(self, action):
        throttle, brake = action

        seg = self.track.get_segment(self.segment_idx)

        # longitudinal dynamics
        accel = throttle * 6 - brake * 10 - 0.02 * self.speed**2
        self.speed = max(0, self.speed + accel * 0.1)

        # corner grip constraint
        if seg["type"] == "corner":
            vmax = max_corner_speed(seg["radius"])
            if self.speed > vmax:
                grip_penalty = (self.speed - vmax)**2
                self.speed *= 0.9
            else:
                grip_penalty = 0.0
        else:
            grip_penalty = 0.0

        # move along track
        self.position += self.speed * 0.1
        self.time += 0.1

        # segment switching
        if seg["type"] == "straight" and self.position > seg["length"]:
            self.position = 0
            self.segment_idx += 1
        elif seg["type"] == "corner" and self.position > seg["radius"] * np.pi / 2:
            self.position = 0
            self.segment_idx += 1

        done = self.segment_idx >= len(self.track.segments)

        # reward = negative lap time + penalties
        reward = -0.1 - grip_penalty * 0.01

        return self._get_state(), reward, done, {}


# =====================================================
# 4. PPO AGENT
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
# 5. TRAINING LOOP (TEST)
# =====================================================

if __name__ == "__main__":
    env = AdvancedRacingEnv()
    agent = PPOAgent()

    for episode in range(50):
        state = env.reset()
        traj = []

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            traj.append((state, action, reward))
            state = next_state

            if done:
                break

        agent.train(traj)
        print(f"Episode {episode} complete")
