import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange
from case_closed_env import Game, Direction  # import your env file here


# --- Neural network for spatial representation + policy ---
class SRLPolicy(nn.Module):
    def __init__(self, input_channels=1, height=18, width=20, num_actions=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        flatten_size = 32 * height * width
        self.policy_head = nn.Sequential(
            nn.Linear(flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.policy_head(x)


# --- Simple epsilon-greedy agent ---
class TronRLAgent:
    def __init__(self, model, lr=1e-3, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)
        with torch.no_grad():
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state, done):
        q_values = self.model(state)
        next_q_values = self.model(next_state)
        q_target = q_values.clone()

        q_target[0, action] = reward if done else reward + self.gamma * torch.max(next_q_values).item()

        loss = self.criterion(q_values, q_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# --- Training loop ---
def train(num_episodes=500):
    game = Game()
    model = SRLPolicy()
    agent = TronRLAgent(model)

    os.makedirs("models", exist_ok=True)

    for episode in trange(num_episodes, desc="Training"):
        game.reset()
        done = False
        total_reward = 0

        # Initial state
        state = torch.tensor(np.array(game.board.grid), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        while not done:
            action_idx = agent.select_action(state)
            direction = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT][action_idx]

            result = game.step(direction, Direction.LEFT)  # Opponent always LEFT for now

            # Compute reward
            if result is None:
                reward = 0.1  # survived
                done = False
            elif result.name == "AGENT1_WIN":
                reward = 1.0
                done = True
            elif result.name == "AGENT2_WIN":
                reward = -1.0
                done = True
            else:
                reward = 0.0
                done = True

            total_reward += reward

            next_state = torch.tensor(np.array(game.board.grid), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            agent.update(state, action_idx, reward, next_state, done)
            state = next_state

        # Decay epsilon
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        print(f"[Episode {episode+1}] Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

        if (episode + 1) % 100 == 0:
            torch.save(model.state_dict(), f"models/tron_srl_policy_ep{episode+1}.pt")
            print(f"✅ Saved model checkpoint at episode {episode+1}")

    print("🏁 Training complete!")


if __name__ == "__main__":
    train(num_episodes=300)

