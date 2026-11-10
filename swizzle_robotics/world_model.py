import numpy as np
from typing import List, Tuple

class LatentWorldModel:
    """
    Lightweight forward predictive world model.
    Predicts next latent robotic state given current state and candidate action trajectory:
      s_{t+1} = f(s_t, a_t)
    """
    def __init__(self, state_dim: int = 16, action_dim: int = 6):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Transition matrix placeholder
        self.W_s = np.eye(state_dim, dtype=np.float32) * 0.95
        self.W_a = np.ones((state_dim, action_dim), dtype=np.float32) * 0.05

    def predict_next(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        next_s = np.dot(self.W_s, state) + np.dot(self.W_a, action)
        return np.tanh(next_s)

class MPCPlanner:
    """Model Predictive Control trajectory selector using sampling."""
    def __init__(self, world_model: LatentWorldModel, horizon: int = 5, num_samples: int = 32):
        self.model = world_model
        self.horizon = horizon
        self.num_samples = num_samples

    def plan(self, initial_state: np.ndarray, goal_state: np.ndarray) -> np.ndarray:
        best_action_seq = None
        min_cost = float('inf')

        for _ in range(self.num_samples):
            actions = np.random.randn(self.horizon, self.model.action_dim).astype(np.float32) * 0.1
            s = initial_state.copy()
            cost = 0.0
            for t in range(self.horizon):
                s = self.model.predict_next(s, actions[t])
                cost += float(np.linalg.norm(s - goal_state))

            if cost < min_cost:
                min_cost = cost
                best_action_seq = actions

        return best_action_seq[0] if best_action_seq is not None else np.zeros(self.model.action_dim)
