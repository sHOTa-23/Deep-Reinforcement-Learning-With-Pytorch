import numpy as np
import torch as T
import torch.functional as F
from Networks import ActorCriticNetwork


class Agent:
    def __init__(self, lr, gamma, n_action, input_dim, fc1_dim, fc2_dim):
        self.lr = lr
        self.gamma = gamma
        self.n_action = n_action
        self.network = ActorCriticNetwork(lr=lr, gamma=gamma, n_action=n_action, input_dim=input_dim, fc1_dim=fc1_dim,
                                          fc2_dim=fc2_dim)
        self.log_prob = None

    def get_action(self, state):

        state = T.tensor([state], dtype=T.float).to(self.network.device)
        probs, _ = self.network.forward(state)
        probs = T.softmax(probs, dim=1)
        action_probs = T.distributions.Categorical(probs)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        self.log_prob = log_prob

        return action.item()

    def learn(self, state, reward, next_state, done):

        self.network.optimiser.zero_grad()

        state = T.tensor([state], dtype=T.float).to(self.network.device)
        next_state = T.tensor([next_state], dtype=T.float).to(self.network.device)
        reward = T.tensor(reward, dtype=T.float).to(self.network.device)

        _, critic_value = self.network.forward(state)
        _, next_critic_value = self.network.forward(next_state)

        delta = reward - critic_value
        if not done:
            delta += self.gamma * next_critic_value

        actor_loss = -self.log_prob * delta
        critic_loss = delta ** 2

        all_loss = actor_loss + critic_loss
        all_loss.backward()
        self.network.optimiser.step()
