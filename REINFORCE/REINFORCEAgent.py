import numpy as np
from Networks import REINFORCE_Policy_Network
import torch as T
import torch.nn.functional as F
import torch.optim as optim


class Agent:
    def __init__(self, lr, input_dim, gamma=0.99, n_action=4, ):
        self.lr = lr
        self.gamma = gamma
        self.n_action = n_action
        self.policy_network = REINFORCE_Policy_Network(lr=lr, input_dim=input_dim, n_action=n_action)
        self.rewards = []
        self.actions = []

    def get_action(self, state):
        state = T.tensor(state).to(self.policy_network.device)
        probs = F.softmax(self.policy_network.forward(state))
        action_probs = T.distributions.Categorical(probs)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.actions.append(log_probs)
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def learn(self):
        self.policy_network.optimiser.zero_grad()
        G = np.zeros(len(self.rewards))
        for i in range(len(self.rewards)):
            G_i = 0
            d = 0
            for k in range(i, len(self.rewards)):
                d += 1
                G_i += self.rewards[i] * (self.gamma ** d)
            G[i] = G_i
        G = T.tensor(G, dtype=T.float64).to(self.policy_network.device)
        loss = 0
        for g, prob in zip(G, self.actions):
            loss += -g * prob
        loss.backward()
        self.policy_network.optimiser.step()
        self.rewards = []
        self.actions = []
