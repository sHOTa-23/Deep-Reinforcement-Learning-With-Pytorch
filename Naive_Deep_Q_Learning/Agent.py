import numpy as np
import torch as T
from Networks import Naive_Q_network


class Agent:
    def __init__(self, input_dim, n_action, lr=0.001, gamma=0.99, eps_max=1.0, eps_min=0.01,
                 eps_decay=1e-5, ):
        self.input_dim = input_dim
        self.n_action = n_action
        self.lr = lr
        self.gamma = gamma
        self.eps = eps_max
        self.min_eps = eps_min
        self.eps_decay = eps_decay
        self.action_space = [i for i in range(self.n_action)]

        self.Q = Naive_Q_network(lr=self.lr, n_action=self.n_action, input_dim=self.input_dim)

    def get_action(self, state):
        action = np.random.choice(self.action_space)
        if np.random.random() > self.eps:
            state = T.tensor(state, dtype=T.float).to(self.Q.device)
            action = T.argmax(self.Q.forward(state)).item()
        return action

    def decrease_epsilon(self):
        self.eps = self.eps - self.eps_decay
        if self.eps < self.min_eps:
            self.eps = self.min_eps

    def learn(self, state, action, reward, next_state):
        self.Q.optimiser.zero_grad()
        states = T.tensor(state, dtype=T.float).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        next_states = T.tensor(next_state, dtype=T.float).to(self.Q.device)

        curr_q = self.Q.forward(states)[actions]
        next_state_value = self.Q.forward(next_states).max()
        target_q = rewards + self.gamma * next_state_value
        loss = self.Q.loss(target_q,curr_q).to(self.Q.device)
        loss.backward()
        self.Q.optimiser.step()
        self.decrease_epsilon()
