import numpy as np


class Agent:
    def __init__(self, lr, gamma, max_epsilon, min_epsilon, epsilon_decay, n_action, n_state):
        self.lr = lr
        self.gamma = gamma
        self.curr_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.n_action = n_action
        self.n_state = n_state

        self.q_values = {}
        self.init_q()

    def init_q(self):
        for state in range(self.n_state):
            for action in range(self.n_action):
                self.q_values[(state, action)] = 0

    def _get_best_action(self, state):
        mx = float('-inf')
        best_action = None
        for action in range(self.n_action):
            q_value = self.q_values[(state, action)]
            if mx < q_value:
                best_action = action
                mx = q_value
        return best_action

    def get_action(self, state):
        best_action = np.random.choice([i for i in range(self.n_action)])
        if np.random.random() > self.curr_epsilon:
            best_action = self._get_best_action(state)
        return best_action

    def decrease_epsilon(self):
        self.curr_epsilon = self.curr_epsilon * self.epsilon_decay
        if self.curr_epsilon < self.min_epsilon:
            self.curr_epsilon = self.min_epsilon

    def learn(self, state, action, reward, new_state):
        best_action = self._get_best_action(new_state)
        self.q_values[(state, action)] += self.lr * (reward + self.gamma * self.q_values[(new_state, best_action)]
                                                     - self.q_values[(state, action)])
        self.decrease_epsilon()
