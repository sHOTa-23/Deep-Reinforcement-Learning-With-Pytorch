import numpy as np


class Replay_Buffer:
    def __init__(self, input_dim, max_size):
        self.log_size = 0
        self.alloc_size = max_size
        self.states = np.zeros((self.alloc_size, *input_dim), dtype=np.float32)
        self.new_states = np.zeros((self.alloc_size, *input_dim), dtype=np.float32)
        self.actions = np.zeros(self.alloc_size, dtype=np.int64)
        self.rewards = np.zeros(self.alloc_size, dtype=np.float32)
        self.dones = np.zeros(self.alloc_size, dtype=np.bool)

    def add(self, state, action, reward, new_state, done):
        ind = self.log_size % self.alloc_size
        self.log_size += 1
        self.states[ind] = state
        self.new_states[ind] = new_state
        self.rewards[ind] = reward
        self.actions[ind] = action
        self.dones[ind] = done

    def get_sample(self, batch_size):
        curr_ind = min(self.log_size, self.alloc_size)
        batch = np.random.choice(curr_ind, batch_size, replace=False)
        return self.states[batch], self.actions[batch], self.rewards[batch], self.new_states[batch], self.dones[batch]
