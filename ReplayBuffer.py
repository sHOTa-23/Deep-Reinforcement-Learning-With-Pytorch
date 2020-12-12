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


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

    class ReplayBuffer():
        def __init__(self, max_size, input_shape, n_actions):
            self.mem_size = max_size
            self.mem_cntr = 0
            self.state_memory = np.zeros((self.mem_size, *input_shape))
            self.new_state_memory = np.zeros((self.mem_size, *input_shape))
            self.action_memory = np.zeros((self.mem_size, n_actions))
            self.reward_memory = np.zeros(self.mem_size)
            self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

        def store_transition(self, state, action, reward, state_, done):
            index = self.mem_cntr % self.mem_size
            self.state_memory[index] = state
            self.action_memory[index] = action
            self.reward_memory[index] = reward
            self.new_state_memory[index] = state_
            self.terminal_memory[index] = done

            self.mem_cntr += 1

        def sample_buffer(self, batch_size):
            max_mem = min(self.mem_cntr, self.mem_size)

            batch = np.random.choice(max_mem, batch_size)

            states = self.state_memory[batch]
            actions = self.action_memory[batch]
            rewards = self.reward_memory[batch]
            states_ = self.new_state_memory[batch]
            dones = self.terminal_memory[batch]

            return states, actions, rewards, states_, dones


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones
