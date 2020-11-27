import numpy as np
import torch as T
from Networks import Deep_Q_Network
from ReplayBuffer import Replay_Buffer


class Agent:
    def __init__(self, lr, gamma, n_actions, input_dim, mem_size, batch_size, eps_max=1.0, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = eps_max
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dim
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.steps = 0
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]

        self.replay_buffer = Replay_Buffer(input_dim, mem_size)
        self.Q_eval = Deep_Q_Network(lr, n_actions, self.env_name + '_' + self.algo + '_Q_eval', input_dim, chkpt_dir)
        self.Q_target = Deep_Q_Network(lr, n_actions, self.env_name + '_' + self.algo + '_Q_target', input_dim,
                                       chkpt_dir)
    def get_action(self, state):
        action = np.random.choice(self.action_space)
        if np.random.random() > self.epsilon:
            state = T.tensor(state, dtype=T.float32).to(self.Q_eval.device)
            action = T.argmax(self.Q_eval.forward(state)).item()
        return action

    def store_transition(self, state, action, reward, new_state, done):
        self.replay_buffer.add(state, action, reward, new_state, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.replay_buffer.get_sample(self.batch_size)

        states = T.tensor(state).to(self.Q_eval.device)
        rewards = T.tensor(reward).to(self.Q_eval.device)
        dones = T.tensor(done).to(self.Q_eval.device)
        actions = T.tensor(action).to(self.Q_eval.device)
        new_states = T.tensor(new_state).to(self.Q_eval.device)

        return states, actions, rewards, new_states, dones

    def replace_target_network(self):
        if self.steps % self.replace_target_cnt == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def decrease_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec
        if self.epsilon < self.eps_min:
            self.epsilon = self.eps_min

    def load_models(self):
        self.Q_eval.load_checkpoint()
        self.Q_target.load_checkpoint()

    def save_models(self):
        self.Q_eval.save_checkpoint()
        self.Q_target.save_checkpoint()

    def learn(self):
        if self.batch_size > self.replay_buffer.log_size:
            return

        self.Q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, next_states, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.Q_eval.forward(states)[indices, actions]
        q_next = self.Q_target.forward(next_states).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next

        loss = self.Q_eval.loss(q_target, q_pred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.steps += 1
        self.decrease_epsilon()
