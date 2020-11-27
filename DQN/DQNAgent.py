import numpy as np
import torch as T
from Networks import Deep_Q_Network
from ReplayBuffer import Replay_Buffer


class Agent():
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
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]

        self.replay_buffer = Replay_Buffer(mem_size, input_dim)
        self.Q_eval = Deep_Q_Network(lr, n_actions, self.env_name + '_' + self.algo + '_Q_eval', input_dim, chkpt_dir)
        self.Q_target = Deep_Q_Network(lr, n_actions, self.env_name + '_' + self.algo + '_Q_target', input_dim, chkpt_dir)


    def get_action(self,state):
        action = np.random.choice(self.action_space)
        if np.random.random() > self.epsilon:

