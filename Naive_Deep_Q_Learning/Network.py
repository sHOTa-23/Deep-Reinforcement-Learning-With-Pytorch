import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Naive_Q_network(nn.Module):
    def __init__(self, lr, n_action, input_dim):
        super(Naive_Q_network, self).__init__()
        self.fc1 = nn.Linear(*input_dim, 128)
        self.fc2 = nn.Linear(128, n_action)

        self.optimiser = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        fc1 = F.relu(self.fc1(state))
        fc2 = self.fc2(fc1)

        return fc2

