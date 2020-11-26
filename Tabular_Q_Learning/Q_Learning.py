import gym
import numpy as np
import matplotlib.pyplot as plt
from Tabular_Q_Learning.Agent import Agent

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    agent = Agent(lr=0.001, gamma=0.9, max_epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.999995, n_action=4,
                  n_state=16)
    scores = []
    win_pct_list = []
    n_games = 500000

    for i in range(n_games):
        done = False
        state = env.reset()
        score = 0
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state)
            score += reward
            state = next_state
        scores.append(score)
        if i % 100 == 0:
            win_pct = np.mean(scores[-100:])
            win_pct_list.append(win_pct)
            if i % 1000 == 0:
                print('Episode: ',i, ' win pct %.2f' % win_pct, ' epsilon %.2f' %agent.curr_epsilon)
    plt.plot(win_pct_list)
    plt.show()