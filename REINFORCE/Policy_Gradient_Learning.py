import gym
import matplotlib.pyplot as plt
import numpy as np
from Utils import plot_scores_curve
from REINFORCE.REINFORCEAgent import Agent

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_games = 3000
    agent = Agent(gamma=0.99, lr=0.0005, input_dim=[8],
                  n_action=4)

    fname = 'REINFORCE_' + 'lunar_lunar_lr' + str(agent.lr) + '_' \
            + str(n_games) + 'games'
    figure_file = 'tmp/' + fname + '.png'

    scores = []
    for i in range(n_games):

        done = False
        state = env.reset()
        score = 0
        while not done:
            env.render()
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            score += reward
            agent.store_reward(reward)
            state = next_state
        agent.learn()
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score)

    x = [i + 1 for i in range(len(scores))]
    plot_scores_curve(scores, x, figure_file)
