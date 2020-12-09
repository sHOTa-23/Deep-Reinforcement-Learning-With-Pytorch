import gym
import numpy as np
from ActorCritics.ActorCriticAgent import Agent
from Utils import plot_scores_curve

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, lr=5e-6, input_dim=[8], n_action=4,
                  fc1_dim=2048, fc2_dim=1536)
    n_games = 3000

    fname = "actor_critics"
    figure_file = fname + '.png'

    scores = []
    for i in range(n_games):
        done = False
        state = env.reset()
        score = 0
        while not done:
            #env.render()
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            score += reward
            agent.learn(state, reward, next_state, done)
            observation = next_state
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg_score)

    x = [i + 1 for i in range(n_games)]
    plot_scores_curve(x, scores, figure_file)
