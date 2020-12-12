import gym
import numpy as np
from Naive_Deep_Q_Learning.Agent import Agent
from Utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    n_games = 10000
    scores = []
    eps_history = []

    agent = Agent(lr=0.0001, input_dim=env.observation_space.shape,
                  n_action=env.action_space.n)

    for i in range(n_games):
        score = 0
        done = False
        state = env.reset()

        while not done:
            env.render()
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            score += reward
            agent.learn(state, action, reward, next_state)
            state = next_state
        scores.append(score)
        eps_history.append(agent.eps)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print('episode ', i, 'score %.1f avg score %.1f epsilon %.2f' %
                  (score, avg_score, agent.eps))
    filename = 'cartpole_naive_dqn.png'
    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(x, scores, eps_history, filename)
