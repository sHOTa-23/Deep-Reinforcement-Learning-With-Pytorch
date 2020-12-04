import numpy as np
from DDQN.DDQNAgent import Agent
from Utils import plot_learning_curve
from OpenAIGymWrappers import make_env

if __name__ == '__main__':
    env = make_env('PongNoFrameskip-v4')
    best_score = -np.inf
    load_checkpoint = False
    n_games = 250

    agent = Agent(gamma=0.99, eps_max=1, lr=0.0001, input_dim=env.observation_space.shape,
                  n_actions=env.action_space.n, mem_size=9000, eps_min=0.1, batch_size=32,
                  replace=1000, eps_dec=1e-5, chkpt_dir='models/', algo='DDQNAgent', env_name='PongNoFrameskip-v4')

    if load_checkpoint:
        agent.load_models()
    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'
    n_steps = 0
    scores = []
    eps_history = []
    steps_array = []

    for i in range(n_games):
        done = False
        state = env.reset()
        score = 0
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            score += reward
            if not load_checkpoint:
                agent.store_transition(state, action, reward, next_state, done)
                agent.learn()
            state = next_state
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score: ', score,' average score %.1f' % avg_score,
              'best score %.2f' % best_score,'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    x = [i + 1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
