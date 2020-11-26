import gym
import numpy as np
import collections
import cv2


class RepeatActionsAndMaxFrame(gym.wrappers):
    def __init__(self, env, n_repeat):
        super(RepeatActionsAndMaxFrame, self).__init__(env)
        self.n_repeat = n_repeat
        self.env = env
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2, self.shape))

    def step(self, action):
        t_reward = 0
        done = False
        for i in range(self.n_repeat):
            state, reward, done, info = self.env.step(action)
            indx = i % 2
            self.frame_buffer[indx] = state
            t_reward += reward
            if done:
                break
        state = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return state, t_reward, done, info

    def reset(self):
        state = self.env.reset()
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.frame_buffer[0] = state

        return state


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super(PreprocessFrame, self).__init__(env)
        self.env = env
        self.shape = shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, observation):
        new_frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized_image = cv2.resize(new_frame, self.shape[1,], interpolation=cv2.INTER_AREA)
        image = np.array(resized_image, dtype=np.uint8).reshape(self.shape)
        image /= 255

        return image


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(repeat, axis=0),
            env.observation_space.high.repeat(repeat, axis=0),
            dtype=np.float32)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)


class MyEnv():
    def make_env(env_name, shape=(84, 84, 1), repeat=4, clip_rewards=False,
                 no_ops=0, fire_first=False):
        env = gym.make(env_name)
        env = RepeatActionsAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
        env = PreprocessFrame(shape, env)
        env = StackFrames(env, repeat)
        return env
