import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from collections import deque

LR = 1e-3

GAMMA = 0.95

MEMORY_SIZE = 1000000
BATCH_SIZE = 16

EXPLORATION_MAX = 1
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 24, activation='relu')
    network = fully_connected(network, 24, activation='relu')

    network = fully_connected(network, 2, activation='linear')
    network = regression(network, optimizer='adam', loss='mean_square', learning_rate=LR, name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


class DQNSolver:
    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = neural_network_model(observation_space)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        state_batch = []
        q_values_batch = []

        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            state_batch.append(state[0])
            q_values_batch.append(q_values[0])

        self.model.fit({'input': state_batch}, {'targets': q_values_batch}, n_epoch=1, snapshot_epoch=False, run_id='openai_learning')
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def cartpole():
    env = gym.make('CartPole-v0')
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    # the dqn_solver is used to generate network, select action,
    # store the state and train the model
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    # we run 1000 rounds games to train model
    for i in range(1000):
        run += 1
        state = env.reset()
        state = np.reshape(state, (-1, len(state), 1))
        step = 0
        while True:
            step += 1
            action = dqn_solver.act(state)
            state_next, reward, terminal, _ = env.step(action)
            reward = -reward if terminal else reward
            state_next = state_next.reshape(-1, len(state_next), 1)
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                break
            dqn_solver.experience_replay()


cartpole()