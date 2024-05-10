import numpy as np
import tensorflow as tf
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Assuming you have data loaded into variables: open_prices, close_prices, high_prices, low_prices, volumes

# Preprocess data
state_size = 5  # number of features (Open, Close, High, Low, Volume)
action_size = 1  # we are predicting only the next Close price
data_length = len(open_prices)
timesteps = 60  # number of timesteps to consider

# Create the agent
agent = DQNAgent(state_size, action_size)

# Training the agent
batch_size = 32
for e in range(EPISODES):
    for t in range(data_length - timesteps - 1):
        state = np.column_stack((open_prices[t:t+timesteps], close_prices[t:t+timesteps], 
                                 high_prices[t:t+timesteps], low_prices[t:t+timesteps], 
                                 volumes[t:t+timesteps]))
        state = np.reshape(state, [1, state_size])
        next_state = np.column_stack((open_prices[t+1:t+timesteps+1], close_prices[t+1:t+timesteps+1], 
                                      high_prices[t+1:t+timesteps+1], low_prices[t+1:t+timesteps+1], 
                                      volumes[t+1:t+timesteps+1]))
        next_state = np.reshape(next_state, [1, state_size])
        action = agent.act(state)
        reward = close_prices[t+timesteps+1] - close_prices[t+timesteps]  # reward is the change in Close price
        agent.remember(state, action, reward, next_state, False)
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)