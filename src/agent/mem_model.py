import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.optimizers.legacy import Adam,SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
#from keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras import initializers

from settings import *


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        
        #print(self.state_memory,index,state)
        #print(self.state_memory[index],state)
        
        self.state_memory[index] = state 
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done # I have modified to upload the true done value original 1-done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

        
def build_dqn(lr,n_actions, input_dims,fc1_dims, fc2_dims,new_model,path_model):
    """
    """
    
    #Arguments
    #mean: a python scalar or a scalar tensor. Mean of the random values to generate.
    #stddev: a python scalar or a scalar tensor. Standard deviation of the random values to generate.
    #seed: A Python integer. Used to make the behavior of the initializer deterministic. 
    #Note that a seeded initializer will produce the same random values across multiple calls.
    if new_model:
        initializer = tf.keras.initializers.RandomNormal(mean=0.2,stddev=0.12,seed=7)
        print("New Model was selected. Restart weights")
            
        #https://medium.com/@natsunoyuki/speeding-up-model-training-with-google-colab-b1ad7c48573e
        device_name = tf.test.gpu_device_name()
        if len(device_name) > 0:
            print("Found GPU at: {}".format(device_name))
        else:
            device_name = "/device:CPU:0"
            print("No GPU, using {}.".format(device_name))
        with tf.device(device_name):
            model = Sequential()
            model.add(Input(shape=input_dims)) # Input tensor
            model.add(Dense(fc1_dims,kernel_initializer=initializer ,bias_initializer=initializers.Zeros(),activation='relu'))
            model.add(Dense(fc2_dims, kernel_initializer=initializer ,bias_initializer=initializers.Zeros(),activation='relu')) #original * 2 sigmoid
            model.add(Dense(n_actions, activation=None)) #original is activation=None Why?
            model.compile(optimizer=eval(OPTIMIZER)(lr), loss=LOSS)
            #model.summary()
    else:
        model = tf.keras.models.load_model(path_model)
        print(" Model selected was restored... Have a nice run")
    return model
