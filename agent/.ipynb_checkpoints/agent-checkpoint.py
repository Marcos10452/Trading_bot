#!/usr/bin/env python
# coding: utf-8


from agent.mem_model import ReplayBuffer, build_dqn
import numpy as np
import pandas as pd

class Agent:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, batch_size, 
                 eps_min=0.01, eps_dec=5e-7,mem_size=1000000, env_name=None, 
                 dir='tmp/dqn/',dir2='tmp/dqn2/',replace=1000,n_neurons1=128,n_neurons2=128):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.env_name = env_name

        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.fname = dir + 'dqn_model_eval'
        self.fname_next = dir2 + 'dqn_model_next'

        self.memory = ReplayBuffer(mem_size, input_dims)
        # this will be the DNN for agent
        self.policy_net = build_dqn(lr,n_actions,input_dims, n_neurons1,n_neurons2) #256,256
        self.policy_net.summary()
        # this will be the DNN which is going to work as ORACLE
        self.target_net = build_dqn(lr,n_actions,input_dims, n_neurons1,n_neurons2)
        self.target_net.summary()

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        
    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.target_net.set_weights(self.policy_net.get_weights())
            print('Target_net has changed...')
    
    def choose_action(self, observation,training):
        #Next statement is true if I am in Training mode, otherwise in preddiction mode it will retun model value only.
        if training:
            if (np.random.random() > self.epsilon):
      
                state=np.array([observation]) #it is passing a batch
                # actions: predicted value for each output for a particular state
                #in example [[ 67.19449  77.82185 117.29195 144.89502]]
            
                actions = self.policy_net.predict(state)
                # I will keep with the argument of the highest actions for a particular state 
                # for previous example will be    3 
                action = np.argmax(actions)
            else:
                action = np.random.choice(self.action_space)
        
        else:
                state=np.array([observation]) #it is passing a batch
                # actions: predicted value for each output for a particular state
                #in example [[ 67.19449  77.82185 117.29195 144.89502]]
            
                actions = self.policy_net.predict(state)
                # I will keep with the argument of the highest actions for a particular state 
                # for previous example will be    3 
                action = np.argmax(actions)
        return action
   
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            #print("return")
            return

        #self.replace_target_network()

        states, actions, rewards, new_state, dones = self.memory.sample_buffer(self.batch_size)

        q_eval = self.policy_net.predict(states)
        q_next = self.target_net.predict(new_state)
        
        q_target=np.copy(q_eval) #using q_eval as a tamplate. Then, I am going to fill in q_target with belman's equation.
        batch_index=np.arange(self.batch_size, dtype=np.int32)

        # if episode terminate at step j+1 --> rj  (dones=[True])
        #otherwise step bellman's equation
        # I am not using loop to calculate this to improve speed.
        #q_target is the yi, which is was calculated with Bellman's equation
        q_target[batch_index,actions] = rewards + \
                (self.gamma*np.max(q_next, axis=1) * (1-np.array(dones)))
        
        
        # this is as same as (yi - Q(s,a))²
        loss=self.policy_net.train_on_batch(states,q_target)

        self.epsilon= self.epsilon  - self.eps_dec if self.epsilon> self.eps_min else self.eps_min

        
        self.learn_step_counter += 1
        
        #self.decrement_epsilon()
        return loss


    def save_models(self):
        self.policy_net.save(f"{self.fname}_{self.learn_step_counter}.h5")
        self.target_net.save(f"{self.fname_next}_{self.learn_step_counter}.h5")
        print('... models saved successfully ...')

    def load_models(self):
        self.policy_net = keras.models.load_model(self.fname+'q_eval')
        self.q_next = keras.models.load_model(self.fname+'q_next')
        print('... models loaded successfully ...')

