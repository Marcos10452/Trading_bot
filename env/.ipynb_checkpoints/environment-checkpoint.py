#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


#https://github.com/ClementPerroud/Gym-Trading-Env/tree/main/src/gym_trading_env


# In[ ]:


#--------------------------------------------------------------------     Environment   ----------------------------------------------------------------

#import gymnasium as gym
#from gymnasium import spaces
import pandas as pd
import numpy as np
import datetime
# import glob
# from pathlib import Path    

# from collections import Counter
#from history import History
from env.portfolio import *

import tempfile, os
# ------------- f*ck warning. It was hanging code.----------------------
#import warnings
#warnings.filterwarnings("error")

def dynamic_feature_last_position_taken(history):
    return history['position', -1]

def dynamic_feature_real_position(history):
    return history['real_position', -1]

#----------------------------------------------------------------------------------------------
class TradingEnv():
 
 #------------------------------------------------------------------------------------------
 # Classs   
    def __init__(self,
                df : pd.DataFrame,
                positions : list = [0, 1],
                #functions defined above. It could be changed 
                dynamic_feature_functions = [dynamic_feature_last_position_taken, dynamic_feature_real_position],
                # reward is based on basic_reward_function which is the log (actual close, previous close)
                #reward_function = basic_reward_function,
                windows = None,
                trading_fees = 0,
                borrow_interest_rate = 0,
                portfolio_initial_value = 1000,
                initial_position ='random',
                max_episode_duration = 'max',
                verbose = 1,
                name = "Stock",
                render_mode= "logs",
                obs_columns=['close']
                ):
        self.max_episode_duration = max_episode_duration
        self.name = name
        self.verbose = verbose

        self.positions = positions
        self.dynamic_feature_functions = dynamic_feature_functions
       # self.reward_function = reward_function
        self.windows = windows
        self.trading_fees = trading_fees
        self.borrow_interest_rate = borrow_interest_rate
        self.portfolio_initial_value = float(portfolio_initial_value)
        self.initial_position = initial_position
        self.obs_columns = obs_columns
        assert self.initial_position in self.positions or self.initial_position == 'random', "The 'initial_position' parameter must be 'random' or a position mentionned in the 'position' (default is [0, 1]) parameter."
        #assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.max_episode_duration = max_episode_duration
        #self.render_mode = render_mode
        
        #------------ check data and columns-------
        self._set_df(df)
        #-----------------------------------------
        
        #self.action_space = spaces.Discrete(len(positions))
        self.action_space = positions
        print(self.positions)

        self.observation_space =(len(self.obs_columns)+1,) # +1 was added for the postion column

        self.log_metrics = []
# created by me
        self._step=0
        

#------------------------------------------------------------------------------------------
#    Read df with market values for 1st time
    def _set_df(self, df):
        # df = df.copy()

        self.df = df
        assert "open" in self.df and "high" in self.df and "low" in self.df and "close" in self.df, "Your DataFrame needs to contain columns : open, high, low, close to render !"
       
  
        self.df =  self.df.reset_index()

        self._info_columns = list(self.df.columns) 

        
        
        #_obs_array has all "feature" columns in it 
        self._obs_array = np.array(self.df[self.obs_columns], dtype= np.float32)
        #_info_array is the array wich gives all information. In this case all columns minus feature_columns
        self._info_array = np.array(self.df[self._info_columns])
        # give "Close" price
        self._price_array = np.array(self.df["close"])


#------------------------------------------------------------------------------------------    
    def _get_ticker(self, delta = 0):
        return self.df.iloc[self._idx + delta]
#------------------------------------------------------------------------------------------
    def _get_price(self, delta = 0):
        return self._price_array[self._idx + delta]
 #------------------------------------------------------------------------------------------   
    #function to return the obs based on  self._obs_array calculated previously and then 
    # is added the dynamic_feature_functions which are the position[-1] and position[-2]
    def _get_obs(self):
        # for i, dynamic_feature_function in enumerate(self.dynamic_feature_functions):
        #     self._obs_array[self._idx, self._nb_static_features + i] = dynamic_feature_function(self.historical_info)

        _step_index = self._idx
      
        #return self._obs_array[_step_index]
        return np.concatenate((self._obs_array[_step_index], [self.df['position'][_step_index]])) #addint position t-1

#------------------------------------------------------------------------------------------    
    def reset(self, seed = 0, options=None):
        #super().reset(seed = seed)
        #set random with seed 
        np.random.seed(seed)
        self._step = 0
        #self.positions is a list and self_postion is an scalar.
        self._position = np.random.choice(self.positions) if self.initial_position == 'random' else self.initial_position
        self._limit_orders = {}
        
        self._idx = 0
    
        #instancing Class TargetPortfolio
        self._portfolio  = TargetPortfolio(
            position = self._position,
            value = self.portfolio_initial_value,
            price = self._get_price()
        )

        
        #select position's index
        position_index =self.positions.index(self._position)
        position = self._position

        #start initial values
        portfolio_valuation = self.portfolio_initial_value
        
        #return portfolio distribution
        portfolio_distribution = self._portfolio.get_portfolio_distribution(),
        reward = 0
        # Add columns to dataframe
        self.df['position']=""
        self.df['real_position']=""
        self.df['portfolio_valuation']=""
        self.df.loc[self._idx,"Asset_value"]=""
        self.df.loc[self._idx,"Money_fiat"]="" 
        self.df['reward']=""
        self.df.loc[self._idx,'position']=self._position
        self.df.loc[self._idx,"portfolio_valuation"]=self.portfolio_initial_value
        self.df.loc[self._idx,"Asset_value"]= self.portfolio_initial_value-self._portfolio.fiat_value()
        self.df.loc[self._idx,"Money_fiat"]= self._portfolio.fiat_value()
        self.df.loc[self._idx,'reward']=reward
        #print(dict(zip(self._info_columns, self._info_array[self._idx])))
        return self._get_obs(),"no info"
#--------------------------------------------
        
#------------------------------------------------------------------------------------------
    def basic_reward_function(self):
    
        if  self._idx:
            #SAR reward 2*self.df.loc[self._idx-1,"position"] -1) 
            diff=(self.df.loc[self._idx,"portfolio_valuation"] - self.df.loc[self._idx-1,"portfolio_valuation"])
            if self.df.loc[self._idx-1,"position"]==0 and self.df.loc[self._idx,"position"]:
                if diff>=1: #increase reward when stock was sold
                    reward=diff
                if diff<-0.5: #I am penalize worst when there is a lost selling a stock
                    reward=diff
                else:
                    reward=diff
            else:
                if diff>=1: #increase reward when stock was sold
                    reward=diff*(self.df.loc[self._idx-1,"position"] )
                if diff<-0.5: #I am penalize worst when there is a lost selling a stock
                    reward=diff*(self.df.loc[self._idx-1,"position"] )
                else:
                    reward=diff*(self.df.loc[self._idx-1,"position"] )
        else:
            reward=0
        return reward 

#
#------------------------------------------------------------------------------------------
    def _trade(self, position, price = None):
        self._portfolio.trade_to_position(
            position, 
            price = self._get_price() if price is None else price, 
            trading_fees = self.trading_fees
        )
        self._position = position
        return
#------------------------------------------------------------------------------------------
    def _take_action(self, position):
        
        if position != self._position:
            self._trade(position)

#------------------------------------------------------------------------------------------
    # def add_limit_order(self, position, limit, persistent = False):
    #     self._limit_orders[position] = {
    #         'limit' : limit,
    #         'persistent': persistent
    #     }
 #------------------------------------------------------------------------------------------   
    def step(self, position_index = None):
         #position_index is the "action" with the porfolio. It is "the index" not the value

        #ATTENTION:  I have changed order, first increment index and then calculate new treading
        self._idx += 1
        self._step += 1
        #select from position_index check action call _trade function which call   trade_to_position in library portfolio
        if position_index is not None: self._take_action(self.positions[position_index])

        
        # self._take_action_order_limit()
        price = self._get_price()
        # self._portfolio.update_interest(borrow_interest_rate= sel f.borrow_interest_rate)
        
        #sum (asset numbers * price - the interest)
        portfolio_value = self._portfolio.valorisation(price)
    
        self.df.loc[self._idx,"portfolio_valuation"]= portfolio_value
        #I think this for info (no need now)
        #portfolio_distribution = self._portfolio.get_portfolio_distribution()

        done, truncated = False, False
        # When using a max duration, each episode will start at a random starting point.
        if portfolio_value <= 0:
            done = True
        #check if m_idx has reached to dataframe length
        if self._idx >= len(self.df) - 1:
            truncated = True
        #check if max_episode_duration is int and if has reached to value
        if isinstance(self.max_episode_duration,int) and self._step >= self.max_episode_duration - 1:
            truncated = True
        
        position = self._position
        self.df.loc[self._idx,"position"]=position
        real_position = self._portfolio.real_position(price)
        self.df.loc[self._idx,"real_position"]= real_position
        self.df.loc[self._idx,"Money_fiat"]= self._portfolio.fiat_value()
        self.df.loc[self._idx,"Asset_value"]= portfolio_value-self._portfolio.fiat_value()
        

        reward = 0
        # )
        if not done:
            reward = self.basic_reward_function()
            self.df.loc[self._idx,"reward"] = reward

        # if done or truncated:
        #     self.calculate_metrics()
  
        return self._get_obs(),  reward, done, truncated, "no info"

#------------------------------------------------------------------------------------------
#     def add_metric(self, name, function):
#         self.log_metrics.append({
#             'name': name,
#             'function': function
#         })
# #------------------------------------------------------------------------------------------
#     def calculate_metrics(self): # it uses the data from historical_info
#         aux=self.df
#         self.results_metrics = {
#             "Market Return" : f"{100*(self.df.loc[self._idx,'close'] / self.df.loc[0,'close'] -1):5.2f}%",
#             "Portfolio Return" : f"{100*(self.df.loc[self._idx,'portfolio_valuation'] / self.df.loc[0,'portfolio_valuation'] -1):5.2f}%",

#              "Position Changes" : f"{np.sum(np.diff(self.df['position']) != 0)}",
#              "Episode Lenght" : f"{len(self.df['position'])}",

#             }
     
#         # for metric in self.log_metrics:
#         #     self.results_metrics[metric['name']] = metric['function'](self.historical_info)
#         text = ""
#         for key, value in self.results_metrics.items():
#             text += f"{key} : {value}   |   "
#         print("-------------------------------------------------------------------------------------------------------")
#         print(text)
#         print("-------------------------------------------------------------------------------------------------------")
# #------------------------------------------------------------------------------------------
#     def get_metrics(self):
#         return self.results_metrics
# #------------------------------------------------------------------------------------------

