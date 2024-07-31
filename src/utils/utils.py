import collections
#import cv2
import numpy as np
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
import pandas as pd



# https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html
# https://www.color-hex.com/color-palette

def linear_chart_score(x,y,dir):
    #fig=plt.figure(figsize=(20, 6))
    fig,ax = plt.subplots(figsize=(20, 6))
    # ax.plot(steps_array, scores, color='green', linestyle='dashed', marker='o',
    #      markerfacecolor='blue', markersize=1)
    # Average
    N = len(y)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(y[max(0, t-20):(t+1)])
        
    ax.plot(x, y, color='#a7adba', marker='o',
         markerfacecolor='#343d46', markersize=4, label='Score')
    
    ax.plot(x, running_avg,color='#343d46', label ='Avg Score',linewidth=3)
    
    ax.set(xlabel='samples', ylabel='P(t)-P(t-1)/P(t) %',
           title='Score')
    ax.grid()
    ax.legend()
    fig.savefig(dir+"score.png")
    plt.show()
    
def linear_chart_loss(x,y,dir,max_limit):

    fig, ax = plt.subplots(figsize=(20, 6))
    
    N = len(y)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(y[max(0, t-5):(t+1)])
    
    ax.plot(x, y, color='#eec540', marker='d',
         markerfacecolor='#927a2c', markersize=4, label='Loss')
    
    ax.plot(x, running_avg,color='#58460e', label ='Avg Loss',linewidth=2)
    
    ax.set(xlabel='samples', ylabel='Loss',
           title='Loss')
    ax.grid()
    ax.set_ylim([0, max_limit])
    ax.set_ylim()
    ax.legend()
    fig.savefig(dir+"Loss.png")
    plt.show()

def linear_chart_portfolio(x,y,dir): 

    fig, ax = plt.subplots(figsize=(20, 6))

    N = len(y)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(y[max(0, t-20):(t+1)])
    
    ax.plot(x, y, color='#b3cde0', marker='d',
         markerfacecolor='#005b96', markersize=4, label='Portfolio Return')
    
    ax.plot(x, running_avg,color='#011f4b', label ='Avg Portfolio Return',linewidth=2)
    
    ax.set(xlabel='samples', ylabel='Pt/P0 %',
           title='Porfolio Return %')
    ax.grid()
    ax.legend()
    fig.savefig(dir+"Portfolio.png")
    plt.show()

#____________________ Sharpe Ratio _________________________

def annualised_sharpe(returns, N=252):
    """
    Calculate the annualised Sharpe ratio of a returns stream
    based on a number of trading periods, N. N defaults to 252,
    which then assumes a stream of daily returns.

    The function assumes that the returns are the excess of
    those compared to a benchmark.
    """
    return np.sqrt(N) * returns.mean() / returns.std()
    
def equity_sharpe(df,column_name,free_risk_rate,N):
    """
    Calculates the annualised Sharpe ratio based on the daily
    returns of an equity ticker symbol listed in AlphaVantage.
    """
    # Use the percentage change method to easily calculate hour returns
    df['hour_ret'] = df[column_name].pct_change()

    # Assume an average annual risk-free rate over the period of 5%
    #eval(ticker)['excess_daily_ret'] = eval(ticker)['daily_ret'] - 0.05/252
    #Calculate per hour the benchmark rate (US federal reserve)
    # 5% annul divided 252 trading days by 7 hours of trading per day.=1764
    df['excess_hour_ret'] = df['hour_ret'] - free_risk_rate
    
    # Return the annualised Sharpe ratio based on the excess daily returns
    return annualised_sharpe(df['excess_hour_ret'],N)


