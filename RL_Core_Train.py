# ================================================================
#
#   File name   : RL-Bitcoin-trading-bot_5.py
#   Author      : PyLessons
#   Created date: 2021-01-20
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/RL-Bitcoin-trading-bot
#   Description : Trading Crypto with Reinforcement Learning #5
#
#   Code revised by: Alireza Alikhani
#   Email       : alireza.alikhani@outlook.com 
#   Version     : 1.0.1
#
#
# ================================================================
from indicators import AddIndicators
from datetime import datetime
import matplotlib.pyplot as plt


from tensorflow.keras.optimizers import Adam, RMSprop

from collections import deque
import random
import numpy as np
import pandas as pd
import copy
import os
from icecream import ic
import re
from numpy.core.numeric import NaN
from custom_environment import custom_environment
from custom_agent import custom_agent

os.environ['CUDA_VISIBLE_DEVICES'] = '0'



## This part is used to start training
def train_agent(env, agent:custom_agent, visualize=False, train_episodes=50, training_batch_size=500):
    agent.create_writer(env.initial_balance, env.normalize_value,
                        train_episodes)  # create TensorBoard writer
    ## save n recent (maxlen=n) episodes net worth
    total_average = deque(maxlen=10)  
    best_average = 0  # used to track best average net worth
    for episode in range(train_episodes):
        state = env.reset(env_steps_size=training_batch_size)

        states, actions, rewards, predictions, dones, next_states = [], [], [], [], [], []
        for t in range(training_batch_size):
            env.render(visualize)
            action, prediction = agent.act(state)
            next_state, reward, done = env.step(action)
            states.append(np.expand_dims(state, axis=0))
            next_states.append(np.expand_dims(next_state, axis=0))
            action_onehot = np.zeros(len(agent.action_space))
            action_onehot[action] = 1
            actions.append(action_onehot)
            rewards.append(reward)
            dones.append(done)
            predictions.append(prediction)
            state = next_state

        a_loss, c_loss = agent.replay(
            states, actions, rewards, predictions, dones, next_states)

        total_average.append(env.net_worth)
        average = np.average(total_average)

        agent.writer.add_scalar('Data/average net_worth', average, episode)
        agent.writer.add_scalar('Data/episode_orders',
                                env.episode_orders, episode)

        agent.space = agent.next_space
        
        print("episode: {:<5} net worth {:<7.2f} average: {:<7.2f} orders: {}".format(
            episode, env.net_worth, average, env.episode_orders))
        if episode > len(total_average):
            if best_average < average:
                best_average = average
                print("Saving model")
                agent.save(score="{:.2f}".format(best_average), args=[
                           episode, average, env.episode_orders, a_loss, c_loss])
            agent.save()

    agent.end_training_log()


if __name__ == "__main__":

    ## Importing Data with 1H period
    df = pd.read_csv('./Binance_BTCUSDT_1h_Base_MACD_PSAR_ATR_BB_ADX_RSI_ICHI_KC_Williams_Cnst_Interpolated.csv')  # [::-1]
    lookback_window_size = 12
    test_window = 24 * 30    # 30 days

    ## Training Section:
    train_df = df[:-test_window-lookback_window_size]
    agent = custom_agent(lookback_window_size=lookback_window_size,
                        learning_rate=0.0001, epochs=5, optimizer=Adam, batch_size=24
                                                        , model="Dense", state_size=10+3)

    train_env = custom_environment(train_df, lookback_window_size=lookback_window_size)
    train_agent(train_env, agent, visualize=False,
              train_episodes=3000, training_batch_size=500)
    


