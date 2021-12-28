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

from Configs import *
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

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

pltfile = open("pltfile.csv", "w")
# This part is used to start training


# plt.ion()
# plt.show()
# plt.grid()
p1 = []
p2 = []


def plt_value(x, y):
    p1.append(x)
    p2.append(y)

    plt.plot(p1, c="black")
    plt.plot(p2, c="red")

    plt.pause(0.01)


def train_agent(env, agent: custom_agent, visualize=False, train_episodes=50, training_batch_size=500):
    agent.create_writer(env.initial_balance, env.normalize_value,
                        train_episodes)  # create TensorBoard writer
    # save n recent (maxlen=n) episodes net worth
    total_average = deque(maxlen=10)
    best_average = 0  # used to track best average net worth
    for episode in range(train_episodes):
        state = env.reset(env_steps_size=training_batch_size)

        states, spaces, actions, rewards, predictions, dones, next_states = [], [], [], [], [], [], []
        for t in range(training_batch_size):
            env.render(visualize)
            space = agent.get_onehot_mask()
            action, prediction = agent.act(state)

            next_state, reward, done = env.step(action, agent.space)
            states.append(np.expand_dims(state, axis=0))
            spaces.append(np.expand_dims(space, axis=0))
            next_states.append(np.expand_dims(next_state, axis=0))
            action_onehot = np.zeros(len(agent.action_space))
            action_onehot[action.value] = 1
            # ic(reward)
            actions.append(action_onehot)
            #rewards.append(reward)
            # ic(rewards)
            # ic(action)
            dones.append(done)
            predictions.append(prediction)
            state = next_state
            agent.space = agent.next_space
        reward_total = env.get_total_reward()
        rewards = [reward_total/training_batch_size for i in range(training_batch_size)]
        # ic(rewards)
        a_loss, c_loss = agent.replay(
            states, spaces, actions, rewards, predictions, dones, next_states)

        total_average.append(env.net_worth)
        average = np.average(total_average)

        agent.writer.add_scalar('Data/average net_worth', average, episode)
        agent.writer.add_scalar('Data/episode_orders',
                                env.episode_orders, episode)

        # print("episode: {:<5} net worth {:<7.2f} average: {:<7.2f} orders: {}".format(
        #     episode, env.net_worth, average, env.episode_orders),file=pltfile)
        # pltfile.flush()
        print("episode: {:<5} net worth {:<7.2f} average: {:<7.2f} orders: {} action_types: {}".format(
            episode, env.net_worth, average, env.episode_orders,

            {key: round(env.episode_actions[key]*100)
             for key in env.episode_actions}
        ))
        # plt_value(env.net_worth,average)

        if episode % 20 == 0:
            #if best_average < average:
            #best_average = average
            print("Saving model")
            agent.save(score="{:.2f}".format(episode), args=[
                        episode, average, env.episode_orders, a_loss, c_loss])
            #agent.save()

    agent.end_training_log()


if __name__ == "__main__":

    # Importing Data with 1H period
    # df = pd.read_csv('./Binance_BTCUSDT_1h_Base_MACD_PSAR_ATR_BB_ADX_RSI_ICHI_KC_Williams_Cnst_Interpolated.csv')  # [::-1]
    df = pd.read_csv('./new_data.csv')
    lookback_window_size = 12
    test_window = 24 * 30    # 30 days

    # Training Section:
    train_df = df[:-test_window-lookback_window_size]
    agent = custom_agent(lookback_window_size=lookback_window_size,
                         learning_rate=0.001, epochs=50, optimizer=SGD, batch_size=32, model="Dense", state_size=10+7)

    train_env = custom_environment(
        train_df, lookback_window_size=lookback_window_size)
    train_agent(train_env, agent, visualize=False,
                train_episodes=100000, training_batch_size=training_batch_size)
