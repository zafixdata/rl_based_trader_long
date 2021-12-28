from utils import TradingGraph, Write_to_file, Plot_OHCL
from collections import deque
import random
import numpy as np
import pandas as pd
from custom_agent import Acts, Spaces
from icecream import ic
from Configs import *

file_id = open('rewards.txt', 'w')
class custom_environment:
    # A custom Bitcoin trading environment
    def __init__(self, df, initial_balance=1000, lookback_window_size=50, Render_range=100, Show_reward=False, Show_indicators=False, normalize_value=40000):
        # Define action space and state size and other custom parameters
        self.df = df.dropna().reset_index()
        self.df_total_steps = len(self.df) - 1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.Render_range = Render_range  # render range in visualization
        self.Show_reward = Show_reward  # show order reward in rendered visualization
        # show main indicators in rendered visualization
        self.Show_indicators = Show_indicators

        # Orders history contains the balance, net_worth, crypto_bought, crypto_sold, crypto_held values for the last lookback_window_size steps
        self.orders_history = deque(maxlen=self.lookback_window_size)

        # Market history contains the OHCL values for the last lookback_window_size prices
        self.market_history = deque(maxlen=self.lookback_window_size)

        self.indicators_history = deque(maxlen=self.lookback_window_size)

        self.normalize_value = normalize_value

    # Reset the state of the environment to an initial state
    def reset(self, env_steps_size=0):
        # self.visualization = TradingGraph(Render_range=180, Show_reward=self.Show_reward,
        #                                  Show_indicators=self.Show_indicators)  # init visualization
        # limited orders memory for visualization
        #self.trades = deque(maxlen=self.Render_range)
        self.trades = deque(maxlen=180)

        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        self.episode_orders = 0  # track episode orders count
        self.episode_actions = {'remain_out_of_position': 0,
                                'enter_long': 0, 'exit_long': 0, 'remain_in_position': 0}
        self.episode_actions_number = 0
        self.prev_episode_orders = 0  # track previous episode orders count
        self.rewards = deque(maxlen=self.Render_range)
        self.env_steps_size = env_steps_size
        self.punish_value = 0
        if env_steps_size > 0:  # used for training dataset
            self.start_step = random.randint(
                self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else:  # used for testing dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps

        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append(
                [self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])

            self.market_history.append([#self.df.loc[current_step, 'open'],
                                        #self.df.loc[current_step, 'high'],
                                        #self.df.loc[current_step, 'low'],
                                        #self.df.loc[current_step, 'close'],
                                        #self.df.loc[current_step, 'volume'],
                                        self.df.loc[self.current_step, 'h_ratio'],
                                        self.df.loc[self.current_step, 'c_ratio'],
                                        self.df.loc[self.current_step, 'l_ratio'],
                                        self.df.loc[self.current_step, 'h_ratio_log'],
                                        self.df.loc[self.current_step, 'c_ratio_log'],
                                        self.df.loc[self.current_step, 'l_ratio_log'],
                                        self.df.loc[self.current_step, 'avg'],
                                        self.df.loc[self.current_step, 'avg_pctchange'],
                                        self.df.loc[self.current_step, 'avg_pctchange_log'],
                                        ])

            self.indicators_history.append(
                [
                    self.df.loc[current_step, 'macd_1h'] / 400,
                    #self.df.loc[current_step, 'macd_4h'] / 100,
                    #self.df.loc[current_step, 'macd_2h'] / 100,
                    #self.df.loc[current_step, 'psar_1'] / 1000,
                    #self.df.loc[current_step, 'psar_2'] / 40000,
                    #self.df.loc[current_step, 'psar_4'] / 40000,
                    #self.df.loc[current_step, 'psar_8'] / 40000,
                    #self.df.loc[current_step, 'ATR_2']/100,
                    #self.df.loc[current_step, 'ATR_4']/100
                    #self.df.loc[current_step, 'bb_bbh_1']/self.normalize_value,
                    #self.df.loc[current_step, 'bb_bbl_1']/self.normalize_value,
                    #self.df.loc[current_step, 'bb_bbm_1']/self.normalize_value,
                    #self.df.loc[current_step, 'bb_bbh_2']/self.normalize_value,
                    #self.df.loc[current_step, 'bb_bbl_2']/self.normalize_value,
                    #self.df.loc[current_step, 'bb_bbm_2']/self.normalize_value,
                    #self.df.loc[current_step, 'bb_bbh_4']/self.normalize_value,
                    #self.df.loc[current_step, 'bb_bbl_4']/self.normalize_value,
                    #self.df.loc[current_step, 'bb_bbm_4']/self.normalize_value,
                    #self.df.loc[current_step, 'ADX_1']/40,
                    #self.df.loc[current_step, 'RSI_1']/80,
                    #self.df.loc[current_step, 'kc_hb_1']/self.normalize_value,
                    #self.df.loc[current_step, 'kc_mb_1']/self.normalize_value,
                    #self.df.loc[current_step, 'kc_lb_1']/self.normalize_value,
                    #self.df.loc[current_step, 'kc_hb_2']/self.normalize_value,
                    #self.df.loc[current_step, 'kc_mb_2']/self.normalize_value,
                    #self.df.loc[current_step, 'kc_lb_2']/self.normalize_value,
                    #self.df.loc[current_step, 'kc_hb_4']/self.normalize_value,
                    #self.df.loc[current_step, 'kc_mb_4']/self.normalize_value,
                    #self.df.loc[current_step, 'kc_lb_4']/self.normalize_value,
                    #self.df.loc[current_step, 'kc_hb_8']/self.normalize_value,
                    #self.df.loc[current_step, 'kc_mb_8']/self.normalize_value,
                    #self.df.loc[current_step, 'kc_lb_8']/self.normalize_value,
                    #self.df.loc[current_step, 'Williams_R_1']/self.normalize_value,
                    self.df.loc[current_step, 'williams_2h'] / \
                    self.normalize_value,
                    self.df.loc[current_step, 'williams_4h'] / \
                    self.normalize_value,
                    #self.df.loc[current_step, 'williams_8h']/self.normalize_value,
                ])

        state = np.concatenate(
            (self.market_history, self.orders_history), axis=1) / self.normalize_value
        state = np.concatenate((state, self.indicators_history), axis=1)

        return state

    # Get the data points for the given current_step
    def _next_observation(self):
        self.market_history.append([#self.df.loc[self.current_step, 'open'],
                                    #self.df.loc[self.current_step, 'high'],
                                    #self.df.loc[self.current_step, 'low'],
                                    #self.df.loc[self.current_step, 'close'],
                                    #self.df.loc[self.current_step, 'volume'],
                                    self.df.loc[self.current_step, 'h_ratio'],
                                    self.df.loc[self.current_step, 'c_ratio'],
                                    self.df.loc[self.current_step, 'l_ratio'],
                                    self.df.loc[self.current_step, 'h_ratio_log'],
                                    self.df.loc[self.current_step, 'c_ratio_log'],
                                    self.df.loc[self.current_step, 'l_ratio_log'],
                                    self.df.loc[self.current_step, 'avg'],
                                    self.df.loc[self.current_step, 'avg_pctchange'],
                                    self.df.loc[self.current_step, 'avg_pctchange_log']
                                    ])
        

        self.indicators_history.append(
            [
                self.df.loc[self.current_step, 'macd_1h'] / 400,
                #self.df.loc[self.current_step, 'macd_4'] / 100,
                #self.df.loc[self.current_step, 'macd_2'] / 100,
                #self.df.loc[self.current_step, 'psar_1'] / 1000,
                #self.df.loc[self.current_step, 'psar_2'] / 40000,
                #self.df.loc[self.current_step, 'psar_4'] / 40000,
                #self.df.loc[self.current_step, 'psar_8'] / 40000,
                #self.df.loc[self.current_step, 'ATR_2']/100,
                #self.df.loc[self.current_step, 'ATR_4']/100
                #self.df.loc[self.current_step, 'bb_bbh_1']/self.normalize_value,
                #self.df.loc[self.current_step, 'bb_bbl_1']/self.normalize_value,
                #self.df.loc[self.current_step, 'bb_bbm_1']/self.normalize_value,
                #self.df.loc[self.current_step, 'bb_bbh_2']/self.normalize_value,
                #self.df.loc[self.current_step, 'bb_bbl_2']/self.normalize_value,
                #self.df.loc[self.current_step, 'bb_bbm_2']/self.normalize_value,
                #self.df.loc[self.current_step, 'bb_bbh_4']/self.normalize_value,
                #self.df.loc[self.current_step, 'bb_bbl_4']/self.normalize_value,
                #self.df.loc[self.current_step, 'bb_bbm_4']/self.normalize_value,
                #self.df.loc[self.current_step, 'ADX_1']/80,
                #self.df.loc[self.current_step, 'kc_hb_1']/self.normalize_value,
                #self.df.loc[self.current_step, 'kc_mb_1']/self.normalize_value,
                #self.df.loc[self.current_step, 'kc_lb_1']/self.normalize_value,
                #self.df.loc[self.current_step, 'kc_hb_2']/self.normalize_value,
                #self.df.loc[self.current_step, 'kc_mb_2']/self.normalize_value,
                #self.df.loc[self.current_step, 'kc_lb_2']/self.normalize_value,
                #self.df.loc[self.current_step, 'kc_hb_4']/self.normalize_value,
                #self.df.loc[self.current_step, 'kc_mb_4']/self.normalize_value,
                #self.df.loc[self.current_step, 'kc_lb_4']/self.normalize_value,
                #self.df.loc[self.current_step, 'kc_hb_8']/self.normalize_value,
                #self.df.loc[self.current_step, 'kc_mb_8']/self.normalize_value,
                #self.df.loc[self.current_step, 'kc_lb_8']/self.normalize_value,
                #self.df.loc[self.current_step, 'williams_R_1']/self.normalize_value,
                self.df.loc[self.current_step, 'williams_2h'] / \
                self.normalize_value,
                self.df.loc[self.current_step, 'williams_4h'] / \
                self.normalize_value,
                #self.df.loc[self.current_step, 'williams_8h']/self.normalize_value,
            ])

        obs = np.concatenate(
            (self.market_history, self.orders_history), axis=1) / self.normalize_value
        obs = np.concatenate((obs, self.indicators_history), axis=1)

        return obs

    # Execute one time step within the environment
    def step(self, action, space):
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1
        # ic(action)

        # Set the current price to Open
        current_price = self.df.loc[self.current_step, 'open']
        date = self.df.loc[self.current_step, 'date']  # for visualization
        High = self.df.loc[self.current_step, 'high']  # for visualization
        Low = self.df.loc[self.current_step, 'low']  # for visualization

        if action == Acts.remain_out_of_position:  # Hold
            self.episode_actions['remain_out_of_position'] += 1/training_batch_size
            self.episode_actions_number += 1
            pass

        elif action == Acts.enter_long and self.balance > self.initial_balance/100:
            # Buy with 100% of current balance
            self.crypto_bought = self.balance / current_price
            self.balance -= self.crypto_bought * current_price
            self.crypto_held += self.crypto_bought
            self.trades.append({'date': date, 'high': High, 'low': Low,
                               'total': self.crypto_bought, 'type': "buy", 'current_price': current_price})
            self.episode_actions['enter_long'] += 1/training_batch_size
            self.episode_actions_number += 1
            self.episode_orders += 1

        elif action == Acts.exit_long and self.crypto_held > 0:
            # Sell 100% of current crypto held
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * current_price
            self.crypto_held -= self.crypto_sold
            self.trades.append({'date': date, 'high': High, 'low': Low,
                               'total': self.crypto_sold, 'type': "sell", 'current_price': current_price})
            self.episode_actions['exit_long'] += 1/training_batch_size
            self.episode_actions_number += 1
            self.episode_orders += 1

        elif action == Acts.remain_in_position:
            self.episode_actions['remain_in_position'] += 1/training_batch_size
            self.episode_actions_number += 1
            pass

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price

        self.orders_history.append(
            [self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])

        # Receive calculated reward
        reward = self.get_reward(action, space)

        if self.net_worth <= self.initial_balance/2:
            done = True
        else:
            done = False

        obs = self._next_observation()

        return obs, reward, done
    
    def get_total_reward(self):
        return self.net_worth/self.initial_balance - 1

    # Calculating the reward
    def get_reward(self, action, space):
        
        #return np.random.rand()
        # ic(action)
        # ic(space)
        # self.punish_value += self.net_worth * 0.00001
        if self.episode_orders > 1 :#and self.episode_orders > self.prev_episode_orders:
            # <--Just covers Sell-Buy and Buy-Sell, not others -->
            self.prev_episode_orders = self.episode_orders
            if space == Spaces.in_position:
                if action == Acts.exit_long:
                    self.punish_value = 0
                    # reward = self.trades[-2]['total']*self.trades[-2]['current_price'] - \
                    #     self.trades[-1]['total'] * \
                    #     self.trades[-1]['current_price']
                    reward = self.net_worth*((self.trades[-1]['current_price']/ self.trades[-2]['current_price'])-1)
                    file_id.write(f'space={space}, action={action}, reward={reward:.4f}\n')
                    file_id.flush()
                    # print('mamad',file=file_id)
                    # file_id.close()
                    # print(f'reward for exit_long {reward}')
                    return reward
                elif action == Acts.remain_in_position:
                    C0 = self.df.loc[self.current_step, 'open']
                    C_1 = self.df.loc[self.current_step-1, 'open']
                    reward = C0/C_1-1
                    self.punish_value += self.net_worth * 0.00001
                    # print(
                        # f'punish for remain_in_position {-self.punish_value}')
                    file_id.write(f'space={space}, action={action}, reward={reward:.4f}\n')
                    file_id.flush()
                    # file_id.close()
                    return reward#-self.punish_value
            elif space == Spaces.out_of_position:
                if action == Acts.enter_long:
                    self.punish_value = 0
                    # reward = self.trades[-2]['total']*self.trades[-2]['current_price'] - \
                    #     self.trades[-1]['total'] * \
                    #     self.trades[-1]['current_price']
                    reward = 0
                    file_id.write(f'space={space}, action={action}, reward={reward:.4f}\n')
                    file_id.flush()
                    # file_id.close()
                    # print(f'reward for enter_long {reward}')
                    return reward
                elif action == Acts.remain_out_of_position:
                    C0 = self.df.loc[self.current_step, 'open']
                    C_1 = self.df.loc[self.current_step-1, 'open']
                    reward = C_1/C0-1
                    self.punish_value += self.net_worth * 0.00002
                    # print(
                    #     f'punish for remain_out_of_position {-self.punish_value}')
                    file_id.write(f'space={space}, action={action}, reward={reward-self.punish_value:.4f}, reward1={reward:.4f}, reward2={-self.punish_value:.4f}\n')
                    file_id.flush()
                    # file_id.close()
                    return reward-self.punish_value
            else:
                raise Exception

        # self.punish_value += self.net_worth * 0.00001

        # if self.episode_orders > 1 and self.episode_orders > self.prev_episode_orders:
        #     # <--Just covers Sell-Buy and Buy-Sell, not others -->
        #     self.prev_episode_orders = self.episode_orders
        #     if self.trades[-1]['type'] == "buy" and self.trades[-2]['type'] == "sell":
        #         reward = self.trades[-2]['total']*self.trades[-2]['current_price'] - \
        #             self.trades[-1]['total']*self.trades[-1]['current_price']

        #         reward -= self.punish_value
        #         self.punish_value = 0
        #         self.trades[-1]["Reward"] = reward
        #         return reward
        #     elif self.trades[-1]['type'] == "sell" and self.trades[-2]['type'] == "buy":
        #         reward = self.trades[-1]['total']*self.trades[-1]['current_price'] - \
        #             self.trades[-2]['total']*self.trades[-2]['current_price']
        #         reward -= self.punish_value
        #         self.punish_value = 0
        #         self.trades[-1]["Reward"] = reward
        #         return reward
        else:
            return -self.punish_value

    # render environment
    def render(self, visualize=False):
        '''if visualize:
            ## Render the environment to the screen (inside utils.py file)
            img = self.visualization.render(
                self.df.loc[self.current_step], self.net_worth, self.trades)
            return img'''
