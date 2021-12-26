from custom_environment import custom_environment
from custom_agent import custom_agent
from icecream import ic
import pandas as pd
from tensorflow.keras.optimizers import Adam
from datetime import datetime


def test_agent(env, agent, visualize=True, test_episodes=10, folder="", name="Crypto_trader", comment=""):
    agent.load(folder, name)
    average_net_worth = 0
    average_orders = 0
    no_profit_episodes = 0
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            env.render(visualize and (episode == (test_episodes-1)))
            action, prediction = agent.act(state)
            state, reward, done = env.step(action, agent.space)
            agent.space = agent.next_space
            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                average_orders += env.episode_orders
                if env.net_worth < env.initial_balance:
                    # calculate episode count where we had negative profit through episode
                    no_profit_episodes += 1
                print("episode: {:<5}, net_worth: {:<7.2f}, average_net_worth: {:<7.2f}, orders: {} action_types: {}".format(
                    episode, env.net_worth, average_net_worth/(episode+1), env.episode_orders, {key: round(env.episode_actions[key]*100)
                    for key in env.episode_actions}))
                break

    print("average {} episodes agent net_worth: {}, orders: {}".format(
        test_episodes, average_net_worth/test_episodes, average_orders/test_episodes))
    print("No profit episodes: {}".format(no_profit_episodes))
    # save test results to test_results.txt file
    with open("test_results.txt", "a+") as results:
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        results.write(f'{current_date}, {name}, test episodes:{test_episodes}')
        results.write(
            f', net worth:{average_net_worth/(episode+1)}, orders per episode:{average_orders/test_episodes}')
        results.write(
            f', no profit episodes:{no_profit_episodes}, model: {agent.model}, comment: {comment}\n')


if __name__ == "__main__":

    ## Importing Data with 1H period
    df = pd.read_csv('./new_data.csv')
    #df = pd.read_csv('./Binance_BTCUSDT_1h_Base_MACD_PSAR_ATR_BB_ADX_RSI_ICHI_KC_Williams_Cnst_Interpolated.csv')  # [::-1]

    lookback_window_size = 12
    test_window = 24 * 25    # 30 days

    agent = custom_agent(lookback_window_size=lookback_window_size,
                        learning_rate=0.0001, epochs=5, optimizer=Adam, batch_size=24
                                                        , model="Dense", state_size=10+7)
    
    ## Test Section:
    test_df = df[-test_window:-test_window + 180]
    ic(test_df[['open','close']])   # Depicting the specified Time-period
    test_env = custom_environment(test_df, lookback_window_size=lookback_window_size,
                         Show_reward=True, Show_indicators=True)
    test_agent(test_env, agent, visualize=True, test_episodes=100,
                folder="2021_12_25_20_10_Crypto_trader", name="1326.93_Crypto_trader", comment="")