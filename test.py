import pandas as pd
import numpy as np

df = pd.read_csv('Binance_BTCUSDT_1h_Base_MACD_PSAR_ATR_BB_ADX_RSI_ICHI_KC_Williams_Cnst_Interpolated.csv')

df['h_ratio'] = df['high']/df['close'].shift(1)
df['c_ratio'] = df['close']/df['close'].shift(1)
df['l_ratio'] = df['low']/df['close'].shift(1)
df['o_ratio'] = df['open']/df['close'].shift(1)
df['h_ratio_log'] = np.log(df['h_ratio'])
df['c_ratio_log'] = np.log(df['c_ratio'])
df['l_ratio_log'] = np.log(df['l_ratio'])
df['o_ratio_log'] = np.log(df['o_ratio'])
df['avg'] = (df['high']+df['close']+df['low']+df['open'])/4
df['avg_pctchange'] = df['avg'].pct_change()
df['avg_pctchange_log'] = np.log(df['avg_pctchange'])

df

df.to_csv('new_data.csv', index=False)