# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 20:48:27 2023

@author: ritwi
"""

from datetime import timedelta
import datetime as dt
import math
import numpy as np
import pandas as pd
import pandas_ta as ta
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.graph_objs as go_objs
import plotly.subplots as sp
from plotly.subplots import make_subplots

df = pd.DataFrame()

ticker = st.sidebar.text_input('Enter Ticker', 'SPY')
# t = st.sidebar.selectbox('Select Number of Days', ('1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'))
# i = st.sidebar.selectbox('Select Time Granularity', ('1d', '1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'))
t = st.sidebar.selectbox('Select Number of Days', (180, 3000, 1000, 735, 450, 400, 350, 252, 150, 90, 60, 45, 30, 15))
i = st.sidebar.selectbox('Select Time Granularity', ('1d', '1wk', '1h', '15m'))
st.header(f'{ticker.upper()} Technical Indicators')

start = dt.datetime.today()-dt.timedelta(t)
end = dt.datetime.today()
df = yf.download(ticker, start, end, interval= i)

# Heikin Ashi
df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
df['HA_Open'] = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
df['HA_High'] = df[['High', 'HA_Open', 'HA_Close']].max(axis=1)
df['HA_Low'] = df[['Low', 'HA_Open', 'HA_Close']].min(axis=1)

# Calculate the RSI
n = 14
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(n).mean()
avg_loss = loss.rolling(n).mean().abs()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))
df['20RSI'] = df['RSI'].rolling(window=20).mean()

# Copy df for Double Supertrend
df1 = df.copy()
df2 = df.copy()
df3 = df.copy()
df4 = df.copy()

# Compute RSI divergence
def compute_rsi_divergence(data, window):
    high = df["High"].rolling(window).max()
    low = df["Low"].rolling(window).min()
    rsi = df["RSI"]
    divergence = (rsi - rsi.shift(window)) / (high - low)
    return divergence

rsi_divergence_window = 10
df["RSI_Divergence"] = compute_rsi_divergence(df, rsi_divergence_window)

# Compute buy and sell signals
buy_signal = (df["RSI_Divergence"] > 0) & (df["RSI_Divergence"].shift(1) < 0)
sell_signal = (df["RSI_Divergence"] < 0) & (df["RSI_Divergence"].shift(1) > 0)

# Calculate the MACD
df['12EMA'] = df['Close'].ewm(span=12).mean()
df['26EMA'] = df['Close'].ewm(span=26).mean()
df['MACD'] = df['12EMA'] - df['26EMA']
df['Signal Line'] = df['MACD'].ewm(span=9).mean()
df['Histogram'] = df['MACD'] - df['Signal Line']

# calculate the Average True Range (ATR)
df['tr1'] = abs(df['High'] - df['Low'])
df['tr2'] = abs(df['High'] - df['Close'].shift())
df['tr3'] = abs(df['Low'] - df['Close'].shift())
df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
df['atr'] = df['tr'].rolling(n).mean()

# calculate the Average Directional Index (ADX)
df['up_move'] = df['High'] - df['High'].shift()
df['down_move'] = df['Low'].shift() - df['Low']
df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
df['plus_di'] = 100 * (df['plus_dm'] / df['atr']).ewm(span=n, adjust=False).mean()
df['minus_di'] = 100 * (df['minus_dm'] / df['atr']).ewm(span=n, adjust=False).mean()
df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])).ewm(span=n, adjust=False).mean()
df['adx'] = df['dx'].ewm(span=n, adjust=False).mean()

# Calculate PSAR
def psar(df, iaf = 0.02, maxaf = 0.2):
    length = len(df)
    dates = list(df.index)
    high = list(df['High'])
    low = list(df['Low'])
    close = list(df['Close'])
    psar = close[0:len(close)]
    psarbull = [None] * length # Bullish signal - dot below candle
    psarbear = [None] * length # Bearish signal - dot above candle
    bull = True
    af = iaf # acceleration factor
    ep = low[0] # ep = Extreme Point
    hp = high[0] # High Point
    lp = low[0] # Low Point

    # https://www.investopedia.com/terms/p/parabolicindicator.asp - Parabolic Stop & Reverse Formula from Investopedia 
    for i in range(2,length):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
        reverse = False
        if bull:
            if low[i] < psar[i]:
                bull = False
                reverse = True
                psar[i] = hp
                lp = low[i]
                af = iaf
        else:
            if high[i] > psar[i]:
                bull = True
                reverse = True
                psar[i] = lp
                hp = high[i]
                af = iaf
        if not reverse:
            if bull:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + iaf, maxaf)
                if low[i - 1] < psar[i]:
                    psar[i] = low[i - 1]
                if low[i - 2] < psar[i]:
                    psar[i] = low[i - 2]
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + iaf, maxaf)
                if high[i - 1] > psar[i]:
                    psar[i] = high[i - 1]
                if high[i - 2] > psar[i]:
                    psar[i] = high[i - 2]
        if bull:
            psarbull[i] = psar[i]
        else:
            psarbear[i] = psar[i]
    return {"dates":dates, "high":high, "low":low, "close":close, "psar":psar, "psarbear":psarbear, "psarbull":psarbull}

if __name__ == "__main__":
    import sys
    import os
    
    startidx = 0
    endidx = len(df)
    
    result = psar(df)
    dates = result['dates'][startidx:endidx]
    close = result['close'][startidx:endidx]
    df["psarbear"] = result['psarbear'][startidx:endidx]
    df["psarbull"] = result['psarbull'][startidx:endidx]

# Supertrend

def Supertrend(df, atr_period, multiplier):
    
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # calculate ATR
    price_diffs = [high - low, 
                   high - close.shift(), 
                   close.shift() - low]
    true_range = pd.concat(price_diffs, axis=1)
    true_range = true_range.abs().max(axis=1)
    # default ATR calculation in supertrend indicator
    atr = true_range.ewm(alpha=1/atr_period,min_periods=atr_period).mean() 
    # df['atr'] = df['tr'].rolling(atr_period).mean()
    
    # HL2 is simply the average of high and low prices
    hl2 = (high + low) / 2
    # upperband and lowerband calculation
    # notice that final bands are set to be equal to the respective bands
    final_upperband = upperband = hl2 + (multiplier * atr)
    final_lowerband = lowerband = hl2 - (multiplier * atr)
    
    # initialize Supertrend column to True
    supertrend = [True] * len(df)
    
    for i in range(1, len(df.index)):
        curr, prev = i, i-1
        
        # if current close price crosses above upperband
        if close[curr] > final_upperband[prev]:
            supertrend[curr] = True
        # if current close price crosses below lowerband
        elif close[curr] < final_lowerband[prev]:
            supertrend[curr] = False
        # else, the trend continues
        else:
            supertrend[curr] = supertrend[prev]
            
            # adjustment to the final bands
            if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                final_lowerband[curr] = final_lowerband[prev]
            if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                final_upperband[curr] = final_upperband[prev]

        # to remove bands according to the trend direction
        if supertrend[curr] == True:
            final_upperband[curr] = np.nan
        else:
            final_lowerband[curr] = np.nan
    
    return pd.DataFrame({
        'Supertrend': supertrend,
        'Final Lowerband': final_lowerband,
        'Final Upperband': final_upperband
    }, index=df.index)
    
    
atr_period = 7
atr_multiplier = 3


supertrend = Supertrend(df, atr_period, atr_multiplier)
df = df.join(supertrend)

# Fast Double Supertrend
st_1 = Supertrend(df1, 14, 2)
df1 = df1.join(st_1)
st_2 = Supertrend(df2, 21, 1)
df2 = df2.join(st_2)

# Slow Double Supertrend
st_3 = Supertrend(df3, 21, 3)
df3 = df3.join(st_3)
st_4 = Supertrend(df4, 20, 7)
df4 = df4.join(st_4)

# Calculate the 9SMA and 20SMA
df['5SMA'] = df['Close'].rolling(window=5).mean()
df['9SMA'] = df['Close'].rolling(window=9).mean()
df['20SMA'] = df['Close'].rolling(window=20).mean()
df['50SMA'] = df['Close'].rolling(window=50).mean()
df['200SMA'] = df['Close'].rolling(window=200).mean()
rolling_std = df['Close'].rolling(window=20).std()
df['upper_band'] = df['20SMA'] + (rolling_std * 2)
df['lower_band'] = df['20SMA'] - (rolling_std * 2)

# Fractals
def find_fractals(data):
    fractals = []
    for i in range(5, len(df) - 5):
        if df['High'][i] > df['High'][i-1] and df['High'][i] > df['High'][i+1] and \
           df['High'][i] > df['High'][i-2] and df['High'][i] > df['High'][i+2]:
            fractals.append((df.index[i], df['High'][i], 'peak'))
        elif df['Low'][i] < df['Low'][i-1] and df['Low'][i] < df['Low'][i+1] and \
             df['Low'][i] < df['Low'][i-2] and df['Low'][i] < df['Low'][i+2]:
            fractals.append((df.index[i], df['Low'][i], 'trough'))
    return fractals

fractals = find_fractals(df)

# Function to add Fibonacci retracement lines
# Add Fibonacci retracement lines to the chart
low_price = min(df['Low'])
high_price = max(df['High'])

def add_fibonacci_retracement(fig, low, high, start_date, end_date):
    levels = [0, 0.382, 0.5, 0.618, 0.786, 1]
    diff = high - low
    for level in levels:
        price = high - level * diff
        fig.add_shape(
            go_objs.layout.Shape(
                type='line',
                x0=start_date,
                y0=price,
                x1=end_date,
                y1=price,
                line=dict(color='grey', dash='dash')
            )
        )

def add_fibonacci_extension(fig, low, high, start_date, end_date):
    levels = [1, 1.618, 2.0, 2.618]
    diff = high - low
    for level in levels:
        price = high + level * diff
        fig.add_shape(
            go_objs.layout.Shape(
                type='line',
                x0=start_date,
                y0=high,
                x1=end_date,
                y1=price,
                line=dict(color='grey', dash='dash')
            )
        )

# Keltner Channel
df.ta.kc(append=True)

# Squeeze Momentum Pro
df.ta.squeeze_pro(append=True)

# QQE Mod
df.ta.qqe(append=True)

# Awesome Oscillator
df.ta.ao(append=True)

# Awesome Oscillator
df.ta.ao(append=True)

# Donchian Channels
df.ta.donchian(append=True)

# Stochastic Oscillators
df.ta.stoch(append=True)

# Stochastic RSI
df.ta.stochrsi(append=True)

# Zero Lag Moving Average ribbons
df.ta.zlma(close=df['Adj Close'], length=20, append=True)
df.ta.zlma(close=df['Adj Close'], length=40, append=True)
df.ta.zlma(close=df['Adj Close'], length=60, append=True)
df.ta.zlma(close=df['Adj Close'], length=240, append=True)

# Hull Moving Average ribbons
df.ta.hma(close=df['Adj Close'], length=21, append=True)
df.ta.hma(close=df['Adj Close'], length=55, append=True)
df.ta.hma(close=df['Adj Close'], length=100, append=True)
df.ta.hma(close=df['Adj Close'], length=200, append=True)

# EMA Ribbons
df.ta.ema(length=8, append=True)
df.ta.ema(length=13, append=True)
df.ta.ema(length=21, append=True)
df.ta.ema(length=50, append=True)
df.ta.ema(length=200, append=True)

# Market Bias
df.ta.bias(close=df['Adj Close'], length=26, append=True)

# Z Score
df.ta.zscore(append=True)

# Gann High Low
df.ta.hilo(append=True)

# TD Sequential
df.ta.td_seq(append=True)
buy_signals = df[df['TD_SEQ_DN'] == 9]
sell_signals = df[df['TD_SEQ_UP'] == 9]

# Regression Channels
df.ta.tos_stdevall(append=True)

# Know Sure Thing (KST)
df.ta.kst(append=True)

# RVGI
df.ta.rvgi(append=True)

# Half Trend
df.ta.hl2(append=True)
df.ta.atr(append=True)
atr_multiplier = 2
df['upper_trend'] = df['HL2'] - atr_multiplier * df['ATRr_14']
df['lower_trend'] = df['HL2'] + atr_multiplier * df['ATRr_14']
df['half_trend'] = df['Adj Close'].where(df['Adj Close'] > df['upper_trend'], df['lower_trend'])
buy_ht = (df["half_trend"] > df['Close'].shift(1)) & (df["half_trend"] < df['Close'])
sell_ht = (df["half_trend"] < df['Close'].shift(1)) & (df["half_trend"] > df['Close'])

# Elher's Decycler

def decycler(data, hp_length):
    """Python implementation of Simple Decycler indicator created by John Ehlers
    :param data: list of price data
    :type data: list
    :param hp_length: High Pass filter length
    :type hp_length: int
    :return: Decycler applied price data
    :rtype: list
    """
    hpf = []

    for i, _ in enumerate(data):
        if i < 2:
            hpf.append(0)
        else:
            alpha_arg = 2 * 3.14159 / (hp_length * 1.414)
            alpha1 = (math.cos(alpha_arg) + math.sin(alpha_arg) - 1) / math.cos(alpha_arg)
            hpf.append(math.pow(1.0-alpha1/2.0, 2)*(data[i]-2*data[i-1]+data[i-2]) + 2*(1-alpha1)*hpf[i-1] - math.pow(1-alpha1, 2)*hpf[i-2])

    dec = []
    for i, _ in enumerate(data):
        dec.append(data[i] - hpf[i])

    return dec

df['decycler'] = decycler(df['Adj Close'], 20)
df['decycler_signal_buy'] = np.where(df["decycler"]<df['Adj Close'], 1, 0)
df['decycler_p'] = df['decycler_signal_buy'] * df['decycler']
df['decycler_signal_sell'] = np.where(df["decycler"]>df['Adj Close'], 1, 0)
df['decycler_n'] = df['decycler_signal_sell'] * df['decycler']
df['decycler_p'].replace(0.000000, np.nan, inplace=True)
df['decycler_n'].replace(0.000000, np.nan, inplace=True)

# Engulfing Candles
def find_engulfing_candles(prices):
    # Compute the candle body size for each candle
    prices['BodySize'] = abs(prices['Open'] - prices['Close'])
    
    # Check for bullish engulfing patterns
    bullish_engulfing = (prices['Open'] > prices['Close'].shift(1)) & \
                        (prices['Close'] < prices['Open'].shift(1)) & \
                        (prices['BodySize'] > prices['BodySize'].shift(1))
    
    # Check for bearish engulfing patterns
    bearish_engulfing = (prices['Open'] < prices['Close'].shift(1)) & \
                        (prices['Close'] > prices['Open'].shift(1)) & \
                        (prices['BodySize'] > prices['BodySize'].shift(1))
    
    # Create a new DataFrame to store engulfing candles
    engulfing_candles = pd.DataFrame(index=prices.index)
    engulfing_candles['Bullish'] = bullish_engulfing
    engulfing_candles['Bearish'] = bearish_engulfing
    
    return engulfing_candles
engulfing_candles = find_engulfing_candles(df)

# Doji
def find_doji_candles(prices):
    # Calculate the size of the candle body
    prices['BodySize'] = abs(prices['Open'] - prices['Close'])

    # Define a tolerance level for considering a candle as a doji
    tolerance = 0.01

    # Check for doji patterns
    doji = (prices['BodySize'] <= tolerance * prices['Open'])

    # Create a new DataFrame to store doji candles
    doji_candles = pd.DataFrame(index=prices.index)
    doji_candles['Doji'] = doji

    return doji_candles
doji_candles = find_doji_candles(df)


# Candlestick Patterns
dfc = pd.DataFrame()
dfc = df.ta.cdl_pattern("all")

def create_plot(df, indicators):
    fig = sp.make_subplots(rows=5, cols=1, shared_xaxes=True, row_heights=[0.4, 0.15, 0.15, 0.15, 0.15], vertical_spacing=0.02, subplot_titles=(f"{ticker.upper()} Daily Candlestick Chart", "Lower Indicator 1", "Lower Indicator 2", "Lower Indicator 3", "Lower Indicator 4"))

    for indicator in indicators:
        if indicator == 'Candlestick Chart':
            fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
        elif indicator == 'Heikin Ashi Candles':
            fig.add_trace(go.Candlestick(x=df.index, open=df["HA_Open"], high=df["HA_High"], low=df["HA_Low"], close=df["HA_Close"], name="Price"), row=1, col=1)
        elif indicator == 'RSI':
            fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index[buy_signal], y=df["RSI"][buy_signal], mode="markers", marker=dict(symbol="triangle-up", size=10, color="green"), name="Buy"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index[sell_signal], y=df["RSI"][sell_signal], mode="markers", marker=dict(symbol="triangle-down", size=10, color="red"), name="Sell"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['20RSI'], name='Mean RSI', line=dict(color='Orange', width=2)), row = 2, col = 1)
        elif indicator == 'MACD':
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=2)), row = 3, col = 1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Signal Line'], name='Signal', line=dict(color='red', width=2)), row = 3, col = 1)
            fig.add_trace(go.Bar(x=df.index, y=df['Histogram'], name='Histogram', marker=dict(color=df['Histogram'], colorscale='rdylgn')), row = 3, col = 1)
        elif indicator == 'ATR':
            fig.add_trace(go.Scatter(x=df.index, y=df['atr'], name='ATR', line=dict(color='purple', width=2)), row = 4, col = 1)
        elif indicator == 'ADX':
            fig.add_trace(go.Scatter(x=df.index, y=df['adx'], name='ADX', line=dict(color='blue', width=2)), row = 5, col = 1)
        elif indicator == 'PSAR':
            fig.add_trace(go.Scatter(x=dates, y=df["psarbull"], name='buy',mode = 'markers', marker = dict(color='green', size=2)))
            fig.add_trace(go.Scatter(x=dates, y=df["psarbear"], name='sell', mode = 'markers',marker = dict(color='red', size=2)))
        elif indicator == 'Supertrend':
            fig.add_trace(go.Scatter(x=df.index, y=df['Final Lowerband'], name='Supertrend Lower Band', line = dict(color='green', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['Final Upperband'], name='Supertrend Upper Band', line = dict(color='red', width=2)))
        elif indicator == 'Fast Double Supertrend':
            fig.add_trace(go.Scatter(x=df1.index, y=df1['Final Lowerband'], name='Supertrend Fast Lower Band', line = dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=df1.index, y=df1['Final Upperband'], name='Supertrend Fast Upper Band', line = dict(color='purple', width=2)))
            fig.add_trace(go.Scatter(x=df2.index, y=df2['Final Lowerband'], name='Supertrend Slow Lower Band',line = dict(color='green', width=2)))
            fig.add_trace(go.Scatter(x=df2.index, y=df2['Final Upperband'], name='Supertrend Slow Upper Band',line = dict(color='red', width=2)))
        elif indicator == 'Slow Double Supertrend':
            fig.add_trace(go.Scatter(x=df3.index, y=df3['Final Lowerband'], name='Supertrend Fast Lower Band', line = dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=df3.index, y=df3['Final Upperband'], name='Supertrend Fast Upper Band', line = dict(color='purple', width=2)))
            fig.add_trace(go.Scatter(x=df4.index, y=df4['Final Lowerband'], name='Supertrend Slow Lower Band',line = dict(color='green', width=2)))
            fig.add_trace(go.Scatter(x=df4.index, y=df4['Final Upperband'], name='Supertrend Slow Upper Band',line = dict(color='red', width=2)))
        elif indicator == 'SMA Ribbons':
            fig.add_trace(go.Scatter(x=df.index, y=df['5SMA'], name='5 SMA', line=dict(color='purple', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['9SMA'], name='9 SMA', line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['50SMA'], name='50 SMA', line=dict(color='green', width=2)))
        elif indicator == 'Bollinger Bands':
            fig.add_trace(go.Scatter(x=df.index, y=df['20SMA'], name='20 SMA', line=dict(color='black', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['upper_band'], name='Upper BB', line=dict(color='black', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['lower_band'], name='Lower BB', line=dict(color='black', width=2)))
        elif indicator == "Zero Lag MA Ribbons":
            fig.add_trace(go.Scatter(x = df.index, y=df['ZL_EMA_20'], line_color = 'purple', name = '20 ZLMA'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['ZL_EMA_40'], line_color = 'blue', name = '40 ZLMA'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['ZL_EMA_60'], line_color = 'green', name = '60 ZLMA'), row =1, col = 1)
        elif indicator == "Keltner Channels":
            fig.add_trace(go.Scatter(x = df.index, y=df['KCLe_20_2'], line_color = 'gray', name = 'Keltner Channel Lower Baad'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['KCBe_20_2'], line_color = 'gray', name = 'Keltner Channel Basis'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['KCUe_20_2'], line_color = 'gray', name = 'Keltner Channel Upper Band'), row =1, col = 1)
        elif indicator == "Squeeze Momentum Indicator Pro":
            colors = ['lightseagreen' if val > 0 else 'lightsalmon' for val in df['SQZPRO_20_2.0_20_2_1.5_1']]
            fig.add_trace(go.Bar(x = df.index, y=df['SQZPRO_20_2.0_20_2_1.5_1'], marker_color=colors, name = 'Squeeze Momentum Pro'), row = 4, col =1)
            fig.add_trace(go.Scatter(x = df[df['SQZPRO_OFF'] != 0].index, y=df[df['SQZPRO_OFF'] != 0]['SQZPRO_20_2.0_20_2_1.5_1'], mode = 'markers', marker = dict(color='green', size=5), name = 'Squeeze Off'), row = 4, col =1)
            fig.add_trace(go.Scatter(x = df[df['SQZPRO_ON_WIDE'] != 0].index, y=df[df['SQZPRO_ON_WIDE'] != 0]['SQZPRO_20_2.0_20_2_1.5_1'], mode = 'markers', marker = dict(color='black', size=5), name = 'Wide Squeeze'), row = 4, col =1)
            fig.add_trace(go.Scatter(x = df[df['SQZPRO_NO'] != 0].index, y=df[df['SQZPRO_NO'] != 0]['SQZPRO_20_2.0_20_2_1.5_1'], mode = 'markers', marker = dict(color='blue', size=5), name = 'No Squeeze'), row = 4, col =1)
            fig.add_trace(go.Scatter(x = df[df['SQZPRO_ON_NORMAL'] != 0].index, y=df[df['SQZPRO_ON_NORMAL'] != 0]['SQZPRO_20_2.0_20_2_1.5_1'], mode = 'markers', marker = dict(color='red', size=5), name = 'Normal Squeeze'), row = 4, col =1)
            fig.add_trace(go.Scatter(x = df[df['SQZPRO_ON_NARROW'] != 0].index, y=df[df['SQZPRO_ON_NARROW'] != 0]['SQZPRO_20_2.0_20_2_1.5_1'], mode = 'markers', marker = dict(color='purple', size=5), name = 'Narrow Squeeze'), row = 4, col =1)
        elif indicator == "QQE MOD":
            fig.add_trace(go.Scatter(x = df.index, y=df['QQE_14_5_4.236_RSIMA'], line_color = 'green', name = 'QQE RSI MA'), row =2, col = 1)
            fig.add_trace(go.Bar(x=df.index, y= df['QQEl_14_5_4.236'],  marker_color='blue', showlegend = False), row =2, col = 1)
            fig.add_trace(go.Bar(x=df.index, y= df['QQEs_14_5_4.236'],  marker_color='purple', showlegend = False), row =2, col = 1)
            fig.add_trace(go.Scatter(x=df.index, y= df['QQE_14_5_4.236'],  line_color='red', name = 'QQE RSI'), row =2, col = 1)
        elif indicator == "Stochastic RSI":
            fig.add_trace(go.Scatter(x = df.index, y=df['STOCHRSIk_14_14_3_3'], line_color = 'orange', name = 'Stochastic RSI %K'), row = 5, col=1)
            fig.add_trace(go.Scatter(x = df.index, y=df['STOCHRSId_14_14_3_3'], line_color = 'blue', name = 'Stochastic RSI %D'), row = 5, col=1)
        elif indicator == "Stochastic Oscillator":
            fig.add_trace(go.Scatter(x = df.index, y=df['STOCHk_14_3_3'], line_color = 'orange', name = 'Stochastic %K'), row = 5, col=1)
            fig.add_trace(go.Scatter(x = df.index, y=df['STOCHd_14_3_3'], line_color = 'blue', name = 'Stochastic %D'), row = 5, col=1)
        elif indicator == "Hull Moving Averages":
            fig.add_trace(go.Scatter(x = df.index, y=df['HMA_21'], line_color ='purple', name = '21 HMA'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['HMA_55'], line_color ='blue', name = '55 HMA'), row =1, col = 1)
        elif indicator == "EMA Ribbons":
            fig.add_trace(go.Scatter(x = df.index, y=df['EMA_8'], line_color = 'purple', name = '8 EMA'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['EMA_13'], line_color = 'blue', name = '13 EMA'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['EMA_21'], line_color = 'orange', name = '21 EMA'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['EMA_50'], line_color = 'green', name = '50 EMA'), row =1, col = 1)
        elif indicator == "200 EMA":
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=200, adjust=False).mean(), name='200EMA', line=dict(color='red', width=2)), row=1, col=1)
        elif indicator == '200 SMA':
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(window=200).mean(), name='200SMA', line=dict(color='red', width=2)), row=1, col=1)
        elif indicator == '100 HMA':
            fig.add_trace(go.Scatter(x = df.index, y=df['HMA_100'], line_color ='green', name = '100 HMA'), row =1, col = 1)
        elif indicator == '200 HMA':
            fig.add_trace(go.Scatter(x = df.index, y=df['HMA_200'], line_color ='red', name = '200 HMA'), row =1, col = 1)
        elif indicator == '240 ZLMA':
            fig.add_trace(go.Scatter(x = df.index, y=df['ZL_EMA_240'], line_color = 'red', name = '240 ZLMA'), row =1, col = 1)
        elif indicator == 'Market Bias':
            fig.add_trace(go.Scatter(x = df.index, y=df['BIAS_SMA_26'], line_color ='brown', name = 'Market Bias'), row =3, col = 1)
        elif indicator == "Awesome Oscillator":
            colors = ['green' if val > 0 else 'red' for val in df['AO_5_34']]
            fig.add_trace(go.Bar(x=df.index, y= df['AO_5_34'],  marker_color=colors, showlegend = False), row = 5, col=1)
        elif indicator == "Donchian Channels":
            fig.add_trace(go.Scatter(x = df.index, y=df['DCL_20_20'], line_color = 'skyblue', name = 'Donchian Channel Lower Baad'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['DCM_20_20'], line_color = 'skyblue', name = 'Donchian Channel Basis'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['DCU_20_20'], line_color = 'skyblue', name = 'Donchian Channel Upper Band'), row =1, col = 1)
        elif indicator == 'Z Score':
            fig.add_trace(go.Scatter(x = df.index, y=df['ZS_30'], line_color ='blue', name = 'Z Score'), row =5, col = 1)
        elif indicator == "Gann High Low":
            fig.add_trace(go.Scatter(x=df.index,y=df['HILOl_13_21'], mode='markers',marker=dict(color='green',symbol='star'),name='Gann High'))
            fig.add_trace(go.Scatter(x=df.index,y=df['HILOs_13_21'], mode='markers',marker=dict(color='red',symbol='star'),name='Gann Low'))
        elif indicator == "Fractals":
            for date, price, marker_type in fractals:
                fig.add_trace(go.Scatter(x=[date], y=[price], mode='markers', marker=dict(color='red' if marker_type == 'peak' else 'green'), name=marker_type))
        elif indicator == "Fibonacci Retracements":
            add_fibonacci_retracement( fig, low_price, high_price, start, end)
        elif indicator == "Fibonacci Extensions":
            add_fibonacci_extension( fig, low_price, high_price, start, end)
        elif indicator == "TD Sequential":
            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', name='Buy Signal', marker=dict(color='green', size=8)))
            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', name='Sell Signal', marker=dict(color='red', size=8)))
        elif indicator == "Linear Regression":
            fig.add_trace(go.Scatter(x = df.index, y=df['TOS_STDEVALL_LR'], line_color = 'black', name = 'Linear Regression Line'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['TOS_STDEVALL_L_1'], line_color = 'green', name = '1 Std Dev Down'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['TOS_STDEVALL_U_1'], line_color = 'red', name = '1 Std Dev Up'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['TOS_STDEVALL_L_2'], line_color = 'green', name = '2 Std Dev Down', visible='legendonly'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['TOS_STDEVALL_U_2'], line_color = 'red', name = '2 Std Dev Up', visible='legendonly'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['TOS_STDEVALL_L_3'], line_color = 'green', name = '3 Std Dev Down', visible='legendonly'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['TOS_STDEVALL_U_3'], line_color = 'red', name = '3 Std Dev Up', visible='legendonly'), row =1, col = 1)
        elif indicator == "Know Sure Thing":
            fig.add_trace(go.Scatter(x=df.index, y=df['KST_10_15_20_30_10_10_10_15'], name='KST', line=dict(color='green', width=2)), row = 3, col = 1)
            fig.add_trace(go.Scatter(x=df.index, y=df['KSTs_9'], name='KST Signal', line=dict(color='red', width=2)), row = 3, col = 1)
        elif indicator == "Relative Vigor Index":
            fig.add_trace(go.Scatter(x=df.index, y=df['RVGI_14_4'], name='RVGI', line=dict(color='green', width=2)), row = 3, col = 1)
            fig.add_trace(go.Scatter(x=df.index, y=df['RVGIs_14_4'], name='RVGI Signal', line=dict(color='red', width=2)), row = 3, col = 1)
        elif indicator == "Decycler":
            fig.add_trace(go.Scatter(x=df.index, y=df['decycler_p'], name='Decycler Bull', line = dict(color='green', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['decycler_n'], name='Decycler Bear',line = dict(color='red', width=2)))
        elif indicator == "Half Trend":
            fig.add_trace(go.Scatter(x=df.index,y=df['upper_trend'], mode='lines',line=dict(color='red'),name='HT Up Trend'))
            fig.add_trace(go.Scatter(x=df.index,y=df['lower_trend'], mode='lines',line=dict(color='green'),name='HT Down Trend'))
            fig.add_trace(go.Scatter(x=df.index,y=df['half_trend'], mode='lines',line=dict(color='blue'),name='Half Trend'))
            for date, price, marker_type in fractals:
                fig.add_trace(go.Scatter(x=[date], y=[price], mode='markers', marker=dict(color='red' if marker_type == 'peak' else 'green'), name=marker_type))
        elif indicator == "Engulfing Candles":
            bullish_engulfing_dates = engulfing_candles[engulfing_candles['Bullish']].index
            fig.add_trace(go.Scatter(x=bullish_engulfing_dates, y=df.loc[bullish_engulfing_dates, 'Low'], mode='markers', name='Bullish Engulfing', marker=dict(color='green', size=10)))
            # Add bearish engulfing candles
            bearish_engulfing_dates = engulfing_candles[engulfing_candles['Bearish']].index
            fig.add_trace(go.Scatter(x=bearish_engulfing_dates, y=df.loc[bearish_engulfing_dates, 'High'], mode='markers', name='Bearish Engulfing', marker=dict(color='red', size=10)))
        elif indicator == "Doji Candles":
            doji_dates = doji_candles[doji_candles['Doji']].index
            fig.add_trace(go.Scatter(x=doji_dates, y=df.loc[doji_dates, 'Low'], mode='markers', name='Doji', marker=dict(color='blue', size=8)))
        elif indicator == "Dragonfly Doji Candles":
            fig.add_trace(go.Scatter(x=df.index, y=dfc['CDL_DRAGONFLYDOJI'], mode="markers", marker=dict(size=10, color="red"), name="Dragonfly Doji"), row = 2, col = 1)
        elif indicator == "Gravestone Doji Candles":
            fig.add_trace(go.Scatter(x=df.index, y=dfc['CDL_GRAVESTONEDOJI'], mode="markers", marker=dict(size=10, color="red"), name="Gravestone Doji"), row = 2, col = 1)
        elif indicator == "Hammer Candles":
            fig.add_trace(go.Scatter(x=df.index, y=dfc['CDL_HAMMER'], mode="markers", marker=dict(size=10, color="red"), name="Hammer"), row = 2, col = 1)
        elif indicator == "Inverted Hammer Candles":
            fig.add_trace(go.Scatter(x=df.index, y=dfc['CDL_INVERTEDHAMMER'], mode="markers", marker=dict(size=10, color="red"), name="Inverted Hammer"), row = 2, col = 1)
        elif indicator == "Morning Star Candles":
            fig.add_trace(go.Scatter(x=df.index, y=dfc['CDL_MORNINGSTAR'], mode="markers", marker=dict(size=10, color="red"), name="Morning Star"), row = 2, col = 1)
        elif indicator == "Evening Star Candles":
            fig.add_trace(go.Scatter(x=df.index, y=dfc['CDL_EVENINGSTAR'], mode="markers", marker=dict(size=10, color="red"), name="Evening Star"), row = 2, col = 1)
        elif indicator == "Abandoned Baby Candles":
            fig.add_trace(go.Scatter(x=df.index, y=dfc['CDL_ABANDONEDBABY'], mode="markers", marker=dict(size=10, color="red"), name="Abandoned Baby"), row = 2, col = 1)
        elif indicator == "Hanging Man Candles":
            fig.add_trace(go.Scatter(x=df.index, y=dfc['CDL_HANGINGMAN'], mode="markers", marker=dict(size=10, color="red"), name="Hanging Man"), row = 2, col = 1)
        elif indicator == "3 White Soldiers":
            fig.add_trace(go.Scatter(x=df.index, y=dfc['CDL_3WHITESOLDIERS'], mode="markers", marker=dict(size=10, color="red"), name="3 White Soldiers"), row = 2, col = 1)
        elif indicator == "3 Black Crows":
            fig.add_trace(go.Scatter(x=df.index, y=dfc['CDL_3BLACKCROWS'], mode="markers", marker=dict(size=10, color="red"), name="3 Black Crows"), row = 2, col = 1)
        elif indicator == "3 Line Strike":
            fig.add_trace(go.Scatter(x=df.index, y=dfc['CDL_3LINESTRIKE'], mode="markers", marker=dict(size=10, color="red"), name="3 Line Strike"), row = 2, col = 1)
        elif indicator == "Shooting Star":
            fig.add_trace(go.Scatter(x=df.index, y=dfc['CDL_SHOOTINGSTAR'], mode="markers", marker=dict(size=10, color="red"), name="Shooting Star"), row = 2, col = 1)
        elif indicator == "Tristar":
            fig.add_trace(go.Scatter(x=df.index, y=dfc['CDL_TRISTAR'], mode="markers", marker=dict(size=10, color="red"), name="Tristar"), row = 2, col = 1)
    # Make it pretty
    layout = go.Layout(
    plot_bgcolor='#efefef',
    # Font Families
    font_family='Monospace',
    font_color='#000000',
    font_size=20,
    height=1000, width=1200)

    if i == '1d':
        fig.update_xaxes(
                rangeslider_visible=False,
                rangebreaks=[
                    # NOTE: Below values are bound (not single values), ie. hide x to y
                    dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                    # dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
                        # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
                    ]
                        )
    elif i == '1wk':
        fig.update_xaxes(
                rangeslider_visible=False,
                rangebreaks=[
                    # NOTE: Below values are bound (not single values), ie. hide x to y
                    dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                    # dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
                        # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
                    ]
                        )
    else:
        fig.update_xaxes(
                rangeslider_visible=False,
                rangebreaks=[
                    # NOTE: Below values are bound (not single values), ie. hide x to y
                    dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                    dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
                        # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
                    ]
                        )
    
    fig.update_layout(layout)
    st.plotly_chart(fig)


indicators = ['Candlestick Chart', 'Heikin Ashi Candles', 'RSI', 'MACD', 'ATR', 'ADX', 'PSAR', 'Supertrend', 'Fast Double Supertrend', 'Slow Double Supertrend', 'SMA Ribbons', 'Bollinger Bands', "Zero Lag MA Ribbons", "Keltner Channels", "Squeeze Momentum Indicator Pro", "QQE MOD", "Stochastic RSI", "Stochastic Oscillator", "Hull Moving Averages", "EMA Ribbons", "200 EMA", "200 SMA", "100 HMA", "200 HMA", "240 ZLMA", 'Market Bias', "Awesome Oscillator", "Donchian Channels", 'Z Score',"Gann High Low", "Fractals", "Fibonacci Retracements", "Fibonacci Extensions", "TD Sequential", "Linear Regression", "Know Sure Thing", "Relative Vigor Index" ,"Half Trend", "Decycler","Engulfing Candles", "Doji Candles", "Dragonfly Doji Candles", "Gravestone Doji Candles", "Hammer Candles", "Inverted Hammer Candles", "Morning Star Candles", "Evening Star Candles", "Abandoned Baby Candles", "Hanging Man Candles", "3 White Soldiers", "3 Black Crows", "3 Line Strike", "Shooting Star", "Tristar"]

default_options = ['Candlestick Chart', 'RSI', 'MACD', 'ATR', 'ADX', 'PSAR', 'Supertrend']


selected_indicators = st.multiselect('Select Indicators', indicators, default = default_options)


create_plot(df, selected_indicators)

st.write(dfc['CDL_INVERTEDHAMMER'])
