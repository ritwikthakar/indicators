# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 14:21:08 2023

@author: ritwi
"""

import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import plotly.subplots as sp
import plotly.graph_objs as go
import streamlit as st

ticker = st.sidebar.text_input('Enter Ticker', 'SPY')
# t = st.sidebar.selectbox('Select Number of Days', ('1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'))
# i = st.sidebar.selectbox('Select Time Granularity', ('1d', '1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'))
t = st.sidebar.selectbox('Select Number of Days', (180, 3000, 1000, 735, 400, 350, 252, 150, 90, 60, 45, 30, 15))
i = st.sidebar.selectbox('Select Time Granularity', ('1d', '1wk'))
st.header(f'{ticker.upper()} Technical Analysis')

start = dt.datetime.today()-dt.timedelta(t)
end = dt.datetime.today()
df = yf.download(ticker, start, end, interval= i)





def create_plot(df, indicators):
    fig = sp.make_subplots(rows=5, cols=1, shared_xaxes=True, row_heights=[0.4, 0.15, 0.15, 0.15, 0.15], vertical_spacing=0.02, subplot_titles=(f"{ticker.upper()} Daily Candlestick Chart", "Lower Indicator 1", "Lower Indicator 2", "Lower Indicator 3", "Lower Indicator 4"))
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)

    for indicator in indicators:
        if indicator == '20SMA':
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(window=20).mean(), name='20SMA', line=dict(color='orange', width=2)), row=1, col=1)
        elif indicator == '8EMA':
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=8, adjust=False).mean(), name='8EMA', line=dict(color='purple', width=2)), row=1, col=1)
        elif indicator == '13EMA':
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=13, adjust=False).mean(), name='13EMA', line=dict(color='blue', width=2)), row=1, col=1)
        elif indicator == '21EMA':
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=21, adjust=False).mean(), name='21EMA', line=dict(color='pink', width=2)), row=1, col=1)
        elif indicator == '50EMA':
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=50, adjust=False).mean(), name='50EMA', line=dict(color='green', width=2)), row=1, col=1)
        elif indicator == '200EMA':
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=200, adjust=False).mean(), name='200EMA', line=dict(color='red', width=2)), row=1, col=1)
        elif indicator == '9SMA':
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(window=9).mean(), name='9SMA', line=dict(color='blue', width=2)), row=1, col=1)
        elif indicator == '5SMA':
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(window=5).mean(), name='5SMA', line=dict(color='purple', width=2)), row=1, col=1)
        elif indicator == '50SMA':
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(window=50).mean(), name='50SMA', line=dict(color='green', width=2)), row=1, col=1)
        elif indicator == '200SMA':
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(window=200).mean(), name='200SMA', line=dict(color='red', width=2)), row=1, col=1)
        elif indicator == 'Percent %B':
            def bollinger_band_percent_b(close_prices, window_size=20, num_std_dev=2):
                rolling_mean = close_prices.rolling(window=window_size).mean()
                rolling_std = close_prices.rolling(window=window_size).std()
                upper_band = rolling_mean + num_std_dev * rolling_std
                lower_band = rolling_mean - num_std_dev * rolling_std
                percent_b = (close_prices - lower_band) / (upper_band - lower_band)
                return percent_b
            df["Percent_B"] = bollinger_band_percent_b(df['Close'])
            fig.add_trace(go.Scatter(x=df.index, y=df['Percent_B'], name='% B', line=dict(color='brown', width=2)), row = 2, col = 1)
        elif indicator == "Bollinger Band Width":
            def calculate_bollinger_bands(df, window_size, num_std):
                rolling_mean = df['Close'].rolling(window=window_size).mean()
                rolling_std = df['Close'].rolling(window=window_size).std()
                upper_band = rolling_mean + (rolling_std * num_std)
                lower_band = rolling_mean - (rolling_std * num_std)
                bollinger_width = (upper_band - lower_band) / rolling_mean
                return bollinger_width
            window_size = 20
            num_std = 2
            df["Bollinger_Width"] = calculate_bollinger_bands(df, window_size, num_std)
            df["Bollinger_Width_Avg"] = df['Bollinger_Width'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Width'], name='Bollinger Band Width', line=dict(color='purple', width=2)), row = 3, col = 1)
        elif indicator == 'Bollinger Band Trend':
            # calculate the Bollinger Band Trend
            def bollinger_band_trend(close_prices, window_size=20, num_std_dev=2):
                rolling_mean = close_prices.rolling(window=window_size).mean()
                rolling_std = close_prices.rolling(window=window_size).std()
                upper_band = rolling_mean + num_std_dev * rolling_std
                lower_band = rolling_mean - num_std_dev * rolling_std
                percent_b = (close_prices - lower_band) / (upper_band - lower_band)
                trend = np.where(percent_b > 0.5, 1, np.where(percent_b < -0.5, -1, 0))
                return pd.Series(trend, index=close_prices.index)
            df["BB_Trend"] = bollinger_band_trend(df['Close'])
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Trend'], name='Bollinger Band Trend', line=dict(color='Orange', width=2)), row = 4, col = 1)
        elif indicator == 'Bollinger Bands':
            # Define the Bollinger Bands function
            def bollinger_bands(df, period=20, std_multiplier=2):
                rolling_mean = df['Close'].rolling(window=period).mean()
                rolling_std = df['Close'].rolling(window=period).std()
                upper_band = rolling_mean + std_multiplier * rolling_std
                lower_band = rolling_mean - std_multiplier * rolling_std
                return rolling_mean, upper_band, lower_band
            # Calculate the Bollinger Bands
            rolling_mean, upper_bollinger_band, lower_bollinger_band = bollinger_bands(df)
            fig.add_trace(go.Scatter(x=df.index, y=upper_bollinger_band, name='Upper Band', line=dict(color='black', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=rolling_mean, name='Rolling Mean', line=dict(color='black', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=lower_bollinger_band, name='Lower Band', line=dict(color='black', width=1.5)), row=1, col=1)
        elif indicator == 'Double Bollinger Band':
            def double_bollinger_bands(df, period=20, std_multiplier1=2, std_multiplier2=3):
                rolling_mean = df['Close'].rolling(window=period).mean()
                rolling_std = df['Close'].rolling(window=period).std()
                upper_band1 = rolling_mean + std_multiplier1 * rolling_std
                lower_band1 = rolling_mean - std_multiplier1 * rolling_std
                upper_band2 = rolling_mean + std_multiplier2 * rolling_std
                lower_band2 = rolling_mean - std_multiplier2 * rolling_std
                return rolling_mean, upper_band1, lower_band1, upper_band2, lower_band2
            rolling_mean, upper_band1, lower_band1, upper_band2, lower_band2 = double_bollinger_bands(df)
            fig.add_trace(go.Scatter(x=df.index, y=upper_band2, name='Upper Band 2', line=dict(color='gray', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=upper_band1, name='Upper Band 1', line=dict(color='black', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=rolling_mean, name='Rolling Mean', line=dict(color='orange', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=lower_band1, name='Lower Band 1', line=dict(color='black', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=lower_band2, name='Lower Band 2', line=dict(color='gray', width=1.5)), row=1, col=1)
        elif indicator == 'Stochastic Oscillator':
            def stochastic_oscillator(df, k_period=14, d_period=3):
                low_min = df['Low'].rolling(window=k_period).min()
                high_max = df['High'].rolling(window=k_period).max()
                k = 100 * (df['Close'] - low_min) / (high_max - low_min)
                d = k.rolling(window=d_period).mean()
                return k, d
            k_period=14
            d_period=3
            k, d = stochastic_oscillator(df, k_period, d_period)
            # Add the k line trace to the figure
            fig.add_trace(go.Scatter(x=df.index, y=k, name='K Line',line=dict(color='blue', width=2)), row = 4, col = 1)
            fig.add_trace(go.Scatter(x=df.index, y=d, name='D Line',line=dict(color='orange', width=2)), row = 4, col = 1)
        elif indicator == 'RSI':
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            df['20RSI'] = rsi.rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='green', width=2)), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['20RSI'], name='Mean RSI', line=dict(color='Orange', width=2)), row = 2, col = 1)
        elif indicator == 'MACD':
            # Calculate the MACD
            df['12EMA'] = df['Close'].ewm(span=12).mean()
            df['26EMA'] = df['Close'].ewm(span=26).mean()
            df['MACD'] = df['12EMA'] - df['26EMA']
            df['Signal Line'] = df['MACD'].ewm(span=9).mean()
            df['Histogram'] = df['MACD'] - df['Signal Line']
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=2)), row = 3, col = 1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Signal Line'], name='Signal', line=dict(color='red', width=2)), row = 3, col = 1)
            fig.add_trace(go.Bar(x=df.index, y=df['Histogram'], name='Histogram', marker=dict(color=df['Histogram'], colorscale='rdylgn')), row = 3, col = 1)
        elif indicator == 'Parabolic Stop & Reverse (PSAR)':
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
            fig.add_trace(go.Scatter(x=dates, y=df["psarbull"], name='buy',mode = 'markers',
                                     marker = dict(color='green', size=2)))
            fig.add_trace(go.Scatter(x=dates, y=df["psarbear"], name='sell', mode = 'markers',
                                     marker = dict(color='red', size=2)))
        elif indicator == 'Donchian Channels':
            # Calculate the Donchian Channels
            n = 20
            df['Upper Band'] = df['High'].rolling(n).max()
            df['Lower Band'] = df['Low'].rolling(n).min()
            df['Middle Band'] = (df['Upper Band'] + df['Lower Band']) / 2
            fig.add_trace(go.Scatter(x=df.index, y=df['Upper Band'], 
                            mode='lines', name='Upper Band', line=dict(color='skyblue', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['Lower Band'], 
                            mode='lines', name='Lower Band', line=dict(color='skyblue', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['Middle Band'], 
                            mode='lines', name='Middle Band', line=dict(color='black', width=2)))
        elif indicator == 'Double Donchian Strategy':
            # Calculate the Donchian Channels
            n = 20
            n1 = 10
            df['Upper Band'] = df['High'].rolling(n).max()
            df['Lower Band'] = df['Low'].rolling(n).min()
            df['Upper Band 1'] = df['High'].rolling(n1).max()
            df['Lower Band 1'] = df['Low'].rolling(n1).min()
            df['Middle Band'] = (df['Upper Band'] + df['Lower Band']) / 2
            fig.add_trace(go.Scatter(x=df.index, y=df['Upper Band'], 
                            mode='lines', name='Upper Band', line=dict(color='skyblue', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['Lower Band'], 
                            mode='lines', name='Lower Band', line=dict(color='skyblue', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['Upper Band 1'], 
                            mode='lines', name='Upper Band', line=dict(color='tomato', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['Lower Band 1'], 
                            mode='lines', name='Lower Band', line=dict(color='tomato', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['Middle Band'], 
                            mode='lines', name='Middle Band', line=dict(color='black', width=2)))
        elif indicator == 'Average True Range (ATR)':
            # calculate the Average True Range (ATR)
            df['tr1'] = abs(df['High'] - df['Low'])
            df['tr2'] = abs(df['High'] - df['Close'].shift())
            df['tr3'] = abs(df['Low'] - df['Close'].shift())
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr'] = df['tr'].rolling(14).mean()
            df['20atr'] = df['atr'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=df.index, y=df['atr'], name='ATR', line=dict(color='purple', width=2)), row = 4, col = 1)
            fig.add_trace(go.Scatter(x=df.index, y=df['20atr'], name='Mean ATR', line=dict(color='orange', width=2)), row = 4, col = 1)
        elif indicator == 'Average Directional Index (ADX)':
            # calculate the Average Directional Index (ADX)
            df['up_move'] = df['High'] - df['High'].shift()
            df['down_move'] = df['Low'].shift() - df['Low']
            df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
            df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
            df['plus_di'] = 100 * (df['plus_dm'] / df['atr']).ewm(span=14, adjust=False).mean()
            df['minus_di'] = 100 * (df['minus_dm'] / df['atr']).ewm(span=14, adjust=False).mean()
            df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])).ewm(span=14, adjust=False).mean()
            df['adx'] = df['dx'].ewm(span=14, adjust=False).mean()
            fig.add_trace(go.Scatter(x=df.index, y=df['adx'], name='ADX', line=dict(color='blue', width=2)), row = 5, col = 1)
        elif indicator == 'Supertrend (Default)':
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
            atr_period = 10
            atr_multiplier = 3
            supertrend = Supertrend(df, atr_period, atr_multiplier)
            df = df.join(supertrend)
        
            fig.add_trace(go.Scatter(x=df.index, y=df['Final Lowerband'], name='Supertrend Lower Band',
                                     line = dict(color='green', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['Final Upperband'], name='Supertrend Upper Band',
                                     line = dict(color='red', width=2)))
        elif indicator == 'Dual Supertrend (Medium)':
            df1 = df.copy()
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
            atr_period = 10
            atr_multiplier = 3
            supertrend = Supertrend(df, atr_period, atr_multiplier)
            df = df.join(supertrend)

            st_1 = Supertrend(df1, 20, 5)
            df1 = df1.join(st_1)
            
            fig.add_trace(go.Scatter(x=df.index, y=df1['Final Lowerband'], name='Supertrend Lower Band',
                                     line = dict(color='green', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df1['Final Upperband'], name='Supertrend Upper Band',
                                     line = dict(color='red', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['Final Lowerband'], name='Supertrend Fast Lower Band',
                         line = dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['Final Upperband'], name='Supertrend Fast Upper Band',
                         line = dict(color='purple', width=2)))
        elif indicator == 'Dual Supertrend (Fast)':
            df1 = df.copy()
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
            atr_period = 10
            atr_multiplier = 3
            supertrend = Supertrend(df, atr_period, atr_multiplier)
            df = df.join(supertrend)

            st_1 = Supertrend(df1, 10, 1)
            df1 = df1.join(st_1)
            
            fig.add_trace(go.Scatter(x=df.index, y=df['Final Lowerband'], name='Supertrend Lower Band',
                                     line = dict(color='green', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['Final Upperband'], name='Supertrend Upper Band',
                                     line = dict(color='red', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df1['Final Lowerband'], name='Supertrend Fast Lower Band',
                         line = dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df1['Final Upperband'], name='Supertrend Fast Upper Band',
                         line = dict(color='purple', width=2)))

            
                
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


indicators = ['5SMA','9SMA','20SMA', '50SMA', '200SMA', '8EMA','13EMA','21EMA','50EMA','200EMA','Bollinger Bands','Double Bollinger Band','Percent %B','Bollinger Band Width','Bollinger Band Trend','Parabolic Stop & Reverse (PSAR)', 'Supertrend (Default)', 'Dual Supertrend (Fast)', 'Dual Supertrend (Medium)', 'Donchian Channels', 'Double Donchian Strategy', 'RSI', 'MACD','Stochastic Oscillator','Average True Range (ATR)','Average Directional Index (ADX)' ]

selected_indicators = st.multiselect('Select Indicators', indicators)


create_plot(df, selected_indicators)
