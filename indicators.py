from datetime import timedelta
import datetime as dt
import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from pandas_datareader import data as pdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta

df = pd.DataFrame()

ticker = st.sidebar.text_input('Enter Ticker', 'SPY')
# t = st.sidebar.selectbox('Select Number of Days', ('1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'))
# i = st.sidebar.selectbox('Select Time Granularity', ('1d', '1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'))
t = st.sidebar.selectbox('Select Number of Days', (180, 3000, 1000, 735, 450, 400, 350, 252, 150, 90, 60, 45, 30, 15))
i = st.sidebar.selectbox('Select Time Granularity', '1d')
st.header(f'{ticker.upper()} Technical Indicators')

start = dt.datetime.today()-dt.timedelta(t)
end = dt.datetime.today()
df = yf.download(ticker, start, end, interval= i)

df.ta.strategy("All")

df.ta.ema(length=8, append=True)
df.ta.ema(length=13, append=True)
df.ta.ema(length=21, append=True)
df.ta.ema(length=50, append=True)
df.ta.ema(length=200, append=True)

df.ta.sma(length=5, append=True)
df.ta.sma(length=9, append=True)
df.ta.sma(length=50, append=True)
df.ta.sma(length=100, append=True)
df.ta.sma(length=200, append=True)

df.ta.bbands(close=df['Adj Close'], length=20, std=2, append=True)

# Define input variables
length_ma = 34
length_signal = 9

# Define functions
def calc_smma(src, length):
    smma = []
    for i in range(len(src)):
        if i == 0:
            smma.append(src[i])
        else:
            smma.append(((length - 1) * smma[-1] + src[i]) / length)
    return smma

def calc_zlema(src, length):
    ema1 = []
    ema2 = []
    d = []
    for i in range(len(src)):
        if i == 0:
            ema1.append(src[i])
        else:
            ema1.append((2 * src[i] + (length - 1) * ema1[-1]) / (length + 1))
    for i in range(len(ema1)):
        if i == 0:
            ema2.append(ema1[i])
        else:
            ema2.append((2 * ema1[i] + (length - 1) * ema2[-1]) / (length + 1))
    for i in range(len(ema1)):
        d.append(ema1[i] - ema2[i])
    return [ema1, ema2, d]

# Calculate Impulse MACD
src = (df['High'] + df['Low'] + df['Close']) / 3
hi = calc_smma(df['High'], length_ma)
lo = calc_smma(df['Low'], length_ma)
mi = calc_zlema(src, length_ma)[0]
md = []
mdc = []
for i in range(len(mi)):
    if mi[i] > hi[i]:
        md.append(mi[i] - hi[i])
        mdc.append('lime')
    elif mi[i] < lo[i]:
        md.append(mi[i] - lo[i])
        mdc.append('red')
    else:
        md.append(0)
        mdc.append('orange')
sb = calc_smma(md, length_signal)
sh = [md[i] - sb[i] for i in range(len(md))]
            
df.ta.ichimoku()

def get_fill_color(label):
    if label >= 1:
        return 'rgba(0,250,0,0.4)'
    else:
        return 'rgba(250,0,0,0.4)'

def create_plot(df, indicators):
    fig = make_subplots(rows=5, cols=1, row_heights=[0.4, 0.15, 0.15, 0.15, 0.15], vertical_spacing = 0.05, subplot_titles=(f"{ticker.upper()} Daily Candlestick Chart", "Lower Indicator 1", "Lower Indicator 2",  "Lower Indicator 3", "Lower Indicator 4"))
     
    fig.append_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Adj Close'],
            increasing_line_color='green',
            decreasing_line_color='red',
            showlegend=False
        ), row=1, col=1
    )
    for indicator in indicators:
        if indicator == "Volume Based Support & Resistance":
            mean_volume = df['Volume'].mean()
            support_level = df[df['Volume'] < mean_volume]['Close'].max()
            resistance_level = df[df['Volume'] >= mean_volume]['Close'].min()
            fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[support_level, support_level], name='Support', line=dict(color='green', width=1, dash='dash')))
            fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]],
                                         y=[resistance_level, resistance_level],
                                         name='Resistance',
                                         line=dict(color='red', width=1, dash='dash')))
        elif indicator == "Regression Channels":
            x = np.arange(len(df))
            y = df['Close']
            p = np.polyfit(x, y, 1)
            slope = p[0]
            intercept = p[1]
            regression_line = slope * x + intercept
            upper_line = slope * x + intercept + (np.std(y) * 2)
            lower_line = slope * x + intercept - (np.std(y) * 2)
            fig.add_trace(go.Scatter(x=df.index, y=upper_line, mode='lines', name='Upper Regression Channel',
                         line=dict(color='red', width=2, dash='dash')))
            fig.add_trace(go.Scatter(x=df.index, y=lower_line, mode='lines', name='Lower Regression Channel',
                                     line=dict(color='green', width=2, dash='dash')))
            fig.add_trace(go.Scatter(x=df.index, y=regression_line, mode='lines', name='Regression Line',
                         line=dict(color='blue', width=2)))
        elif indicator == "Bollinger Bands":
            fig.add_trace(go.Scatter(x = df.index, y=df['BBU_20_2.0'], line_color = 'black', name = 'Bollinger Upper Band'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['BBM_20_2.0'], line_color = 'black', name = '20 SMA'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['BBL_20_2.0'], line_color = 'black', name = 'Bollinger Lower Band'), row =1, col = 1)
        elif indicator == "Keltner Channels":
            fig.add_trace(go.Scatter(x = df.index, y=df['KCLe_20_2'], line_color = 'gray', name = 'Keltner Channel Lower Baad'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['KCBe_20_2'], line_color = 'gray', name = 'Keltner Channel Basis'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['KCUe_20_2'], line_color = 'gray', name = 'Keltner Channel Upper Band'), row =1, col = 1)
        elif indicator == "Donchian Channels":
            fig.add_trace(go.Scatter(x = df.index, y=df['DCL_20_20'], line_color = 'skyblue', name = 'Donchian Channel Lower Baad'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['DCM_20_20'], line_color = 'skyblue', name = 'Donchian Channel Basis'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['DCU_20_20'], line_color = 'skyblue', name = 'Donchian Channel Upper Band'), row =1, col = 1)            
        elif indicator == "EMA Ribbons":
            fig.add_trace(go.Scatter(x = df.index, y=df['EMA_8'], line_color = 'purple', name = '8 EMA'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['EMA_13'], line_color = 'blue', name = '13 EMA'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['EMA_21'], line_color = 'orange', name = '21 EMA'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['EMA_13'], line_color = 'green', name = '50 EMA'), row =1, col = 1)
        elif indicator == "SMA Ribbons":
            fig.add_trace(go.Scatter(x = df.index, y=df['SMA_5'], line_color = 'purple', name = '5 SMA'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['SMA_9'], line_color = 'blue', name = '9 SMA'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['SMA_50'], line_color = 'green', name = '50 SMA'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['SMA_100'], line_color = 'yellow', name = '100 SMA'), row =1, col = 1)
        elif indicator == "200 EMA":
            fig.add_trace(go.Scatter(x = df.index, y=df['EMA_200'], line_color = 'red', name = '200 EMA'), row =1, col = 1)
        elif indicator == "200 SMA":
            fig.add_trace(go.Scatter(x = df.index, y=df['SMA_200'], line_color = 'red', name = '200 SMA'), row =1, col = 1)
        elif indicator == "Adaptive Moving Avergae":
            fig.add_trace(go.Scatter(x = df.index, y=df['KAMA_10_2_30'], line_color = 'purple', name = 'Adaptive MA'), row =1, col = 1)
        elif indicator == "Supertrend":
            fig.add_trace(go.Scatter(x = df.index, y=df['SUPERTl_7_3.0'], line_color = 'green', name = 'Supertrend-L'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['SUPERTs_7_3.0'], line_color = 'red', name = 'Supertrend-S'), row =1, col = 1)
        elif indicator == "Parabolic Stop & Reverse (PSAR)":
            fig.add_trace(go.Scatter(x = df.index, y=df['PSARl_0.02_0.2'], mode = 'markers', marker = dict(color='green', size=2), name = 'PSAR-L'), row =1, col = 1)
            fig.add_trace(go.Scatter(x = df.index, y=df['PSARs_0.02_0.2'], mode = 'markers', marker = dict(color='red', size=2), name = 'PSAR-S'), row =1, col = 1)
        elif indicator == "MACD":
            fig.add_trace(go.Scatter(x = df.index, y=df['MACD_12_26_9'], line_color = 'orange', name = 'macd'), row = 2, col=1)
            fig.add_trace(go.Scatter(x = df.index, y=df['MACDs_12_26_9'], line_color = 'deepskyblue', name='sig'), row =2, col = 1)
            colors = ['green' if val > 0 else 'red' for val in df['MACDh_12_26_9']]
            fig.add_trace(go.Bar(x=df.index, y= df['MACDh_12_26_9'],  marker_color=colors, showlegend = False), row = 2, col=1)
        elif indicator == "RSI":
            fig.add_trace(go.Scatter(x = df.index, y=df['RSI_14'], line_color = 'green', name = 'RSI'), row =3, col = 1)
        elif indicator == "ATR":    
            fig.add_trace(go.Scatter(x = df.index, y=df['ATRr_14'], line_color = 'red', name = 'ATR'), row = 4, col =1)
        elif indicator == "Chopiness Index":    
            fig.add_trace(go.Scatter(x = df.index, y=df['CHOP_14_1_100'], line_color = 'blue', name = 'Choppiness Index'), row = 4, col =1)
        elif indicator == "Squeeze Momentum Indicator Pro":
            colors = ['green' if val > 0 else 'red' for val in df['SQZPRO_20_2.0_20_2_1.5_1']]
            fig.add_trace(go.Bar(x = df.index, y=df['SQZPRO_20_2.0_20_2_1.5_1'], marker_color=colors, name = 'Squeeze Momentum Pro'), row = 4, col =1)
            fig.add_trace(go.Scatter(x = df[df['SQZPRO_OFF'] != 0].index, y=df[df['SQZPRO_OFF'] != 0]['SQZPRO_20_2.0_20_2_1.5_1'], mode = 'markers', marker = dict(color='purple', size=5), name = 'Squeeze Off'), row = 4, col =1)
            fig.add_trace(go.Scatter(x = df[df['SQZPRO_ON_WIDE'] != 0].index, y=df[df['SQZPRO_ON_WIDE'] != 0]['SQZPRO_20_2.0_20_2_1.5_1'], mode = 'markers', marker = dict(color='blue', size=5), name = 'Wide Squeeze'), row = 4, col =1)
            fig.add_trace(go.Scatter(x = df[df['SQZPRO_NO'] != 0].index, y=df[df['SQZPRO_NO'] != 0]['SQZPRO_20_2.0_20_2_1.5_1'], mode = 'markers', marker = dict(color='orange', size=5), name = 'Squeeze On'), row = 4, col =1)
            fig.add_trace(go.Scatter(x = df[df['SQZPRO_ON_NORMAL'] != 0].index, y=df[df['SQZPRO_ON_NORMAL'] != 0]['SQZPRO_20_2.0_20_2_1.5_1'], mode = 'markers', marker = dict(color='tomato', size=5), name = 'Normal Squeeze'), row = 4, col =1)
            fig.add_trace(go.Scatter(x = df[df['SQZPRO_ON_NARROW'] != 0].index, y=df[df['SQZPRO_ON_NARROW'] != 0]['SQZPRO_20_2.0_20_2_1.5_1'], mode = 'markers', marker = dict(color='orange', size=5), name = 'Narrow Squeeze'), row = 4, col =1)
        elif indicator == "ADX":
            fig.add_trace(go.Scatter(x = df.index, y=df['ADX_14'], line_color = 'orange', name = 'ADX'), row = 5, col=1)
        elif indicator == "TTM Trend":
            colors = ['green' if val > 0 else 'red' for val in df['TTM_TRND_6']]
            fig.add_trace(go.Bar(x = df.index, y=df['TTM_TRND_6'], marker_color=colors, name = 'Trend'), row = 5, col=1)
        elif indicator == "Rate of Change (ROC)":
            fig.add_trace(go.Scatter(x = df.index, y=df['ROC_10'], line_color = 'blue', name = 'ROC'), row = 5, col=1)
        elif indicator == "Commodity Channel Index (CCI)":
            fig.add_trace(go.Scatter(x = df.index, y=df['CCI_14_0.015'], line_color = 'maroon', name = 'CCI'), row = 3, col=1)
        elif indicator == "Balance of Power (BOP)":
            fig.add_trace(go.Scatter(x = df.index, y=df['BOP'], line_color = 'Brown', name = 'BOP'), row = 3, col=1)
        elif indicator == "On Balance Volume (OBV)":
            fig.add_trace(go.Scatter(x = df.index, y=df['OBV'], line_color = 'purple', name = 'OBV'), row = 3, col=1)
        elif indicator == "Srochastic RSI":
            fig.add_trace(go.Scatter(x = df.index, y=df['STOCHRSIk_14_14_3_3'], line_color = 'orange', name = 'Stochastic RSI %K'), row = 4, col=1)
            fig.add_trace(go.Scatter(x = df.index, y=df['STOCHRSId_14_14_3_3'], line_color = 'blue', name = 'Stochastic RSI %D'), row = 4, col=1)
        elif indicator == "Stochastic Oscillator":
            fig.add_trace(go.Scatter(x = df.index, y=df['STOCHk_14_3_3'], line_color = 'orange', name = 'Stochastic %K'), row = 4, col=1)
            fig.add_trace(go.Scatter(x = df.index, y=df['STOCHd_14_3_3'], line_color = 'blue', name = 'Stochastic %D'), row = 4, col=1)
        elif indicator == "Eleher's Sine Wave":
            fig.add_trace(go.Scatter(x = df.index, y=df['EBSW_40_10'], line_color = 'blue', name='Sine Wave'), row =5, col = 1)
        elif indicator == "MACD 2":
            def impulsive_macd(prices, short_period, long_period, signal_period):
                prices = df['Close']
                ema_short = prices.ewm(span=short_period, min_periods=short_period).mean()
                ema_long = prices.ewm(span=long_period, min_periods=long_period).mean()
                macd = ema_short - ema_long
                signal_line = macd.ewm(span=signal_period, min_periods=signal_period).mean()
                histogram = macd - signal_line
                return pd.concat([macd, signal_line, histogram], axis=1, keys=['MACD', 'Signal', 'Histogram'])
            prices = df['Close']
            imp_macd = impulsive_macd(prices, short_period=12, long_period=26, signal_period=9)
            line_colors = ['skyblue' if imp_macd.loc[date, 'MACD'] > imp_macd.loc[date, 'Signal'] else 'orange' for date in imp_macd.index]
            fig.add_trace(go.Bar(x=imp_macd.index, y=imp_macd['MACD'], name='Impulsive MACD', marker_color=line_colors),row = 2, col=1)
            fig.add_trace(go.Scatter(x=imp_macd.index, y=imp_macd['Signal'], line_color = 'purple',name='Imp MACD Signal Line'),row = 2, col=1)
            fig.add_trace(go.Bar(x=imp_macd.index, y=imp_macd['Histogram'], marker_color=['green' if x > 0 else 'red' for x in imp_macd['Histogram']], name='Imp MACD Histogram'),row = 2, col=1)
        elif indicator == "Impulse MACD":    
            # Create Plotly figure
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=[0] * len(df),
                    name="MidLine",
                    mode="lines",
                    line=dict(color="gray")
                ), row = 2, col=1
            )

            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=md,
                    name="ImpulseMACD",
                    marker=dict(color=mdc)
                ),row = 2, col=1
            )

            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=sh,
                    name="ImpulseHisto",
                    marker=dict(color="blue")
                ),row = 2, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=sb,
                    name="ImpulseMACDCDSignal",
                    mode="lines",
                    line=dict(color="maroon")
                ),row = 2, col=1
            )
        elif indicator == "Ichimoku Cloud":
            
            # Plotting Ichimoku
            baseline = go.Scatter(x=df.index, y=df['IKS_26'], 
                               line=dict(color='orange', width=2), name="Baseline")
            
            conversion = go.Scatter(x=df.index, y=df['ITS_9'], 
                              line=dict(color='blue', width=1), name="Conversionline")
            
            lagging = go.Scatter(x=df.index, y=df['ICS_26'], 
                              line=dict(color='purple', width=2, dash='solid'), name="Lagging")
            
            span_a = go.Scatter(x=df.index, y=df['ISA_9'],
                              line=dict(color='green', width=2, dash='solid'), name="Span A")
            
            span_b = go.Scatter(x=df.index, y=df['ISB_26'],
                                line=dict(color='red', width=1, dash='solid'), name="Span B")
            
            # Add plots to the figure
            # fig7.add_trace(candle)
            fig.add_trace(baseline)
            fig.add_trace(conversion)
            fig.add_trace(lagging)
            fig.add_trace(span_a)
            fig.add_trace(span_b)
            
            df['label'] = np.where(df['ISA_9'] > df['ISB_26'], 1, 0)
            df['group'] = df['label'].ne(df['label'].shift()).cumsum()
            df = df.groupby('group')
            
            dfs = []
            for name, data in df:
                dfs.append(data)
            
            for df in dfs:
                fig.add_traces(go.Scatter(x=df.index, y = df['ISA_9'],
                                          line = dict(color='rgba(0,0,0,0)')))
                
                fig.add_traces(go.Scatter(x=df.index, y = df['ISB_26'],
                                          line = dict(color='rgba(0,0,0,0)'), 
                                          fill='tonexty', 
                                          fillcolor = get_fill_color(df['label'].iloc[0])))
         # Make it pretty
        layout = go.Layout(
#         xaxis_rangeslider_visible=False, 
#         xaxis_tradingcalendar=True,
        plot_bgcolor='#efefef',
        # Font Families
        font_family='Monospace',
        font_color='#000000',
        font_size=20,
        height=1000, width=1200,)
        
#         if i == '1h':    
#             fig.update_xaxes(
#                     rangeslider_visible=False,
#                     rangebreaks=[
#                         # NOTE: Below values are bound (not single values), ie. hide x to y
#                         dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
#                         # dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
#                             # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
#                         ]
#                             )
#         elif i == '1wk':    
#             fig.update_xaxes(
#                     rangeslider_visible=False,
#                     rangebreaks=[
#                         # NOTE: Below values are bound (not single values), ie. hide x to y
#                         dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
#                         # dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
#                             # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
#                         ]
#                             )

#         else:
#             fig.update_xaxes(
#                     rangeslider_visible=False,
#                     rangebreaks=[
#                         # NOTE: Below values are bound (not single values), ie. hide x to y
#                         dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
#                         # dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
#                             # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
#                         ]
#                             )        
    fig.update_xaxes(
    rangebreaks=[
        dict(bounds=["sat", "mon"]), #hide weekends
        dict(values=["2015-12-25", "2016-01-01"])  # hide Christmas and New Year's
    ]
)  
    # Update options and show plot
    fig.update_layout(layout)
    st.plotly_chart(fig)

indicators = ["Volume Based Support & Resistance", "Regression Channels" ,"Bollinger Bands", "Keltner Channels" , "Donchian Channels" , "EMA Ribbons", "SMA Ribbons", "200 EMA", "200 SMA", "Adaptive Moving Avergae", "Supertrend", "Parabolic Stop & Reverse (PSAR)", "MACD", "RSI", "ATR", "Chopiness Index" , "Squeeze Momentum Indicator Pro", "ADX", "TTM Trend", "Rate of Change (ROC)", "Commodity Channel Index (CCI)" , "Balance of Power (BOP)", "On Balance Volume (OBV)","Srochastic RSI" ,"Stochastic Oscillator", "Eleher's Sine Wave", "MACD 2", "Impulse MACD" , "Ichimoku Cloud"]

default_options = ["Regression Channels","Parabolic Stop & Reverse (PSAR)", "MACD 2", "RSI", "Squeeze Momentum Indicator Pro", "ADX"]

selected_indicators = st.multiselect('Select Indicators', indicators, default=default_options)

create_plot(df, selected_indicators)
