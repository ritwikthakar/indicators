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
i = st.sidebar.selectbox('Select Time Granularity', ('1d', '1wk', '1h', '15m'))
st.header(f'{ticker.upper()} Technical Indicators')

start = dt.datetime.today()-dt.timedelta(t)
end = dt.datetime.today()
df = yf.download(ticker, start, end, interval= i)

df.ta.strategy("All")

df.ta.ema(length=8, append=True)
df.ta.ema(length=13, append=True)
df.ta.ema(length=21, append=True)

df.ta.sma(length=9, append=True)
df.ta.sma(length=50, append=True)
df.ta.sma(length=100, append=True)

df.ta.bbands(close=df['Adj Close'], length=20, std=2, append=True)

df.ta.ichimoku()

def get_fill_color(label):
    if label >= 1:
        return 'rgba(0,250,0,0.4)'
    else:
        return 'rgba(250,0,0,0.4)'

fig = make_subplots(rows=4, cols=1, vertical_spacing = 0.05, subplot_titles=(f"{ticker.upper()} Daily Candlestick Chart", "Lower Indicator 1", "Lower Indicator 2",  "Lower Indicator 3"))

fig.append_trace(
    go.Candlestick(
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

fig.add_trace(go.Scatter(x = df.index, y=df['BBU_20_2.0'], line_color = 'black', name = 'Bollinger Upper Bnad'), row =1, col = 1)
fig.add_trace(go.Scatter(x = df.index, y=df['BBM_20_2.0'], line_color = 'orange', name = '20 SMA'), row =1, col = 1)
fig.add_trace(go.Scatter(x = df.index, y=df['BBL_20_2.0'], line_color = 'black', name = 'Bollinger Lower Bnad'), row =1, col = 1)
fig.add_trace(go.Scatter(x = df.index, y=df['EMA_8'], line_color = 'blue', name = '8 EMA', visible='legendonly'), row =1, col = 1)
fig.add_trace(go.Scatter(x = df.index, y=df['EMA_13'], line_color = 'yellow', name = '13 EMA', visible='legendonly'), row =1, col = 1)
fig.add_trace(go.Scatter(x = df.index, y=df['EMA_21'], line_color = 'red', name = '13 EMA', visible='legendonly'), row =1, col = 1)
fig.add_trace(go.Scatter(x = df.index, y=df['SMA_9'], line_color = 'blue', name = '9 SMA', visible='legendonly'), row =1, col = 1)
fig.add_trace(go.Scatter(x = df.index, y=df['SMA_50'], line_color = 'green', name = '50 SMA', visible='legendonly'), row =1, col = 1)
fig.add_trace(go.Scatter(x = df.index, y=df['SMA_100'], line_color = 'purple', name = '100 SMA', visible='legendonly'), row =1, col = 1)
fig.add_trace(go.Scatter(x = df.index, y=df['SUPERTl_7_3.0'], line_color = 'green', name = 'Supertrend-L', visible='legendonly'), row =1, col = 1)
fig.add_trace(go.Scatter(x = df.index, y=df['SUPERTs_7_3.0'], line_color = 'red', name = 'Supertrend-S', visible='legendonly'), row =1, col = 1)
fig.add_trace(go.Scatter(x = df.index, y=df['PSARl_0.02_0.2'], mode = 'markers', marker = dict(color='green', size=2), name = 'PSAR-L', visible='legendonly'), row =1, col = 1)
fig.add_trace(go.Scatter(x = df.index, y=df['PSARs_0.02_0.2'], mode = 'markers', marker = dict(color='red', size=2), name = 'PSAR-S', visible='legendonly'), row =1, col = 1)
fig.add_trace(go.Scatter(x = df.index, y=df['MACD_12_26_9'], line_color = 'orange', name = 'macd'), row = 2, col=1)
fig.add_trace(go.Scatter(x = df.index, y=df['MACDs_12_26_9'], line_color = 'deepskyblue', name='sig'), row =2, col = 1)
colors = ['green' if val > 0 else 'red' for val in df['MACDh_12_26_9']]
fig.add_trace(go.Bar(x=df.index, y= df['MACDh_12_26_9'],  marker_color=colors, showlegend = False), row = 2, col=1)
fig.add_trace(go.Scatter(x = df.index, y=df['RSI_14'], line_color = 'green', name = 'RSI'), row =3, col = 1)
fig.add_trace(go.Scatter(x = df.index, y=df['ATRr_14'], line_color = 'red', name = 'ATR', visible='legendonly'), row = 3, col =1)
fig.add_trace(go.Scatter(x = df.index, y=df['ADX_14'], line_color = 'orange', name = 'ADX', visible='legendonly'), row = 3, col=1)
fig.add_trace(go.Scatter(x = df.index, y=df['ROC_10'], line_color = 'blue', name = 'ROC', visible='legendonly'), row = 3, col=1)
fig.add_trace(go.Scatter(x = df.index, y=df['CCI_14_0.015'], line_color = 'maroon', name = 'CCI', visible='legendonly'), row = 3, col=1)
fig.add_trace(go.Scatter(x = df.index, y=df['BOP'], line_color = 'Brown', name = 'BOP', visible='legendonly'), row = 3, col=1)
fig.add_trace(go.Scatter(x = df.index, y=df['OBV'], line_color = 'purple', name = 'OBV', visible='legendonly'), row = 3, col=1)
fig.add_trace(go.Scatter(x = df.index, y=df['STOCHRSIk_14_14_3_3'], line_color = 'orange', name = 'Stochastic RSI %K'), row = 4, col=1)
fig.add_trace(go.Scatter(x = df.index, y=df['STOCHRSId_14_14_3_3'], line_color = 'blue', name = 'Stochastic RSI %D'), row = 4, col=1)
fig.add_trace(go.Scatter(x = df.index, y=df['STOCHk_14_3_3'], line_color = 'orange', name = 'Stochastic %K', visible='legendonly'), row = 4, col=1)
fig.add_trace(go.Scatter(x = df.index, y=df['STOCHd_14_3_3'], line_color = 'blue', name = 'Stochastic %D', visible='legendonly'), row = 4, col=1)
fig.add_trace(go.Scatter(x = df.index, y=df['EBSW_40_10'], line_color = 'blue', name='Sine Wave', visible='legendonly'), row =4, col = 1)

# Plotting Ichimoku
baseline = go.Scatter(x=df.index, y=df['IKS_26'], 
                   line=dict(color='orange', width=2),visible='legendonly', name="Baseline")

conversion = go.Scatter(x=df.index, y=df['ITS_9'], 
                  line=dict(color='blue', width=1),visible='legendonly', name="Conversionline")

lagging = go.Scatter(x=df.index, y=df['ICS_26'], 
                  line=dict(color='purple', width=2, dash='solid'),visible='legendonly', name="Lagging")

span_a = go.Scatter(x=df.index, y=df['ISA_9'],
                  line=dict(color='green', width=2, dash='solid'),visible='legendonly', name="Span A")

span_b = go.Scatter(x=df.index, y=df['ISB_26'],
                    line=dict(color='red', width=1, dash='solid'),visible='legendonly', name="Span B")

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
                              line = dict(color='rgba(0,0,0,0)'), visible='legendonly'))
    
    fig.add_traces(go.Scatter(x=df.index, y = df['ISB_26'],
                              line = dict(color='rgba(0,0,0,0)'), visible='legendonly',
                              fill='tonexty', 
                              fillcolor = get_fill_color(df['label'].iloc[0])))
# Make it pretty
layout = go.Layout(
    plot_bgcolor='#efefef',
    # Font Families
    font_family='Monospace',
    font_color='#000000',
    font_size=20,
    height=1600, width=800,
)

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


# Update options and show plot
fig.update_layout(layout)

st.plotly_chart(fig)
