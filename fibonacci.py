import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go

def fibonacci_retracement(high, low):
    levels = []
    diff = high - low
    levels.append(high)
    levels.append(high - 0.236 * diff)
    levels.append(high - 0.382 * diff)
    levels.append(high - 0.5 * diff)
    levels.append(high - 0.618 * diff)
    levels.append(low)
    return levels

def fibonacci_extension(high, low, levels):
    diff = high - low
    extension_levels = []
    extension_levels.append(high + 1.618 * diff)
    extension_levels.append(high + diff)
    extension_levels.append(high + 0.618 * diff)
    extension_levels.append(high)
    extension_levels.append(low - 0.618 * diff)
    extension_levels.append(low)
    extension_levels.append(low - 1.618 * diff)
    extension_levels.append(low - 2.618 * diff)
    return extension_levels

st.title("Trend-based Fibonacci Retracement with yfinance data")

ticker = st.text_input("Enter a stock symbol (e.g. AAPL)", "AAPL")
start_date = st.date_input("Enter a start date", value=None)
end_date = st.date_input("Enter an end date", value=None)

if start_date and end_date:
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    high = max(stock_data["High"])
    low = min(stock_data["Low"])
    levels = fibonacci_retracement(high, low)
    extension_levels = fibonacci_extension(high, low, levels)
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=stock_data.index, open=stock_data["Open"], high=stock_data["High"], low=stock_data["Low"], close=stock_data["Close"], name="Price"))
    for i, level in enumerate(levels):
        fig.add_shape(
            type="line",
            x0=stock_data.index[0],
            y0=level,
            x1=stock_data.index[-1],
            y1=level,
            line=dict(
                color="green",
                width=1,
                dash="dashdot",
            )
        )
        fig.add_annotation(
            x=stock_data.index[-1],
            y=level,
            text=f"{level:.2f}",
            showarrow=False,
            font=dict(size=10),
            xshift=10,
            yshift=-10,
            align="left"
        )
#     for i, level in enumerate(extension_levels):
#         fig.add_shape(
#             type="line",
#             x0=stock_data.index[0],
#             y0=level,
#             x1=stock_data.index[-1],
#             y1=level,
#             line=dict(
#                 color="red",
#                 width=1,
#                 dash="dashdot",
#             ),
#             name=f"Fib Extension {i+1}"
#         )
        
#         fig.add_annotation(
#             x=stock_data.index[-1],
#             y=extension_levels,
#             text=f"{level:.2f}",
#             showarrow=False,
#             font=dict(size=10),
#             xshift=10,
#             yshift=-10,
#             align="right"
#         )
        
    layout = go.Layout(
#         xaxis_rangeslider_visible=False, 
#         xaxis_tradingcalendar=True,
        plot_bgcolor='#efefef',
        # Font Families
        font_family='Monospace',
        font_color='#000000',
        font_size=20,
        height=600, width=1000,)
    
    fig.update_xaxes(
                rangeslider_visible=False,
                rangebreaks=[
                    # NOTE: Below values are bound (not single values), ie. hide x to y
                    dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
                        # dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
                    ]
                        )
    fig.update_layout(layout)
    
    df = pd.DataFrame({"Fibonacci Level": ["0% Fibonaci Level", "23.6% Fibonaci Level", "38.2%% Fibonaci Level", "50% Fibonaci Level", "61.8%% Fibonaci Level", "100% Fibonaci Level"], "Price": levels})
    
    df2 = pd.DataFrame({"Fibonacci Extension Level": ["Fibonaci Extension Level 1", "Fibonaci Extension Level 2", "Fibonaci Extension Level 3", "Fibonaci Extension Level 4", "Fibonaci Extension Level 5", "Fibonaci Extension Level 6","Fibonaci Extension Level 7", "Fibonaci Extension Level 8"], "Price": extension_levels})
    
    st.write(df, df2, columns=2)
    
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)
