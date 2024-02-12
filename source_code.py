import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import *
import plotly.express as px
import yfinance  as yf
from datetime import datetime
import plotly.graph_objects as go
import subprocess

#file_link=r"C:\Users\DELL\Downloads\HDFCBANK.NS.csv"
data=None
columns_to_drop=["High","Low","Adj Close","Open"]
column="Close"
window_size=5
span=3
window_size_short_term_sma=50
window_size_long_term_sma=100
window_size_short_term_ema=50
window_size_long_term_ema=100

# Data extraction 
def Stock_data(stock_symbol,start_date_str,end_date_str):
    date_formate="%Y-%m-%d"
    try:
        start_date = datetime.strptime(start_date_str,date_formate)
        end_date=datetime.strptime(end_date_str,date_formate)
        data = yf.download(stock_symbol, start=start_date, end=end_date)
        data.reset_index(inplace=True)
        print("Entered date:", start_date,end_date)
    except ValueError:
        print("Invalid date format. Please enter the date in the specified format.")
    return data

def data_extraction(file_link,columns_to_drop,column):
    years=int(input("How many years of data you wana to Analyze :"))
    column="Close"
    try:
        if file_link[-3:] == "csv":
            data = pd.read_csv(file_link)
            return data
        elif file_link[-3:] == "xls" or file_link.lower().endswith("xlsx"):
            data = pd.read_excel(file_link)
            return data
        else:
            raise ValueError("Unsupported file format")
        print("File extraction successful.")
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_link}")
    except Exception as e:
        print("An undefined error as occqured")

    data['Date'] = pd.to_datetime(data['Date'])
    max_year = data['Date'].max().year
    start_year = max_year - years
    data = data[data['Date'].dt.year >= start_year]
    data.reset_index(drop=True, inplace=True)
    max_price=data[column].max()
    min_price=data[column].min()
    mean_data=data[column].mean()
    print(f"The Maximum share price from {max_year} to {start_year} is : {max_price}")
    print(f"The Minimum share price from {max_year} to {start_year} is : {min_price}")
    print(f"The Average  share price from {max_year} to {start_year} is : {mean_data}")
    return data

def data_wrandling(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data.drop(columns_to_drop,axis=1,inplace=True)
    data=data.drop_duplicates()
    data.dropna(axis=0, how='any', subset=["Date"], inplace=True)
    data.reset_index(inplace=True, drop=True)
    return data
def stock_description(data,column):
        max_price=data[column].max()
        min_price=data[column].min()
        mean_data=data[column].mean()
        st.subheader("Stock Description:")
        st.write(f"The Maximum share price : {max_price}")
        st.write(f"The Minimum share price  : {min_price}")
        st.write(f"The Average  share price : {mean_data}")
def get_pe_ratio(stock_symbol):
        stock = yf.Ticker(stock_symbol)
        pe_ratio = stock.info['trailingPE']
        #current_price = stock.info['ask']
        pb_ratio = stock.info['priceToBook']
        st.subheader("COMAPANY DETAILS")
        st.write(f"The Price-to-Earnings (P/E) ratio for {stock_symbol} is: {pe_ratio:.2f}")
        #print(f"The current market price for {stock_symbol} is rupees :{ current_price:.2f}")#this code will work when market is open 
        st.write(f"Price-to-Book (P/B) ratio for {stock_symbol}: {pb_ratio}")
        return pe_ratio,pb_ratio #,current_price
def calculating_avgs(data, column, window_size, span):
        data['Simple_Moving_Average'] = data[column].rolling(window=window_size).mean()
        data['Simple_Moving_Average'].loc[0:2] = data[column].iloc[0:2].copy()

        for i in range(2, data.shape[0]):
            if pd.isnull(data['Simple_Moving_Average'].iloc[i]):
                data.loc[i, 'Simple_Moving_Average'] = data[column].iloc[i-2:i].mean()

        data['EMA'] = data[column].ewm(span=span, adjust=False).mean()
        data['EMA'].loc[0:1] = data[column].iloc[0:2].copy()

        for i in range(2, data.shape[0]):
            if pd.isnull(data['EMA'].iloc[i]):
                data.loc[i, 'EMA'] = data[column].iloc[i-2:i].mean()

        mask = data[column].notna()  # Mask for non-null values in 'Close'
        SMA_rmse = np.sqrt(mean_squared_error(data[column][mask], data['Simple_Moving_Average'][mask]))
        EMA_rmse = np.sqrt(mean_squared_error(data[column][mask], data['EMA'][mask]))

        if SMA_rmse < EMA_rmse:
            data[column] = data['Simple_Moving_Average'].where(data[column].isna(), data[column])
            print(f"Used Simple Moving Average to fill null values. RMSE: {SMA_rmse}")
        elif EMA_rmse < SMA_rmse:
            data[column] = data['EMA'].where(data[column].isna(), data[column])
            print(f"Used EMA Moving Average to fill null values. RMSE: {EMA_rmse}")
        else:
            print("There are no null values to fill.")
        return data


def graph_exponential_moving_avg(data):
    fig = px.line(data, x='Date', y=['Close',"EMA"],
                    labels={'value': 'Close and EMA'},
                    color_discrete_map={'Close': 'green', 'EMA': 'black'},
                    title='Moving Average that places more weight on Recent data points',
                        width=1000,
                        height=650)

    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Price')
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1),plot_bgcolor="#FBE4D8")
    st.plotly_chart(fig)
    return fig

def long_short_SME(data, column, window_size_short_term_sma,window_size_long_term_sma,short_term_column,long_term_column):
        data[short_term_column] = data[column].rolling(window=window_size_short_term_sma).mean()
        data[short_term_column].loc[0:2] = data[column].iloc[0:2].copy()

        data[long_term_column] = data[column].rolling(window=window_size_long_term_sma).mean()
        data[long_term_column].loc[0:2] = data[column].iloc[0:2].copy()

        for i in range(2, data.shape[0]):
            if pd.isnull(data[short_term_column].iloc[i]):
                data.loc[i, short_term_column] = data[column].iloc[i-2:i].mean()

        for i in range(2, data.shape[0]):
            if pd.isnull(data[long_term_column].iloc[i]):
                data.loc[i, long_term_column] = data[column].iloc[i-2:i].mean()

        fig = px.line(data, x='Date', y=['Close', 'Short_term_SSE',"Long_Term_SSE"],
                      labels={'value': 'Price and Short_term_SSE and Long_Term_SSE'},
                      color_discrete_map={'Close': 'green', 'Short_term_SSE': 'red',"Long_Term_SSE":"blue",},
                      title='Price and Short-Term and Long-Term Simple Moving Averages bull and bear signals',
                      width=1000,
                      height=650)

        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Price')
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1),plot_bgcolor="#C38EB4")
        st.plotly_chart(fig)
        return fig
        

def long_short_EMA(data, column, window_size_short_term_ema,window_size_long_term_ema,short_term_column,long_term_column):
        data[short_term_column] = data[column].ewm(span=window_size_short_term_ema, adjust=False).mean()
        data[short_term_column].loc[0:1] = data[column].iloc[0:2].copy()

        data[long_term_column] = data[column].rolling(window=window_size_long_term_ema).mean()
        data[long_term_column].loc[0:1] = data[column].iloc[0:2].copy()

        for i in range(2, data.shape[0]):
            if pd.isnull(data[short_term_column].iloc[i]):
                data.loc[i, short_term_column] = data[column].iloc[i-2:i].mean()

        for i in range(2, data.shape[0]):
            if pd.isnull(data[long_term_column].iloc[i]):
                data.loc[i, long_term_column] = data[column].iloc[i-2:i].mean()

    # Example usage    
        fig = px.line(data, x='Date', y=['Close', 'Short_term_EMA',"Long_Term_EMA"],
                      labels={'value': 'Price and Short_term_EMA and Long_Term_EMA'},
                      color_discrete_map={'Close': 'green', 'Short_term_EMA': 'red',"Long_Term_EMA":"blue",},
                      title='Price and Exponential Moving Average (EMA) short and long term ',
                      width=1000,
                      height=650)

        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Price')
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1),plot_bgcolor="#6DA5C0")
        st.plotly_chart(fig)
        return fig

def Relative_Strength_index(data):
    # Calculate RSI
    delta = data["Close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(com=14 - 1, min_periods=14).mean()
    avg_loss = losses.ewm(com=14 - 1, min_periods=14).mean()
    data["RSI"] = 100 - 100 / (1 + avg_gain / avg_loss)
    
    # Detect bullish and bearish signals
    bullish_signals = (data["RSI"] > 30) & (data["Short_term_EMA"] > data["Long_Term_EMA"])
    bearish_signals = (data["RSI"] < 70) & (data["Short_term_EMA"] < data["Long_Term_EMA"])

    # Create Plotly traces
    close_trace = go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="Close Price", line=dict(color='black'))
    short_ema_trace = go.Scatter(x=data["Date"], y=data["Short_term_EMA"], mode="lines", name="EMA short term", line=dict(color='orange'))
    long_ema_trace = go.Scatter(x=data["Date"], y=data["Long_Term_EMA"], mode="lines", name="EMA long term", line=dict(color='blue'))
    bullish_signal_trace = go.Scatter(x=data["Date"][bullish_signals], y=data["Close"][bullish_signals], mode="markers", name="Bullish Signal", marker=dict(color='green', size=8))
    bearish_signal_trace = go.Scatter(x=data["Date"][bearish_signals], y=data["Close"][bearish_signals], mode="markers", name="Bearish Signal", marker=dict(color='red', size=8))

    # Create Plotly figure
    fig = go.Figure(data=[close_trace, short_ema_trace, long_ema_trace, bullish_signal_trace, bearish_signal_trace])

    # Configure layout
    fig.update_layout(title="Price with EMAs, RSI, and Signals",
                      xaxis_title="Date",
                      yaxis_title="Price",
                      xaxis=dict(type='date', tickformat='%Y-%m-%d'),
                      yaxis=dict(tickformat='Price: ,.2f'),
                      legend=dict(x=0.02, y=0.98),
                      plot_bgcolor='rgba(0,0,0,0)',
                      hovermode="x",
                      width=1000,
                      height=650)

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)
def volume_analysis(data):
    # Create Plotly trace
    volume_trace = go.Scatter(x=data["Date"], y=data["Volume"], mode="lines", name="Volume", line=dict(color='#F1916D'))

    # Create Plotly figure
    fig = go.Figure(data=[volume_trace])

    # Configure layout
    fig.update_layout(title="Date volume",
                      xaxis_title="Date",
                      yaxis_title="Volume",
                      xaxis=dict(type='date', tickformat='%Y-%m-%d'),
                      yaxis=dict(tickformat=',.0f'),
                      legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1),
                      plot_bgcolor='rgba(0,0,0,0)',
                      width=1000,
                      height=650)

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

def resistance_supports(data):
    min_close = data['Close'].min()
    max_close = data['Close'].max()

    # First support and resistance levels
    support1 = min_close
    resistance1 = max_close

    # Calculate second and third support levels as a percentage of the range
    range_percentage = 0.05
    support2 = min_close - range_percentage * (max_close - min_close)
    support3 = min_close - 2 * range_percentage * (max_close - min_close)

    # Create a Plotly figure
    fig = go.Figure()

    # Plotting the stock prices
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Stock Price', line=dict(color='black')))

    # Adding support and resistance lines
    fig.add_shape(type="line", x0=data['Date'].iloc[0], x1=data['Date'].iloc[-1], y0=support1, y1=support1,
                   line=dict(color='green', width=2, dash='dash'), name='Support1')

    fig.add_shape(type="line", x0=data['Date'].iloc[0], x1=data['Date'].iloc[-1], y0=resistance1, y1=resistance1,
                   line=dict(color='red', width=2, dash='dash'), name='Resistance1')

    fig.add_shape(type="line", x0=data['Date'].iloc[0], x1=data['Date'].iloc[-1], y0=support2, y1=support2,
                   line=dict(color='orange', width=2, dash='dash'), name='Support2')

    fig.add_shape(type="line", x0=data['Date'].iloc[0], x1=data['Date'].iloc[-1], y0=support3, y1=support3,
                   line=dict(color='yellow', width=2, dash='dash'), name='Support3')

    # Adding labels and title
    fig.update_layout(title='Support and Resistance Levels OF the SHARE',
                      xaxis_title='Date',
                      yaxis_title='Stock Price',
                      height=700,
                      width=1000)

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)
def graph_simple_removing_avg(data):

        fig = px.line(data, x='Date', y=['Close',"Simple_Moving_Average"],
                      labels={'value': 'Close and Simple_Moving_Average and EMA'},
                      color_discrete_map={'Close': 'green', 'Simple_Moving_Average': 'red',"EMA":"yellow"},
                      title='Simple Moving Average reveal underlying trends and patterns for buy and sell signals',
                         width=1100,
                        height=650)

        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Price')
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1),plot_bgcolor="#E8BCB9")
        #st.plotly_chart(fig)
        return fig