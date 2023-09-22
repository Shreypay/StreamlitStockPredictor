import math
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import yfinance as yf
from datetime import date
import plotly.graph_objects as go

#UI

st.title("Stock Prediction App")
password = st.text_input("Enter your password:", type="password")
if password == "1333":
  stocks = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN", "Other"]
  ticker = st.selectbox("Select dataset for preediction",stocks)
  if ticker == "Other":
      custom_symbol = st.sidebar.text_input("Enter Custom Stock Symbol", "")
      if custom_symbol:
          ticker = custom_symbol

  start_date = "2017-01-01"
  end_date = date.today().strftime("%Y-%m-%d")

  data = yf.download(ticker,start=start_date, end=end_date)
  df=pd.DataFrame(data)
  # df['date'] = pd.to_datetime(df.index)

  #Graphs
  st.write('Today\'s closing price:', df['Close'].iloc[-1])
  st.header('Data')
  st.write('Raw Data')
  st.write(data)

  def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index,y=data['Open'],name = 'Stock_open'))
    fig.add_trace(go.Scatter(x=data.index,y=data['Close'],name = 'Stock_close'))
    fig.layout.update(title_text="Linear Stock Price Chart "+ticker,xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
  plot_raw_data()


  fig = go.Figure(data=[go.Candlestick(
      x=df.index,  # Date column for the x-axis
      open=df['Open'],  # Open price column
      high=df['High'],  # High price column
      low=df['Low'],  # Low price column
      close=df['Close']  # Close price column
  )])
  fig.update_layout(
      title= "Candle Stick Stock Price Chart "+ticker,
      yaxis_title='Price ($)',
      xaxis_rangeslider_visible=True
  )
  st.plotly_chart(fig)

  #ML Training
  data = df.filter(['Close'])
  dataset = data.values
  training_data_len = math.ceil(len(dataset)*.8)

  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(dataset)

  train_data = scaled_data[0:training_data_len]
  x_train = []
  y_train = []
  for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<=60:
      print(x_train)
      print(y_train)
      print()
  x_train, y_train = np.array(x_train),np.array(y_train)
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
  model = Sequential()
  model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
  model.add(LSTM(50, return_sequences=False))
  model.add(Dense(25))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mean_squared_error')
  model.fit(x_train,y_train,batch_size=1,epochs=1)
  test_data = scaled_data[training_data_len-60:,:]
  x_test = []
  y_test = dataset[training_data_len:,:]
  for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0])
  x_test = np.array(x_test)
  x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
  predictions = model.predict(x_test)
  predictions = scaler.inverse_transform(predictions)
  rmse = np.sqrt(np.mean(predictions-y_test)**2)
  train = data[:training_data_len]
  valid = data[training_data_len:]
  valid['Predictions'] = predictions

  # plt.figure(figsize=(16,8))
  # plt.title('Prediction')
  # plt.xlabel('Date',fontsize=18)
  # plt.ylabel('Close Price USD($)',fontsize=18)
  # plt.plot(train['Close'])
  # plt.plot(valid[['Close','Predictions']])
  # plt.legend(['Train','Test','Predictions'], loc='lower right')
  # plt.show()
  st.header('Prediction')

  def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index,y=train['Close'],name = 'Close Training Data'))
    fig.layout.update(title_text="Testing Data for  "+ticker,xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
  plot_raw_data()
  def plot_prediction_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index,y=train['Close'],name = 'Close Training Data'))
    fig.add_trace(go.Scatter(x=valid.index,y=valid['Close'],name = 'Actual Close'))
    fig.add_trace(go.Scatter(x=valid.index,y=valid['Predictions'],name = 'Predicted Close'))
    fig.layout.update(title_text="Linear Prediction vs Actual Stock Price Chart "+ticker,xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
  plot_prediction_data()

  st.write(valid.tail())

  # Get the last 60 days of closing price 
  last_60_days = data['Close'][-60:].values
  last_60_days_scaled = scaler.transform(last_60_days.reshape(-1,1))

  # Create an empty list to store the predictions
  predictions = []

  # Loop over the next 90 days
  for i in range(90):
      # Prepare the data for prediction
      X_test = np.array([last_60_days_scaled[i:i+60]])
      X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

      # Get the predicted scaled price
      pred_price = model.predict(X_test)

      # Undo the scaling 
      pred_price = scaler.inverse_transform(pred_price)

      # Append the predicted price to the predictions list
      predictions.append(pred_price[0][0])

      # Append the predicted scaled price to the last 60 days scaled prices
      last_60_days_scaled = np.append(last_60_days_scaled, scaler.transform(pred_price), axis=0)

  # Create a new dataframe for the future predictions
  future_dates = pd.date_range(start=data.index[-1]+pd.DateOffset(1), periods=90)
  future_df = pd.DataFrame(predictions, index=future_dates, columns=['Predictions'])

  # Concatenate the original dataframe with the future dataframe
  df = pd.concat([data, future_df])

  # Plot the predictions
  st.header('3-Months Future Prediction')
  def plot_future_prediction_data():
      fig = go.Figure()
      fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Actual Close'))
      fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Predictions'], name='Predicted Close'))
      fig.layout.update(title_text="Linear Prediction Stock Price Chart "+ticker, xaxis_rangeslider_visible=True)
      st.plotly_chart(fig)
  plot_future_prediction_data()

  st.write(future_df.tail(180))
else:
  st.write("Damm Shrey told you the wrong password")
  # import streamlit as st
  # import yfinance as yf
  # import pandas as pd
  # import os
  # from datetime import date
  # from datetime import datetime, timedelta
  # import plotly.graph_objects as go
  # import numpy as np
  # from fbprophet import Prophet
  # from fbprophet.plot import plot_plotly

  # #UI
  # st.title("Stock Prediction App")
  # st.write("This application uses ML to detrmine a stocks closing price the next day. This application uses data from 2020 to current day")
  # stocks = ("AAPL", "MSFT", "MRVL","AMZN", "GOOG", "FB", "TSLA", "BRK-B", "V", "JNJ", "WMT", "JPM", "MA", "PG", "UNH", "DIS", "NVDA", "HD", "PYPL", "BAC", "VZ", "ADBE", "CMCSA", "KO", "NKE", "MRK", "PEP", "PFE", "NFLX", "T", "ABT", "ORCL", "CRM", "ABBV", "CSCO", "TMO", "AVGO", "XOM", "ACN", "QCOM", "COST", "CVX", "LLY", "MCD", "DHR", "MDT", "NEE", "TXN", "UNP", "LIN", "BMY","ABNB")
  # ticker = st.selectbox("Select dataset for preediction",stocks)

  # start_date = "2020-01-01"
  # end_date = date.today().strftime("%Y-%m-%d")

  # data = yf.download(ticker,start=start_date, end=end_date)
  # df=pd.DataFrame(data)
  # df['date'] = pd.to_datetime(df.index)


  # #Data ML Training
  # df.reset_index(drop=True, inplace=True)
  # import pandas as pd
  # from sklearn.ensemble import RandomForestRegressor
  # from sklearn.model_selection import train_test_split
  # from sklearn.metrics import mean_squared_error

  # x = df[['Open','Close','High','Low','Adj Close']]
  # y = df['Close']
  # x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

  # rf = RandomForestRegressor(n_estimators=100,random_state=42)
  # rf.fit(x_train,y_train)

  # y_pred = rf.predict(x_test)
  # mse = mean_squared_error(y_test,y_pred)
  # st.write("Accuracy:",100-(mse*100),"%")
  # import numpy as np

  # #Change based on ticker
  # # Download the most recent data
  # most_recent_data = yf.download(ticker, start=start_date, end=end_date)

  # # Select the features for the most recent day and convert to a numpy array
  # new_data = most_recent_data[['Open', 'Close', 'High', 'Low', 'Adj Close']].iloc[-1].values.reshape(1, -1)

  # # Use the model to make a prediction for the next day
  # predicted_price = rf.predict(new_data)

  # st.write('Predicted Stock Price for Tomorrow\'s Close:', predicted_price[0])

  # #Graphs
  # st.header('Data')
  # st.write('Raw Data')
  # st.write(data.tail())

  # def plot_raw_data():
  #   fig = go.Figure()
  #   fig.add_trace(go.Scatter(x=data.index,y=data['Open'],name = 'Stock_open'))
  #   fig.add_trace(go.Scatter(x=data.index,y=data['Close'],name = 'Stock_close'))
  #   fig.layout.update(title_text="Linear Stock Price Chart "+ticker,xaxis_rangeslider_visible=True)
  #   st.plotly_chart(fig)
  # plot_raw_data()


  # fig = go.Figure(data=[go.Candlestick(
  #     x=df.index,  # Date column for the x-axis
  #     open=df['Open'],  # Open price column
  #     high=df['High'],  # High price column
  #     low=df['Low'],  # Low price column
  #     close=df['Close']  # Close price column
  # )])
  # fig.update_layout(
  #     title= "Candle Stick Stock Price Chart "+ticker,
  #     yaxis_title='Price ($)',
  #     xaxis_rangeslider_visible=True
  # )
  # st.plotly_chart(fig)
