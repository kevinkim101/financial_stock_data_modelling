import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
import datetime

# Clean the data
df = pd.read_csv("project_name/data/processed/appl_stock_values.csv", index_col = 0, header=0)
df["Date"] = pd.to_datetime(df["Date"])
del df["Open"]
del df["High"]
del df["Low"]
del df["Volume"]
df["Date Copy"] = df["Date"]
df = df.set_index("Date")

# Check if the given data is stationary
ADF_result = adfuller(df["Close"])

# Make the data stationary by applying first order differencing
stock_close_diff = np.diff(df["Close"], n = 1)
ADF_result = adfuller(stock_close_diff)

# Implement the moving average model

# MA_model uses the ARIMA function but set the parameters p and d to 0. Set the q (lag) value to 10 as identified in ACF plot.
MA_model = ARIMA(endog = df["Close"], order = (0,0,10)) 
results = MA_model.fit()

# Perform forecasting
df["Forecast"] = results.predict(start = "2021-01-04", end = "2021-12-22")

# Rolling average analysis with MA20 and MA50
df["MA20"] = df["Forecast"].rolling(20).mean()
df["MA10"] = df["Forecast"].rolling(10).mean()
df = df.dropna()

buy = {}
sell = {}

Buy = []
Sell = []

# Follow the Golden Cross Over Rule to determine buy and sell signals
for i in range(len(df)):
    # If the shorter MA is higher than the longer MA; and the opposite the previous day - buy
    if (df["MA10"].iloc[i] > df["MA20"].iloc[i]) and (df["MA10"].iloc[i -1] < df["MA20"].iloc[i]):
        Buy.append(i)
        if df["Date Copy"].iloc[i] not in buy:
            buy[df["Date Copy"].iloc[i]] = [df["Forecast"].iloc[i] * -1]
        else:
            buy[df["Date Copy"].iloc[i]].append(df["Forecast"].iloc[i] * -1)
    # If the shorter MA is lower than the longer MA; and the opposite the previous day - sell
    elif (df["MA10"].iloc[i] < df["MA20"].iloc[i]) and (df["MA10"].iloc[i - 1] > df["MA20"].iloc[i - 1]):
        Sell.append(i)
        if df["Date Copy"].iloc[i] not in sell:
            sell[df["Date Copy"].iloc[i]] = [df["Forecast"].iloc[i]]
        else:
            sell[df["Date Copy"].iloc[i]].append(df["Forecast"].iloc[i])

# Initialize Variables
dates = list(buy.keys()) + list(sell.keys())
dates.sort()

# Starting capital of $2000
capital = 2000
stocks = 0

# Calculate profit for each day we buy/sell, and store
for date in dates:
    if date in buy.keys():
        # We only buy 1 stock if we have enough money to buy 1 stock
        if sum(buy[date]) < capital:
            capital += sum(buy[date])
            stocks += 1

    if date in sell.keys():
        # We sell if we have at least 1 stock
        if stocks > 0:
            capital += sum(sell[date])
            stocks -= 1

print("Cash capital at 2021-12-22 is: $%.2f with %d stocks holding" % (capital, stocks))
equity = capital + stocks * df["Forecast"].iloc[-1]
print("Equity: $%.2f" % equity)

# Graph 
plt.figure(figsize=(60,25))
plt.plot(df["Close"], label = "Stock Price", c = "blue", alpha = 0.5)
plt.plot(df["Forecast"], label = "Forecast", c = "red", alpha = 0.5)
plt.plot(df["MA20"], label = "MA20", c = "k", alpha = 0.9)
plt.plot(df["MA10"], label = "MA10", c = "magenta", alpha = 0.9)
plt.scatter(df.iloc[Buy].index, df.iloc[Buy]["Forecast"], marker = "^", color = "g", s = 200)
plt.scatter(df.iloc[Sell].index, df.iloc[Sell]["Forecast"], marker = "v", color = "r", s = 200)
plt.legend()
plt.show()
