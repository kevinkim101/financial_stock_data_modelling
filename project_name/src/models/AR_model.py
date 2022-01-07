import pandas as pd
import datetime
import numpy as np
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

# First we should check the Autocorrelation of APPL Stock since we are using a regression model
df = pd.read_csv("project_name/data/processed/appl_stock_values.csv")

# Split the data
df["Date"] = pd.to_datetime(df["Date"])
df = df.drop(columns=["Open","Volume", "High", "Low", "Unnamed: 0"])
test = df[df["Date"] > datetime.datetime(2021, 6, 22)]
train = df[df["Date"] <= datetime.datetime(2021, 6, 22)]
lag_1_pred = []

test = test.reset_index(drop=True)

# Test and train using AR
for i in range(len(test)):
    model = AutoReg(train["Close"], lags = 1, old_names=False)
    model_fit = model.fit()
    lag_1_pred.append(model_fit.predict(start=len(train), end=len(train)).iloc[0])
    close_val = test.iloc[i]["Close"]
    train = train.append({"Close": close_val}, ignore_index=True)

# Extract forecasted data from after June 22nd, 2021
df = df[df["Date"] > datetime.datetime(2021, 6, 22)]
np_lag_1_pred = np.array(lag_1_pred)
df["Forecast"] = np_lag_1_pred.tolist()
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
        if df["Date"].iloc[i] not in buy:
            buy[df["Date"].iloc[i]] = [df["Forecast"].iloc[i] * -1]
        else:
            buy[df["Date"].iloc[i]].append(df["Forecast"].iloc[i] * -1)
    
    # If the shorter MA is lower than the longer MA; and the opposite the previous day - sell
    elif (df["MA10"].iloc[i] < df["MA20"].iloc[i]) and (df["MA10"].iloc[i - 1] > df["MA20"].iloc[i - 1]):
        Sell.append(i)
        if df["Date"].iloc[i] not in sell:
            sell[df["Date"].iloc[i]] = [df["Forecast"].iloc[i]]
        else:
            sell[df["Date"].iloc[i]].append(df["Forecast"].iloc[i])

profit = 0
profits_data = {}
dates = list(buy.keys()) + list(sell.keys())
dates.sort()

capital = 500
stocks = 0

# Calculate profit for each day we buy/sell, and store
for date in dates:
    if date in buy.keys():
        # Buy a single stock if we have enough capital
        if sum(buy[date]) < capital:
            profit += sum(buy[date])
            capital += sum(buy[date])
            start_sell = True
            stocks += 1

    if date in sell.keys():
        # Sell a single stock if we have at least one stock
        if stocks > 0:
            profit += sum(sell[date])
            capital += sum(sell[date])
            stocks -= 1
    
    # Store the profit data after each transaction
    profits_data[date] = profit

# Find the maximum profit and day when maximum profit is attained
max_profit = max(profits_data.values())
for key in profits_data.keys():
    if profits_data[key] == max_profit:
        the_date = key

print("The Max Profit is: $%.2f on %s" % (max_profit, the_date))
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


