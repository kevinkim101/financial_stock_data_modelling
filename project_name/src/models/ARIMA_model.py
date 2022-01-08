import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from pandas.plotting import lag_plot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import datetime

df = pd.read_csv("project_name/data/processed/appl_stock_values.csv")

#split the data
df["Date"] = pd.to_datetime(df["Date"])
df = df.drop(columns=["Open","Volume", "High", "Low", "Unnamed: 0"])
test = df[df["Date"] > datetime.datetime(2021, 1, 1)]
train = df[df["Date"] <= datetime.datetime(2021, 1, 1)]
lag_1_pred = []
date = []

test = test.reset_index(drop=True)

for i in range(len(test)):
    model = ARIMA(train["Close"], order = (1,1,1))
    model_fit = model.fit()
    lag_1_pred.append(model_fit.forecast().iloc[0])
    train = train.append({"Close": test.iloc[i]["Close"], "Date":test.iloc[i]["Date"]}, ignore_index=True)

model = ARIMA(train["Close"], order = (1,0,1))
model_fit = model.fit()

# Extract data from after January 1st, 2021
df = df[df["Date"] > datetime.datetime(2021, 1, 1)]
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

dates = list(buy.keys()) + list(sell.keys())
dates.sort()

capital = 2000
stocks = 0

# Calculate profit for each day we buy/sell, and store
for date in dates:
    if date in buy.keys():
        if sum(buy[date]) < capital:
            capital += sum(buy[date])
            start_sell = True
            stocks += 1

    if date in sell.keys():
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
