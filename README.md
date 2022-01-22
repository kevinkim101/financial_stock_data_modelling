Stock and Securities Data Modeling
==============================

The following is a passion project of two friends who share a mutual interest in Fintech. We focus on the applications of various statistical and deep learning models and how well they work on predicting the price of stocks.

**The project is defined into 3 main parts:**
1) The notebooks, some rough work and notes done during the planning and prototyping stages of the project.
2) The models, this entails the code to generate the models, not the actual models themselves.
3) The visualization and analysis of the models and their performances.


The data used is courtesy of: https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs


Project Organization
------------

    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └|


--------


Purpose + What we Learned
-------------------------
**Brian:** During my first co-op as an applied machine learning intern, I learned that none of the work of ML is in the making of the model, all of it lies in the analysis and application. I had studied all of the theory and the math behind the models but always fell victim to analysis paralysis when handed a solo project. The purpose of this project is not to show off technicals, but to work on analysis, baselining and proof of concept. 

**Kevin:** The most significant aspect of a data forecasting project lies in the creative works and analyses that can be performed. Implementing ML models to forecast stock data was not the biggest concern, but rather, it was a question of what we can do with that information. The purpose of this project is to implement and test several time-series forecasting models to predict future stock prices of major corporations, and to assess the performance of such predictions. 


Requirements
-------------
- Download [git](https://git-scm.com/downloads) or [GitHub Desktop](https://desktop.github.com/)
- Install [Python3](https://www.python.org/downloads/) 
- Install pip if not already automatically installed
  1. Download the [pip file](https://bootstrap.pypa.io/get-pip.py) and store it in same directory as Python installation
  2. cd to the folder and run ```$ python3 get-pip.py```
- Install Numpy, Pandas and Matplotlib using pip install


How to use
----------- 
**If you want to backtest our models:**
1. Fork this repository and clone it using GitHub Desktop, or run ```git clone https://github.com/thekioskman/Finacial-Data-Modeling.git``` using Terminal/CMD, or download as ZIP. 
2. Navigate to the folder and install Pandas and Matplotlib using ```pip install pandas``` and ```pip install matplotlib```
3. Download data from the Kaggle website

**If you want to see the analysis:**

This is really the bread and butter of this project, you do not need to do anything fancy to look at our analysis. Everything is located in the visualization folder.


# Models
The models that tend to perform best in time series forecasting, especially for financial data, are linear models. Financial data is noisy and often filled with dependent data, thus complex models can easily overfit to that noise and perform poorly in practice. 

Autoregression
---
Autoregression is just a fancy term for a linear regression which uses the same variable at different time positions as the independent and dependent variable. An example would be as follows: 

The price of a stock at time t depends on its price at t-1 and t-2. Notice that it's the same variable (stock price) at different times acting as both the dependent and independent variable.

<img src="https://render.githubusercontent.com/render/math?math=Y_t = \delta %2B \phi_1Y_{t-1} %2B \phi_2Y_{t-2}... %2B \phi_pY_{t-p} %2B A">

You can clearly see that there are n independent variables, and a single dependent (or predicated) value. Our goal is to use the price values at previous time positions to predict the value at the next time interval.

Moving Average
---
Moving Average is a simple method that takes the mean value of a range of numbers in a dynamic manner. Given mass data such as stocks that are spread over a long window of time, the moving average takes a specific smaller time frame and calculates the average over such a time period. This is a useful tool used to identify long term trends in data which can ultimately be used to forecast data. As explained in the next section for ARIMA, the Moving Average term also represents the observational error and error lag. A critical application of the Moving Average model was the Simple Moving Average method where a time frame of 10 and 20 days were selected to apply the Golden Cross Rule - which identifies potential buy and sell signals of stocks.

ARIMA
---
ARIMA stands for Autoregressive Integrated Moving Average which ultimately integrates the AR and MA models to identify dependencies of variables. Overall, it can be seen as a regression model that incorporates both lag from previous data points (i.e. t-1, t-2) and from error. The ARIMA model has three parameters: p, d, and q. The p value is obtained from the lag related to the AR component, and q is the error lag from the MA component. d is a differencing parameter that produces stationary data points necessary for identifying such lag values parameters.

# Rolling Average Buy & Sell Signals
This trading strategy incorporates the Golden Cross Rule which ultimately selects two Simple Moving Averages (MA) or Rolling Averages (RA) of different time frames and identifies bullish or bearish trends. In our case, we selected a 10-day RA and a 20-day RA. If the 10-day RA crosses above the 20-day RA, it indicates an upward trend - thus a buy singla. If the 10-day RA crosses below the 20-day RA, it indicates a downward trend - thus a sell signal. Then, in order to find the efficiency/accuracy of each model, we identified buy/sell signals of AAPL stocks based on the forecasted data from each model.

AR
![Roll_AR](project_name/reports/figures/Roll_AR.png?raw=true "Rolling Avg. with AR")

ARIMA
![Roll_ARIMA](project_name/reports/figures/Roll_ARIMA.png?raw=true "Rolling Avg. with ARIMA")

MA
![Roll_MA](project_name/reports/figures/Roll_MA.png?raw=true "Rolling Avg. with MA")

In our analysis, we trained the data from 1984-09-07 to 2021-01-04 and forecasted data from 2021-01-04 to 2021-12-22. Assuming an initial capital of $2000 and assuming we transact 1 stock at each buy/sell signal, we found the net profit after each transaction and calculated the equity (cash capital and value of held stock) on 2021-12-22. From this simple analysis, the net equity on 2021-12-22 based on the forecasted values from all 3 models are as follows:

AR → $2286.95 ($730.03 with 9 stocks holding)
MA → $2033.91 ($457.88 with 9 stocks holding)
ARIMA → $2286.69 ($729.84 with 9 stocks holding)

Evidently, AR and ARIMA produced the highest equity and are very similar in terms of the equity value calculated. Interestingly, this means that the MA terms are almost negligible. Out of curiosity, we tested stock data from different corporations to see if this was the case (see AR_vs_ARIMA_test.ipynb). As seen in the notebook, the values obtained from running the AR and ARIMA model are quite similar. This shows evidence that models involving linear regression are the most significant for trading.

# Tradings Strategies with Autoregression

So after conducting various tests on our linear models, namely between AR and ARIMA, ARIMA of course just being a buffed version of Autoregression, we came to the conclusion of using AR as our main model. All the models do is forecast the future prices. This ability alone is useless unless you are able to build a profitable trading strategy around it. 

You may be a bit confused because from the looks of the modelled graph and the error values they seem to predict price 100% perfectly!? Why aren't people using these strategies to make millions? Also it can't be that hard to create a trading strategy if you can “SEE INTO THE FUTURE”. 

Well if it really were that simple I would not be sitting here and writing this. (and I certainly would not have gotten my Citadel application rejected 5 times). The main issue with regression models is their limited look ahead period. We are working with daily data (since we are poor), and you can see from the PACF plots that our models rely only on the previous day's price. This also means that our AR model can only reasonably predict one day into the future. 


![wacky line graph](project_name/reports/figures/wacky_line_graph.png?raw=true "1 day + 1 + n day prediction line graph")


From this wacky graph you can see that between the first forecasted value, and the last known value, there is a significant difference. However, for the next 1+n values you can see that the AR model just persists the previous value, essentially giving us a straight line.  

![1 day predict graph](project_name/reports/figures/one_day_forcast.png?raw=true "1 day forcast graph")

In the case that this first graph was too confusing, I produced a second comparison. Looking at these two graphs was actually how I realized that something weird was going on. You can see that if we increase the " consecutive days to forecast value" to 5 or 6, we start to see these drawn out portions of straight lines. This means that our model is acting more like a persistence model (just repeating the previous value) when we ask it to predict more than 1 day into the future. This is due to the fact the stock data is very close to random walk. 


![6 day predict graph](project_name/reports/figures/six_day_forcast.png?raw=true "6 day forcast graph")

Therefore, we deduced that the only  prediction that has weighting is the immediate forecast (the first prediction). Of course this means we can only see one day ahead into the future, and must build our trading strategies with that in mind. 
 

Why Autoregression
---
We decided that AR (Autoregression) was the better choice for our trading purposes as it provides similar results to ARIMA but trains in exponentially faster time. Moreover, as depicted in the PACF plots, the only independent variable in our regression is the price at time lag 1. Or in other words, the only value that affects our predicted price for tomorrow, is today's price. Our regression function comes out to a simple y = mx + b… ARIMA is a little overkill for forecasting purposes in this case.


![Random stock PACF graphs](project_name/reports/figures/RandomGraphs.png?raw=true "Random Sample PACF + Lag correlation plots")


Slope Trend Strategy
---
This is a little strategy that I made up for demonstration purposes. This strategy is not backtested and would likely perform fairly poorly in practice. Broadly this strategy could be classified under mean reversion as it sells when the price is up, and buys when the price is down. 

The strategy uses a time window of the three slopes of t-2 to t-1 and t-1 to t and t to t+1 
if slope(t,t+1) > slope(t-1,t) > slope(t-2,t-1) we are in a strong uptrend
if slope(t,t+1) < slope(t-1,t) < slope(t-2,t-1) we are in a strong downtrend
if other case then we are in a period of consolidation so we do nothing


So if the price increased the day before, and the price increased today, and we predict the price is going to increase tomorrow, then we should sell. On the other hand, if the price went down the day before, went down today, and we predict it will go down tomorrow, we should buy.

Buy Sell Signals
---
![3 slope buy and sell](project_name/reports/figures/3_slope_buy_sell.png?raw=true "3 slope buy sell signals")


We can see that the simple returns of this strategy were not bad during the year of 2021. Surprisingly around 50%. However, we should always look at the performance of the base instrument, and compare the strategy to a simple buy and hold. 

Equity Curve
---
![3 slope equity curve](project_name/reports/figures/3_slope_equity.png?raw=true "3 slope equity curve")


As you can see the 3 slope windows actually does outperform the buy and hold strategy without factoring transaction costs. 

Forecasted Moving Average Strategy
---
This strategy incorporates a dynamic Moving Average method in which we use the average value for a 19-day period to produce a 20-day Simple Moving Average (SMA). We do the same with a 9-day period to produce a 10-day SMA. Then, we implement the Golden Cross Rule once again and see if the 10-day SMA is above the 20-day SMA - which shows a buy signal. The opposite shows a sell signal. 

Buy Sell Signals
---
![SMA Buy&Sell](project_name/reports/figures/SMA_Buy_Sell.png?raw=true "SMA Buy and Sell Signals")

Equity Curve
---
![SMA Equity](project_name/reports/figures/SMA_equity_new.png?raw=true "SMA Equity")


Simple Predict Next Price Strategy
---
The final AR based strategy is a simple price predictions strategy. The motivation is as follows: if the predicted price tomorrow is higher than today's price, then we should buy today and sell tomorrow, on the other hand if the predicted price is lower than today's price, then we should sell today, and buy tomorrow. Although this general concept makes sense, there are a couple kinks in the strategy, after all.. It is a prototype and should be treated as a fun experiment.


Here we see that the strategy is pretty high frequency, close to daily buys and sells. The main issue with this strategy is that it does not take advantage of trending. The green and red circle represent attempted buy and sells. Where, in the case of an attempted buy, we had no cash left and in the case of an attempted sell, we had no holdings of the stock left.


Buy Sell Signals
---
![simple_predict](project_name/reports/figures/simple_predict_buy_sell.png?raw=true "simple_predict")


You can see that we sell our stocks as soon as an uptrend is starting to form, or at least at the very beginning of the uptrend and thus lose out on a lot of profits. A good first change would be to introduce a longer holding period, and a lower sell percentage. So we can take advantage of the trends better. 

Equity Curve
---
![simple_predict_equity](project_name/reports/figures/simple_predict_equity.png?raw=true "simple_predict_equity")


An investigation of the equity curve proves to be interesting. We can see that it appears to be stationary, eventually stabilizing at the mean return of ~2400, slightly worse than the buy and hold approach.

