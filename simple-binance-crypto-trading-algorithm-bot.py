# crypto trading bot code by bitopsy.com published APRIL 1 2024

# 1. add you API keys from binance
# 2. choose any pair of cryptos or stocks
# 3. uncomment the buy and sell calls when you are ready to trade otherwise it will only simulate
# 4. (optional) change the parameters to taste
# 5. use responsibly. We are not financial advisors and we are not resbonsible for any loss


# Replace with your Binance API key and secret
api_key = #your public api here from binance
api_secret = #your secret api here from binance


from binance.client import Client
import ta
import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Initialize Binance client
client = Client(api_key, api_secret)

# Trading pair and parameters
import sys
trading_pair = 'SOLUSDT' #change this to whatever crypto or stock you are interested in trading
btc_pair = 'BTCUSDT'

ass = trading_pair[:-4]

bb_window = 21
bb_std_dev = 2
macd_slow = 26
macd_fast = 12
macd_signal = 9
rsi_period = 14
rsi_overbought = 70
rsi_oversold = 30
rf_window = 60  # Window size for Random Forest



# Function to get the current price
def get_current_price(symbol):
    ticker = client.get_ticker(symbol=symbol)
    return float(ticker['lastPrice'])

# Function to place a buy order
def buy(symbol, quantity):
    order = client.order_market_buy(symbol=symbol, quantity=quantity)
    print(f"Buy order placed for {symbol}: {order}")

# Function to place a sell order
def sell(symbol, quantity):
    order = client.order_market_sell(symbol=symbol, quantity=quantity)
    print(f"Sell order placed for {symbol}: {order}")

# Main trading loop
while True:
    # Get historical OHLCV data
    klines = client.get_historical_klines(symbol=trading_pair, interval=Client.KLINE_INTERVAL_1MINUTE)
    btc_klines = client.get_historical_klines(symbol=btc_pair, interval=Client.KLINE_INTERVAL_1MINUTE)


    # Get current price
    current_price = get_current_price(trading_pair)

    # Calculate RSI
    closes = [float(kline[4]) for kline in klines]
    btc_closes = [float(kline[4]) for kline in btc_klines]

    

    closes_series = pd.Series(closes)  # Convert list to pandas Series
    btc_closes_series = pd.Series(btc_closes)

    rsi_values = ta.momentum.rsi(closes_series, window=rsi_period)
    current_rsi = rsi_values.iloc[-1]  # Get the last RSI value
    print("## RSI current price and RSIi/overbought", current_price, rsi_oversold,"<",current_rsi,"<",rsi_overbought)

    # Calculate Bollinger Bands
    bb_indicator = ta.volatility.BollingerBands(closes_series, window=bb_window, window_dev=bb_std_dev)
    bb_lower = bb_indicator.bollinger_lband().iloc[-1]
    bb_upper = bb_indicator.bollinger_hband().iloc[-1]

    # Calculate MACD
    macd = ta.trend.MACD(closes_series)
    macd_line = macd.macd().iloc[-1]
    macd_signal_line = macd.macd_signal().iloc[-1]


    # Prepare data for Random Forest
    X = []
    y = []
    for i in range(rf_window, len(closes)):
        X.append(list(closes[i - rf_window:i])+ list(btc_closes[i - rf_window:i]))
        y.append(closes[i])

    X = np.array(X)  # Convert list to NumPy array

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predict the next price using Random Forest
    next_price_pred = rf_model.predict([X_test[-1]])
    print("## RF current price and predicted price", current_price, next_price_pred)




    # Check trading conditions
    if current_rsi > rsi_overbought:
        # Sell condition
        balance = client.get_asset_balance(asset=ass)
        sell_quantity = float(balance['free'])
        if sell_quantity > 0:
            print("\nRSI sell", sell_quantity )
            #sell(trading_pair, sell_quantity)
    elif current_rsi < rsi_oversold:
        # Buy condition
        usdt_balance = client.get_asset_balance(asset='USDT')
        buy_amount = float(usdt_balance['free'])
        if buy_amount > 0:
            buy_quantity = buy_amount / current_price
            print("\nRSI buy amount",buy_amount, "price", current_price, "total:",(buy_amount / current_price))
            #buy(trading_pair, buy_quantity)
    print("## BOL",bb_upper,"<",current_price,"<", bb_lower)
    print("## MACD MACD_line:", macd_line, ">" ,macd_signal_line)
    # Check trading conditions
    if current_price < bb_lower and macd_line > macd_signal_line:
        # Buy condition
        usdt_balance = client.get_asset_balance(asset='USDT')
        buy_amount = float(usdt_balance['free'])
        if buy_amount > 0:
            buy_quantity = buy_amount / current_price
            print("\nMACD/BOL buy amount",buy_amount, "price", current_price, "total:",(buy_amount / current_price))
            #buy(trading_pair, buy_quantity)
    elif current_price > bb_upper and macd_line < macd_signal_line:
        # Sell condition
        balance = client.get_asset_balance(asset=ass)
        sell_quantity = float(balance['free'])
        if sell_quantity > 0:
            #sell(trading_pair, sell_quantity)
            print("\nMACD/BOL sell", sell_quantity )

    print("---------------------")
    # Wait for the next iteration
    time.sleep(60)  # Sleep for 1 minute
