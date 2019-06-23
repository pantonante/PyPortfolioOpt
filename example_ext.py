import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.hierarchical_risk_parity import hrp_portfolio
from pypfopt.value_at_risk import CVAROpt
from pypfopt import discrete_allocation

# Reading in the data; preparing expected returns and a risk model
df = pd.read_csv("tests/stock_prices.csv", parse_dates=True, index_col="date")
assets_classes = {
  "GOOG":"Tech",
  "AAPL":"Tech",
  "FB":"Tech",
  "BABA":"Tech",
  "AMZN":"Tech",
  "GE":"Consumer",
  "AMD":"Tech",
  "WMT":"Consumer",
  "BAC":"Financial",
  "GM":"Consumer",
  "T":"Consumer",
  "UAA":"Consumer",
  "SHLD":"Consumer",
  "XOM":"Energy",
  "RRC":"Energy",
  "BBY":"Consumer",
  "MA":"Financial",
  "PFE":"Healthcare",
  "JPM":"Financial",
  "SBUX":"Consumer"
}
assets_allocation = {
  "Tech": (.10, .30),
  "Consumer": (.10, .50),
  "Financial": (.10, .20),
  "Energy": (0.05, .10),
  "Healthcare": (0.05, .10)
}
assert(all([tick in df.columns for tick in assets_classes]))

# tickers = list(df.columns)
# # asset_allocation_idx = {}
# weight_classes = list()
# for asset_cat in assets_allocation:
#   cat_tickers = [k for k,v in assets_classes.items() if v == asset_cat]  
#   # asset_allocation_idx[asset_cat] = [tickers.index(t) for t in cat_tickers]
#   weight_classes.append(([tickers.index(t) for t in cat_tickers], assets_allocation[asset_cat]))

# # print(asset_allocation_idx)
# # print(weight_classes)

# for cat in weight_classes:
#   idx = cat[0]
#   val_min = cat[1][0]
#   val_max = cat[1][1]
#   print("[{}] between {} and {}".format(', '.join(map(str, idx)), val_min,val_max))


returns = df.pct_change().dropna(how="all")
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)
ef = EfficientFrontier(mu, S) #, asset_classes=assets_classes, asset_allocation=assets_allocation
weights = ef.max_sharpe()
ef.portfolio_performance(verbose=True)
print(weights)
