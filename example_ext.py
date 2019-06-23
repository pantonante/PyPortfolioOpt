import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
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

returns = df.pct_change().dropna(how="all")
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)
ef = EfficientFrontier(mu, S, asset_classes=assets_classes, asset_allocation=assets_allocation)
weights = ef.max_sharpe()
ef.portfolio_performance(verbose=True)
print(weights)


df = df[['AAPL','BAC','GE']]
assets_classes = {
  "AAPL":"Tech",
  "GE":"Consumer",
  "BAC":"Financial",
}
assets_allocation = {
  "Tech": (.30, .30),
  "Consumer": (.50, .50),
  "Financial": (.20, .20),
}

returns = df.pct_change().dropna(how="all")
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)
ef = EfficientFrontier(mu, S, asset_classes=assets_classes, asset_allocation=assets_allocation)
weights = ef.max_sharpe()
ef.portfolio_performance(verbose=True)
print(weights)
