# main.py
from data.polygon_data import PolygonClient
from analysis import prep_pair, rolling_return_correlation

def main():
    client = PolygonClient()
    # Pull aligned closes for two ETFs
    df = client.get_aligned_etf_closes("QQQ", "SPY", start="2023-01-01", end=None)

    # Step 1-2 in one shot
    log_prices, log_returns, corr120 = prep_pair(df, lookback=120)
    print("Trailing 120-day return correlation:", round(corr120, 4))

    # Optional: rolling correlation series (for plotting later)
    roll = rolling_return_correlation(log_returns, lookback=120)
    print(roll.tail())

if __name__ == "__main__":
    main()
