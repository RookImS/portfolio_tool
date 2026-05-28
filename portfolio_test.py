import datetime as dt
import pandas as pd
from portfolio import Portfolio

def portfolio_backtest_by_duration(portfolio:Portfolio, benchmark=None, duration=1, start_date=None):
    if not start_date:
        start_date = portfolio.start_date
    start_date_limit = dt.date.today() - dt.timedelta(days=365 * duration)

    stats = pd.DataFrame(columns=['cagr', 'stdev', 'mdd', 'beta', 'alpha'])
    date = start_date
    while date <= start_date_limit:
        _, stat = portfolio.backtest(start_date=date, duration=duration, rebalancing_cycle=5, benchmark=benchmark)
        stats = pd.concat([stats, stat.T])
        date = date + dt.timedelta(days=365)
    
    stats.reset_index(inplace=True)
    stats.rename(columns={'index': 'ratio'}, inplace=True)

    ratio_str = ""
    for ticker, weight in portfolio.target_ratio.items():
        ratio_str += f"{weight:.1f}:"
    ratio_str = ratio_str.rstrip(':')
    stats.loc[:, 'ratio'] = ratio_str

    return stats