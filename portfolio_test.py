import datetime as dt
import pandas as pd
from portfolio import Portfolio

def make_benchmark_data(benchmark_portfolio:Portfolio, start_date:dt.date, duration:int=1, cash_flow:float=700):
    start_date_limit = benchmark_portfolio.price_data.index[-1] - dt.timedelta(days=365 * duration)
    date = start_date
    benchmark_data = []
    while date <= start_date_limit:
        history, _ = benchmark_portfolio.backtest(start_date=date, duration=duration, rebalancing_cycle=5, cash_flow=cash_flow)
        benchmark_data.append(history[('total', 'value')])
        date = date + dt.timedelta(days=365)
    
    return benchmark_data


def portfolio_backtest_by_duration(portfolio:Portfolio, benchmark_data:list=None, start_date:dt.date=None, duration:int=1, cash_flow:float=700):
    if not start_date:
        start_date = portfolio.start_date
    start_date_limit = portfolio.price_data.index[-1] - dt.timedelta(days=365 * duration)

    stats = pd.DataFrame(columns=['cagr', 'stdev', 'mdd', 'beta', 'alpha'])
    date = start_date
    data_idx = 0
    while date <= start_date_limit:
        benchmark = None
        if benchmark_data:
            benchmark = benchmark_data[data_idx]
            data_idx += 1

        _, stat = portfolio.backtest(start_date=date, duration=duration, rebalancing_cycle=5, cash_flow=cash_flow, benchmark=benchmark)
        stats = pd.concat([stats, stat.T])
        date = date + dt.timedelta(days=365)
    
    stats.reset_index(inplace=True)
    stats.rename(columns={'index': 'ratio'}, inplace=True)

    ratio_str = ""
    for ticker, weight in portfolio.target_ratio.items():
        ratio_str += f"{weight:04.1f}:"
    ratio_str = ratio_str.rstrip(':')
    stats.loc[:, 'ratio'] = ratio_str

    return stats