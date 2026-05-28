import itertools
import datetime as dt
import pandas as pd
import numpy as np

class Portfolio:
    # stock_data
    def __init__(self, stock_data: pd.DataFrame, start_cash: float, target_ratio: dict = None, buy_ratio: float = 5.0, sell_ratio: float = 5.0):
        self.__stock_list = set(stock_data.columns.get_level_values(0))
        if target_ratio is None:
            target_ratio = {ticker: round(100 / len(self.__stock_list), 2) for ticker in self.__stock_list}
        self.__stock_list = list(self.__stock_list.intersection(target_ratio.keys()))
        target_ratio_list = list(target_ratio)
        self.__stock_list.sort(key=lambda x: target_ratio_list.index(x))
        stock_data = stock_data.loc[:, self.__stock_list]
        self.__price_data = stock_data.loc[:, (stock_data.columns.get_level_values(1) == 'price')].copy()
        self.__price_data = self.__price_data.droplevel(1, axis=1)
        self.__adj_price_data = stock_data.loc[:, (stock_data.columns.get_level_values(1) == 'adj_price')].copy()
        self.__adj_price_data = self.__adj_price_data.droplevel(1, axis=1)
        self.__divend_data = stock_data.loc[:, (stock_data.columns.get_level_values(1) == 'divend')].copy()
        self.__divend_data = self.__divend_data.droplevel(1, axis=1)
        self.__start_cash = start_cash
        # target_ratio의 구성 { stock명 : 목표 비중, ... }
        self.target_ratio = target_ratio
        self.buy_ratio = buy_ratio
        self.sell_ratio = sell_ratio
        self.__target_buy_ratio = {ticker: target_ratio[ticker] * buy_ratio / 100 for ticker in self.stock_list}
        self.__target_sell_ratio = {ticker: target_ratio[ticker] * sell_ratio / 100 for ticker in self.stock_list}
    

        self.__price_data.dropna(inplace=True)
        self.__avilable_date = self.price_data.iloc[0].name
        self.__adj_price_data = self.__adj_price_data[self.__adj_price_data.index >= self.__avilable_date]
        self.__divend_data = self.__divend_data[self.__divend_data.index >= self.__avilable_date]
        
    @property
    def stock_list(self) -> list:
        return self.__stock_list
    @property
    def price_data(self) -> pd.DataFrame:
        return self.__price_data
    @property
    def adj_price_data(self) -> pd.DataFrame:
        return self.__adj_price_data
    @property
    def divend_data(self) -> pd.DataFrame:
        return self.__divend_data
    @property
    def start_date(self) -> dt.date:
        return self.__avilable_date

    def backtest(self, start_date=None, duration=None, rebalancing_cycle=None, benchmark=None):
        info_list =  ['price', 'number', 'value', 'weight']
        stock_info =  list(itertools.product(self.stock_list, info_list))

        col = [('total', 'value'), ('cash', 'value'), ('cash', 'weight')] + stock_info
        col = pd.MultiIndex.from_tuples(col)

        dates = self.__available_dates(start_date, duration)
        rebalancing_dates = dates[::rebalancing_cycle] if rebalancing_cycle else dates
        stat = pd.DataFrame(columns=col, index=dates)

        # 첫 값 설정
        first_date = dates[0]
        total_value = self.__start_cash
        for ticker in self.stock_list:
            stat.loc[first_date, ('total', 'value')] = total_value
            stat.loc[first_date, ('cash', 'value')] = 0
            stat.loc[first_date, ('cash', 'weight')] = 0
            stat.loc[:, (ticker, 'price')] = self.price_data[ticker]
            stat.loc[first_date, (stat.columns.get_level_values(0) == ticker) & (stat.columns.get_level_values(1) != 'price')] = self.__ideal_nvw(total_value, ticker, first_date)
            
        for i in range(1, len(dates)):
            cash_value = stat.loc[dates[i - 1]][('cash', 'value')]
            total_value = cash_value
            # 가격 변동 처리
            prev_num = stat.loc[dates[i - 1]][:, 'number']
            prev_num_pv = prev_num * self.price_data.loc[dates[i]]
            total_value += prev_num_pv.sum()
            # 배당금 처리
            divend_df = self.acculate_divend(dates[i-1], dates[i])
            divend = (divend_df.sum() * prev_num).sum()
            cash_value += divend
            total_value += divend
            stat.loc[dates[i], ('cash', 'value')] = cash_value
            stat.loc[dates[i], ('total', 'value')] = total_value

            # 변동에 따른 리밸런싱 계산
            prev_num_ratio = prev_num_pv / total_value * 100
            target_ratio_sr = pd.Series(self.target_ratio)
            ratio_diff = (prev_num_ratio - target_ratio_sr)

            need_sell = ratio_diff[ratio_diff >= pd.Series(self.__target_sell_ratio)].index.to_list()
            need_buy = ratio_diff[ratio_diff <= -pd.Series(self.__target_buy_ratio)].index.to_list()
            need_trade = need_sell + need_buy
            for ticker in self.stock_list:
                if dates[i] in rebalancing_dates and ticker in need_trade:
                    stat.loc[dates[i], (stat.columns.get_level_values(0) == ticker) & (stat.columns.get_level_values(1) != 'price')] = self.__ideal_nvw(total_value, ticker, dates[i])
                    stat.loc[dates[i], ('cash', 'value')] += ratio_diff[ticker] * total_value * 0.01
                else:
                    stat.loc[dates[i], (ticker, 'number')] = stat.loc[dates[i - 1], (ticker, 'number')]
                    stat.loc[dates[i], (ticker, 'value')] = prev_num_pv[ticker]
                    stat.loc[dates[i], (ticker, 'weight')] = prev_num_ratio[ticker]
            
            stat.loc[dates[i], ('cash', 'weight')] = stat.loc[dates[i]][('cash', 'value')] / total_value * 100

        # stat.insert(0, ('rebalancing', 'flag'), False)
        # stat.loc[rebalancing_dates, ('rebalancing', 'flag')] = True

        res_df = pd.DataFrame(columns=['portfolio'], index=['cagr', 'stdev', 'mdd', 'beta', 'alpha'])
        start_price = stat.loc[dates[0], ('total', 'value')]
        end_price = stat.loc[dates[-1], ('total', 'value')]
        res_df.loc['cagr', 'portfolio'] = ((end_price / start_price) ** (252 / len(dates)) - 1) * 100
        res_df.loc['stdev', 'portfolio'] = stat.loc[:, ('total', 'value')].pct_change().std() * np.sqrt(252) * 100
        peak_value = stat[('total', 'value')].cummax()
        drawdown = (stat[('total', 'value')] - peak_value) / peak_value
        res_df.loc['mdd', 'portfolio'] = -1 * drawdown.min() * 100
        if benchmark is not None:
            alpha, beta = Portfolio.alphabeta(benchmark, stat.loc[:, ('total', 'value')], duration=duration)
            res_df.loc['alpha', 'portfolio'] = alpha
            res_df.loc['beta', 'portfolio'] = beta

        return stat, res_df

    # 범위는 (start_date, end_date]
    def acculate_divend(self, start_date, end_date):
        divend_df = self.divend_data[(self.divend_data.index > start_date) & (self.divend_data.index <= end_date)]
        return divend_df
    
    def stock_stdev(self, tickers=[], start_date=None, duration=None):
        if not tickers:
            tickers = self.stock_list

        data_df = self.adj_price_data.loc[self.__available_dates(start_date, duration), tickers]

        std_df = pd.DataFrame(columns=tickers, index=['stdev'])
        for ticker in tickers:
            std_df.loc['stdev', ticker] = data_df[ticker].pct_change().std() * np.sqrt(252)

        return std_df * 100

    def stock_cagr(self, tickers=[], start_date=None, duration=None):
        if not tickers:
            tickers = self.stock_list

        data_df = self.adj_price_data.loc[self.__available_dates(start_date, duration), tickers]
        
        cagr_df = pd.DataFrame(columns=tickers, index=['cagr'])
        for ticker in tickers:
            start_price = data_df[ticker].iloc[0]
            end_price = data_df[ticker].iloc[-1]
            cagr_df.loc['cagr', ticker] = (end_price / start_price) ** (252 / len(data_df.index)) - 1

        return cagr_df * 100
    
    def stock_mdd(self, tickers=[], start_date=None, duration=None):
        if not tickers:
            tickers = self.stock_list

        data_df = self.adj_price_data.loc[self.__available_dates(start_date, duration), tickers]
        
        mdd_df = pd.DataFrame(columns=tickers, index=['mdd'])
        for ticker in tickers:
            price_sr = data_df[ticker]
            peak_price = price_sr.cummax()
            drawdown = (price_sr - peak_price) / peak_price
            mdd_df.loc['mdd', ticker] = drawdown.min()

        return -1 * (mdd_df * 100)
    
    def stock_corr(self, tickers=[], start_date=None, duration=None):
        if not tickers:
            tickers = self.stock_list

        data_df = self.adj_price_data.loc[self.__available_dates(start_date, duration), tickers]
        ret_df = data_df.pct_change()
        corr_df = ret_df.corr()

        return corr_df

    def stock_alphabeta(self, market_value, tickers=[], start_date=None, duration=None, risk_free_rate=0.02):
        if not tickers:
            tickers = self.stock_list

        market_value = market_value.loc[self.__available_dates(start_date, duration)]

        alphabeta_df = pd.DataFrame(columns=tickers, index=['alpha', 'beta'])
        for ticker in tickers:
            target_value = self.adj_price_data.loc[self.__available_dates(start_date, duration), ticker]
            alpha, beta = Portfolio.alphabeta(market_value, target_value)
            alphabeta_df.loc['alpha', ticker] = alpha
            alphabeta_df.loc['beta', ticker] = beta

        return alphabeta_df
    
    def __ideal_nvw(self, total_value, ticker, date):
        target_ratio = self.target_ratio[ticker] * 0.01
        value = round(total_value * target_ratio, 3)
        price = self.price_data.loc[date][ticker]
        number = round(value / price, 6)
        return (number, value, target_ratio * 100)
    
    def __available_dates(self, start_date = None, duration=None) -> pd.Index:
        if not start_date or start_date < self.start_date:
            start_date = self.start_date

        if start_date > self.price_data.index[-1]:
            start_date = self.price_data.index[-1]

        for date in self.price_data.index:
            if date >= start_date:
                start_date = date
                break

        start_idx = self.price_data.index.get_loc(start_date)
        
        date_len = len(self.price_data.index) - start_idx
        if duration:
            date_len = min(date_len, 252 * duration)
        end_idx = start_idx + date_len
        
        return self.price_data.index[start_idx:end_idx]
    
    @staticmethod
    def alphabeta(market_value, target_value, risk_free_rate=0.02, duration=None):
        market_value = market_value.dropna()
        target_value = target_value.dropna()

        start_date = max(market_value.index[0], target_value.index[0])
        end_date = min(market_value.index[-1], target_value.index[-1])
        if duration:
            end_date = min(end_date, start_date + pd.Timedelta(days=duration * 365))
            
        market_value = market_value[(market_value.index >= start_date) & (market_value.index < end_date)]
        target_value = target_value[(target_value.index >= start_date) & (target_value.index < end_date)]

        market_ret = np.array(market_value.pct_change().dropna(), dtype=np.float64)
        target_ret = np.array(target_value.pct_change().dropna(), dtype=np.float64)
        
        cov = np.cov(market_ret, target_ret)[0][1]
        var = np.var(market_ret)
        beta = cov / var
        expected_ret = risk_free_rate + beta * (market_ret.mean() * 252 - risk_free_rate)
        alpha = (target_ret.mean() * 252 - expected_ret) * 100

        return alpha, beta