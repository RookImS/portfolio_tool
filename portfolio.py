import itertools
import datetime as dt
import pandas as pd
import numpy as np

class Portfolio:
    # stock_data
    def __init__(self, stock_data: pd.DataFrame, start_cash: float, target_ratio: dict = None, buy_ratio: float = 5.0, sell_ratio: float = 5.0):
        # 주식 데이터들 중에서 포트폴리오 목표 종목만 뽑는다.
        self.__stock_list = set(stock_data.columns.get_level_values(0))
        if target_ratio is None:
            target_ratio = {ticker: round(100 / len(self.__stock_list), 2) for ticker in self.__stock_list}
        self.__stock_list = list(self.__stock_list.intersection(target_ratio.keys()))
        # 뽑은 종목들을 포트폴리오 목표에 맞게 순서를 맞춘다.
        target_ratio_list = list(target_ratio)
        self.__stock_list.sort(key=lambda x: target_ratio_list.index(x))
        # 맞춘 순서에 맞게 실제 데이터를 뽑아낸다.
        stock_data = stock_data.loc[:, self.__stock_list]
        self.__price_data = stock_data.loc[:, (stock_data.columns.get_level_values(1) == 'price')].copy()
        self.__price_data = self.__price_data.droplevel(1, axis=1)
        self.__adj_price_data = stock_data.loc[:, (stock_data.columns.get_level_values(1) == 'adj_price')].copy()
        self.__adj_price_data = self.__adj_price_data.droplevel(1, axis=1)
        self.__divend_data = stock_data.loc[:, (stock_data.columns.get_level_values(1) == 'divend')].copy()
        self.__divend_data = self.__divend_data.droplevel(1, axis=1)
        self.__start_cash = start_cash
        self.__target_ratio = target_ratio
        self.__buy_ratio = buy_ratio
        self.__sell_ratio = sell_ratio
        self.__target_buy_ratio = {ticker: target_ratio[ticker] * buy_ratio / 100 for ticker in self.__stock_list}
        self.__target_sell_ratio = {ticker: target_ratio[ticker] * sell_ratio / 100 for ticker in self.__stock_list}
    

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
    @property
    def target_ratio(self) -> dict:
        return self.__target_ratio
    @property
    def buy_ratio(self) -> float:
        return self.__buy_ratio
    @buy_ratio.setter
    def buy_ratio(self, value):
        self.__buy_ratio = value
        self.__target_buy_ratio = {ticker: self.target_ratio[ticker] * value / 100 for ticker in self.stock_list}
    @property
    def sell_ratio(self) -> float:
        return self.__sell_ratio
    @sell_ratio.setter
    def sell_ratio(self, value):
        self.__sell_ratio = value
        self.__target_sell_ratio = {ticker: self.target_ratio[ticker] * value / 100 for ticker in self.stock_list}
    
    def backtest(self, start_date=None, duration=None, rebalancing_cycle=1, cash_flow=0, cash_flow_growth=3, cash_flow_cycle=None, benchmark=None):
        info_list =  ['price', 'number', 'value', 'weight']
        stock_info =  list(itertools.product(self.stock_list, info_list))

        col_tuple = [('total', 'value'), ('cash', 'value'), ('cash', 'weight')] + stock_info
        col = pd.MultiIndex.from_tuples(col_tuple)

        dates = self.__available_dates(start_date, duration)
        rebalancing_dates = dates[::rebalancing_cycle]
        cash_flow_dates = dates[::cash_flow_cycle] if cash_flow_cycle else rebalancing_dates
        cash_flow_growth_rate = cash_flow_growth * 0.01
        history = pd.DataFrame(columns=col, index=dates)

        # 첫 값 설정
        first_date = dates[0]
        total_value = self.__start_cash
        for ticker in self.stock_list:
            history.loc[first_date, ('total', 'value')] = total_value
            history.loc[first_date, ('cash', 'value')] = 0
            history.loc[first_date, ('cash', 'weight')] = 0
            history.loc[:, (ticker, 'price')] = self.price_data[ticker]
            history.loc[first_date, (history.columns.get_level_values(0) == ticker) & (history.columns.get_level_values(1) != 'price')] = self.__ideal_nvw(total_value, ticker, first_date)
        
        # numpy 연산을 위해 필요한 데이터들
        rebalancing_dates_idx = list(reversed(range(0, len(dates), rebalancing_cycle)))
        cash_flow_dates_idx = list(reversed(range(0, len(dates), cash_flow_cycle))) if cash_flow_cycle else rebalancing_dates_idx.copy()
        history_np = history.to_numpy(copy=True)
        price_np = (self.price_data.loc[dates]).to_numpy()
        col_np_idx = { x:i for i, x in enumerate(col_tuple) }
        target_ratio_np = pd.Series(self.target_ratio).to_numpy()
        sell_ratio_np = pd.Series(self.__target_sell_ratio).to_numpy()
        buy_ratio_np = pd.Series(self.__target_buy_ratio).to_numpy()

        for i in range(1, len(dates)):
            # cash_value = history.loc[dates[i - 1]][('cash', 'value')]
            cash_value = history_np[i - 1][col_np_idx[('cash', 'value')]]
            total_value = cash_value
            # 현금 흐름 처리
            if cash_flow != 0:
                if i % 252 == 0:
                    cash_flow = cash_flow * (1 + cash_flow_growth_rate)
                if not cash_flow_dates_idx and i == cash_flow_dates_idx[-1]:
                    cash_flow_dates_idx.pop()
                    cash_value += cash_flow
            # 가격 변동 처리
            # prev_num = history.loc[dates[i - 1]][:, 'number']
            # prev_num_value = prev_num * self.price_data.loc[dates[i]]
            prev_num = history_np[i - 1][col_np_idx[(self.stock_list[0], 'number')]::4]
            prev_num_value = prev_num * price_np[i]
            total_value += prev_num_value.sum()
            # 배당금 처리
            divend_df = self.acculate_divend(dates[i-1], dates[i])
            divend = (divend_df.sum() * prev_num).sum()
            cash_value += divend
            total_value += divend
            # history.loc[dates[i], ('cash', 'value')] = cash_value
            # history.loc[dates[i], ('total', 'value')] = total_value
            history_np[i][col_np_idx[('cash', 'value')]] = cash_value
            history_np[i][col_np_idx[('total', 'value')]] = total_value

            # 변동에 따른 리밸런싱 계산
            # prev_num_ratio = prev_num_value / total_value * 100
            # target_ratio_sr = pd.Series(self.target_ratio)
            # ratio_diff = (prev_num_ratio - target_ratio_sr)
            prev_num_ratio = prev_num_value / total_value * 100

            # need_sell = ratio_diff[ratio_diff >= pd.Series(self.__target_sell_ratio)].index.to_list()
            # need_buy = ratio_diff[ratio_diff <= -pd.Series(self.__target_buy_ratio)].index.to_list()
            # need_trade = need_sell + need_buy
            # for ticker in self.stock_list:
            #     if dates[i] in rebalancing_dates and ticker in need_trade:
            #         history.loc[dates[i], (history.columns.get_level_values(0) == ticker) & (history.columns.get_level_values(1) != 'price')] = self.__ideal_nvw(total_value, ticker, dates[i])
            #         history.loc[dates[i], ('cash', 'value')] += ratio_diff[ticker] * total_value * 0.01
            #     else:
            #         history.loc[dates[i], (ticker, 'number')] = history.loc[dates[i - 1], (ticker, 'number')]
            #         history.loc[dates[i], (ticker, 'value')] = prev_num_value[ticker]
            #         history.loc[dates[i], (ticker, 'weight')] = prev_num_ratio[ticker]
        
            ratio_diff = None
            need_trade = np.array([False] * len(self.stock_list))
            if not rebalancing_dates_idx and i == rebalancing_dates_idx[-1]:
                rebalancing_dates_idx.pop()
                ratio_diff = (prev_num_ratio - target_ratio_np)
                need_sell = ratio_diff >= sell_ratio_np
                need_buy = ratio_diff <= -buy_ratio_np
                need_trade = need_sell | need_buy

            for ticker_idx in range(len(self.stock_list)):
                if need_trade[ticker_idx]:
                    nvw = self.__ideal_nvw(total_value, self.stock_list[ticker_idx], dates[i])
                    history_np[i][col_np_idx[(self.stock_list[ticker_idx], 'number')]] = nvw[0]
                    history_np[i][col_np_idx[(self.stock_list[ticker_idx], 'value')]] = nvw[1]
                    history_np[i][col_np_idx[(self.stock_list[ticker_idx], 'weight')]] = nvw[2]
                    history_np[i][col_np_idx[('cash', 'value')]] += ratio_diff[ticker_idx] * total_value * 0.01
                else:
                    history_np[i][col_np_idx[(self.stock_list[ticker_idx], 'number')]] = history_np[i - 1][col_np_idx[(self.stock_list[ticker_idx], 'number')]]
                    history_np[i][col_np_idx[(self.stock_list[ticker_idx], 'value')]] = prev_num_value[ticker_idx]
                    history_np[i][col_np_idx[(self.stock_list[ticker_idx], 'weight')]] = prev_num_ratio[ticker_idx]

            history_np[i][col_np_idx[('cash', 'weight')]] = history_np[i][col_np_idx[('cash', 'value')]] / total_value * 100
            # history.loc[dates[i], ('cash', 'weight')] = history.loc[dates[i]][('cash', 'value')] / total_value * 100

        # history.insert(0, ('rebalancing', 'flag'), False)
        # history.loc[rebalancing_dates, ('rebalancing', 'flag')] = True

        # stat = pd.DataFrame(columns=['portfolio'], index=['cagr', 'stdev', 'mdd', 'beta', 'alpha'])
        # start_price = history.loc[dates[0], ('total', 'value')]
        # end_price = history.loc[dates[-1], ('total', 'value')]
        # stat.loc['cagr', 'portfolio'] = ((end_price / start_price) ** (252 / len(dates)) - 1) * 100
        # stat.loc['stdev', 'portfolio'] = history.loc[:, ('total', 'value')].pct_change().std() * np.sqrt(252) * 100
        # peak_value = history[('total', 'value')].cummax()
        # drawdown = (history[('total', 'value')] - peak_value) / peak_value
        # stat.loc['mdd', 'portfolio'] = -1 * drawdown.min() * 100
        # if benchmark is not None:
        #     alpha, beta = Portfolio.alphabeta(benchmark, history.loc[:, ('total', 'value')], duration=duration)
        #     stat.loc['alpha', 'portfolio'] = alpha
        #     stat.loc['beta', 'portfolio'] = beta

        print(history_np)
        print(history_np.shape)
        return history

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
            alpha, beta = Portfolio.alphabeta(market_value, target_value, risk_free_rate, duration)
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