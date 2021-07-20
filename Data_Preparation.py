import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


class Data_Prep:

    def __init__(self, file_path, external_data = None):
        if external_data is None:
            xlsx = pd.ExcelFile(file_path)
            df = pd.read_excel(xlsx, 'Summary')
        else:
            df = external_data
        df.set_index('Dates', inplace=True)
        df = df.loc[~df.index.isna()]
        df.fillna(method = 'ffill', inplace=True)
        df.dropna(inplace = True)
        self.total_data = df
        self.latest_data = df.tail(1)  # Used to make predication. Note that it'll not be standardized until prediction

        self.train_set = None
        self.test_set = None
        self.train_sets = None  # distinguish 3 training sets with different senarios
        self.test_sets = None  # distinguish 3 testing sets with different senarios

        #Standardization recording
        self.train_means = None
        self.train_stds = None
        self.do_standardization = False

        #bucketing parameters
        self.criteria = None
        self.low = None
        self.high = None


    def __rolling(self, n, df, func):
        ls = []
        for i in range(df.shape[0]-n):
            if df['flag'][i]:
                ls.append(func(df['ES_adjusted'][i+1:i+1+n]))
            else:
                ls.append(func(df['ES1'][i+1:i+1+n]))
        ls += [np.nan]*5
        return pd.Series(ls, index=df.index)


    def create_regressand(self, chg_days=5):
        # Create Regressand Y
        # Within next chg_days days, if Y stay within range:0; goes below range:1; goes above range: 2

        df = self.total_data.copy()
        rolling_dates = df.index[df['ES1_tk'] != df['ES1_tk'].shift(-1)]
        group = []
        k = n = 0
        for x in df.index:
            if x < rolling_dates[k]:
                group.append(n)
            else:
                k += 1
                group.append(n)
                n += 1
        df['flag'] = False
        adjust_date = df.groupby(group).apply(lambda x: x[-5:]).index.levels[1]
        df.loc[adjust_date, 'flag'] = True
        df['ES_adjusted'] = df['ES1'].copy()
        df.loc[df.flag, 'ES_adjusted'] = df.loc[df.flag, 'ES2']

        #Use current ES price as baseline, VIX as volatility and +1&-1 volatiliy as range
        df['Low_B'] = df.ES_adjusted * (1 - df.VIX * np.sqrt(chg_days) / 1600)  # Lower boundary of range
        df['Up_B'] = df.ES_adjusted * (1 + df.VIX * np.sqrt(chg_days) / 1600)  # upper boundary of range


        df['max'] = self.__rolling(chg_days, df, max)
        df['min'] = self.__rolling(chg_days, df, min)
        #create categorical regressand
        df['Y'] = np.nan
        df.loc[(df['min'] >= df.Low_B) & (df['max'] <= df.Up_B), 'Y'] = 0
        df.loc[df['min'] < df.Low_B, 'Y'] = 1
        df.loc[df['max'] > df.Up_B, 'Y'] = 2
        df.drop(columns=['min', 'max', 'ES2', 'ES1_tk', 'ES_adjusted','flag','Low_B','Up_B',], inplace=True)

        self.total_data = df
        self.latest_data = df.tail(1)


    def add_new_indicators(self):
        df = self.total_data

        #Here we first transform unstationary regressors we've already found
        #AD_pctchg, MA10_pctchg, OBV_MA10_diff: stationarity-transformed indicators
        df['AD_pctchg'] = df.AD.pct_change()
        df['MA10_pctchg'] = df.SimpleMA10.pct_change()
        df['OBV_MA10'] = df.OBV.rolling(10).mean()
        df['OBV_MA10_diff'] = df.OBV_MA10.diff()

        # Here are indicators we built on our own based on raw indicators
        #Skew_senario, comparing Skew index and its bolling band, which can used as criteria of bucketing
        df['Skew_scenario'] = np.nan
        df['Skew_high'] = df.Skew_last.rolling(20).mean() + df.Skew_last.rolling(20).std()
        df['Skew_low'] = df.Skew_last.rolling(20).mean() - df.Skew_last.rolling(20).std()
        df.loc[df.Skew_last>df.Skew_high,'Skew_scenario'] = 1
        df.loc[(df.Skew_last >= df.Skew_low)&(df.Skew_last<= df.Skew_high),'Skew_scenario']=0
        df.loc[df.Skew_last < df.Skew_low, 'Skew_scenario'] = -1
        df.drop(columns=['Skew_low', 'Skew_high'], inplace=True)


        # VIXind, ADind, ATRind, Skewind, indicators built by comparing indicators and their moving average
        df['VIXind'] = 0
        df['ADind'] = 0
        df['ATRind'] = 0
        df['Skewind'] = 0

        df['MAVIX'] = df.VIX.rolling(5).mean()
        df['MAAD'] = df.AD.rolling(5).mean()
        df['MAATR'] = df.ATR.rolling(5).mean()
        df['MASkew'] = df.Skew_last.rolling(5).mean()

        df.loc[(((df.VIX) - (df.MAVIX)) * 100 / (df.MAVIX)) >= 20, 'VIXind'] = -1
        df.loc[(((df.VIX) - (df.MAVIX)) * 100 / (df.MAVIX)) < -20, 'VIXind'] = 1

        df.loc[(((df.AD) - (df.MAAD)) * 100 / (df.MAAD)) >= 1, 'ADind'] = 1
        df.loc[(((df.AD) - (df.MAAD)) * 100 / (df.MAAD)) < -1, 'ADind'] = -1

        df.loc[(((df.ATR) - (df.MAATR)) * 100 / (df.MAATR)) >= 20, 'ATRind'] = 1
        df.loc[(((df.ATR) - (df.MAATR)) * 100 / (df.MAATR)) < -20, 'ATRind'] = -1

        df.loc[(df.Skew_last - df.MASkew)*100/df.MASkew >= 2.5, 'Skewind'] = -1
        df.loc[(df.Skew_last - df.MASkew)*100/df.MASkew<= -2.5, 'Skewind'] = 1


        # RSI_cross, based on whether RSI_14 and RSI_30 get cross
        df['RSI_cross'] = 0
        df.loc[(df.RSI_14 <= df.RSI_30) & (df.RSI_14.shift(1) > df.RSI_30.shift(1)), 'RSI_cross'] = -1
        df.loc[(df.RSI_14 >= df.RSI_30) & (df.RSI_14.shift(1) < df.RSI_30.shift(1)), 'RSI_cross'] = 1

        # AD_dvg, using AD and ES
        df['AD_dvg'] = 0
        df['ES_MA10_pctchg'] = df.SimpleMA10.pct_change()
        df['AD_MA10'] = df.AD.rolling(10).mean()
        df['AD_MA10_pctchg'] = df.AD_MA10.pct_change()
        df.loc[(df.AD_MA10_pctchg > 0) & (df.ES_MA10_pctchg < 0), 'AD_dvg'] = 1  # bullish divergence
        df.loc[(df.AD_MA10_pctchg < 0) & (df.ES_MA10_pctchg > 0), 'AD_dvg'] = -1  # bearish divergence

        # MACD_cross, based whether MACD and signal line get across
        df['MACD_cross'] = 0
        df['signal_line'] = df.MACD.rolling(9).mean()
        df.loc[(df.MACD >= df.signal_line) & (df.MACD.shift(1) < df.signal_line.shift(1)), 'MACD_cross'] = 1
        df.loc[(df.MACD <= df.signal_line) & (df.MACD.shift(1) > df.signal_line.shift(1)), 'MACD_cross'] = -1

        #MARSTOC, a hybrid indicator based on RSI, MACD and Stochastics
        df['MARSTOC'] = 0
        df.loc[((df['MACD'] > 0) & (df['RSI_30'] > 50) & (df['Stochastics'] > 50)), 'MARSTOC'] = 1
        df.loc[((df['MACD'] < 0) & (df['RSI_30'] < 50) & (df['Stochastics'] < 50)), 'MARSTOC'] = -1
        df['MARSTOC'] = df['MARSTOC'].diff()

        #RSPRIMA, a hybrid indicator based on Relative Strength, Close Price of ES1 and Simple Moving Average
        df['RSPRIMA'] = 0
        df.loc[((df['ES1'] > df['SimpleMA10']) & (df['RSI_30'] > 50)), 'RSPRIMA'] = 1
        df.loc[((df['ES1'] < df['SimpleMA10']) & (df['RSI_30'] < 50)), 'RSPRIMA'] = -1
        df['RSPRIMA'] = df['RSPRIMA'].diff()

        #SMASTOC, a hybrid indicator based on Simple Moving Average and Stochastics
        df['SMASTOC'] = 0
        df.loc[((df['ES1'] < df['SimpleMA10']) & (df['Stochastics'] > 80)), 'SMASTOC'] = -1
        df.loc[((df['ES1'] > df['SimpleMA10']) & (df['Stochastics'] < 20)), 'SMASTOC'] = 1
        df['SMASTOC'] = df['SMASTOC'].diff()

        #Delete redundant data
        df.drop(columns = ['ES1','AD_MA10','AD_MA10_pctchg','ES_MA10_pctchg','OBV_MA10','signal_line','MAVIX','MAAD','MAATR','MASkew'],inplace = True)
        self.total_data = df
        self.latest_data = df.tail(1)

    # Introduce Time Lag Variable
    def add_time_lag(self, lag_inds, lag_list):
        '''
        param lag_inds: list or array of string, containing names of indicators whose time lag will be added to dataframe
        param lag_list: list or array of positive integer, containing value of time lag. For example, if lag_list = [1,2], then
        will add lag-1 and lag-2 value of all of indicators, lag_inds.
        '''
        lag_data = self.total_data[lag_inds].copy()

        for i in lag_list:
            if i <= 0:
                raise Exception('Inputted lag should be positive integers!')
            colnames = []
            for name in lag_inds:
                colnames.append(name + '_lag_' + str(i))
            lag_df = lag_data[lag_inds].shift(i).copy()
            lag_df.columns = colnames
            lag_data = pd.concat([lag_data, lag_df], axis=1)

        new_dataX = pd.concat([self.total_data, lag_data.drop(columns=lag_inds)], axis=1).drop(columns = 'Y').dropna()
        new_total_data = pd.concat([new_dataX, self.total_data['Y']], axis = 1, join_axes=[new_dataX.index])
        self.total_data = new_total_data  #update latest, train and test data while keeping total_data unchaged
        self.latest_data = new_total_data.tail(1)


    def stationary_filter(self, p_val=0.05):
        #Testing stationarity of data and keep only stationary series
        df = self.total_data
        data_X = df.drop(columns=['Y'])
        data_X.dropna(inplace = True)  #Total data won't have NA in the beginning, but may have NA at tail

        #Test Stationarity of all of regressors and drop those unstationary indicators
        colnames = data_X.columns
        pv_list = []
        for ind in colnames:
            p_value = adfuller(data_X[ind])[1]
            pv_list.append(p_value)
        result = pd.DataFrame({'Indicator': colnames, 'p_value': pv_list})
        result['stationary'] = result.p_value < p_val

        if result[~result.stationary].shape[0]:
            print('Drop {0} in Stationary Transform'.format(list(colnames[~result.stationary])))

        data_X = data_X[colnames[result.stationary]]
        self.total_data = pd.concat([data_X, df[['Y']]], axis = 1, join_axes= [data_X.index])
        self.latest_data = self.total_data.tail(1)


    def data_split(self, train_len, test_len):
        #Dataset Split
        self.test_set = self.total_data.dropna().iloc[-test_len:]  # dataset for testing and calculating f1 score
        self.train_set = self.total_data.dropna().iloc[-train_len - test_len : -test_len]  # dataset for feature selection and model training


    def bucketing(self, criteria = 'VIX', low = 15, high = 25, delete_criteria = False):
        '''
        This function will split given data based on bucketing criteria and low/high bound
        param delete_criteria: bool, if True, indicator which serves as criteria will be removed from dataset after bucketing
        '''

        self.criteria = criteria
        self.low = low
        self.high = high

        if self.criteria not in self.train_set.columns:
            raise Exception('Data to split does not have required indicator which serves as criteria!!' )
        train_set = self.train_set.copy()
        train_low = train_set[train_set[self.criteria] < self.low]
        train_mid = train_set[(train_set[self.criteria] >= self.low) & (train_set[self.criteria] <= self.high)]
        train_high = train_set[train_set[self.criteria] > self.high]

        test_set = self.test_set.copy()
        test_low = test_set[test_set[self.criteria] < self.low]
        test_mid = test_set[(test_set[self.criteria] >= self.low) & (test_set[self.criteria] <= self.high)]
        test_high = test_set[test_set[self.criteria] > self.high]

        if delete_criteria:
            train_low = train_low.drop(columns = criteria)
            train_mid = train_mid.drop(columns=criteria)
            train_high = train_high.drop(columns=criteria)
            test_low = test_low.drop(columns=criteria)
            test_mid = test_mid.drop(columns=criteria)
            test_high = test_high.drop(columns=criteria)


        self.train_sets = [train_low, train_mid, train_high]
        self.test_sets = [test_low, test_mid, test_high]
        self.train_set = pd.concat(self.train_sets, axis =0).sort_index()
        self.test_set = pd.concat(self.test_sets, axis=0).sort_index()

    def standardize(self, train_set, test_set):
        '''
        This function will standardize continuous indicators in training dataset, testing dataset as well as lastest
        data using mean and standard deviation of training dataset.
        Note that total data will not be changed.
        :param train_set: pd.DataFrame, training dataset with indicators as its columns
        :param test_set: pd.DataFrame, testing dataset with indicators as its columns
        '''


        train_set_std = train_set.copy()
        test_set_std = test_set.copy()

        #Select continuous indicators
        sample = pd.concat([train_set.head(5), train_set.tail(5)], axis = 0)
        ctn_inds = []
        for ind in sample.columns:
            n = sample[ind].unique().shape[0]
            if n > 6:
                ctn_inds.append(ind)

        train_mean = train_set_std[ctn_inds].mean()
        train_std = train_set_std[ctn_inds].std()

        train_set_std.loc[:, ctn_inds] = (train_set_std[ctn_inds] - train_mean) / train_std
        test_set_std.loc[:, ctn_inds] = (test_set_std[ctn_inds] - train_mean) / train_std

        return train_set_std, test_set_std, train_mean, train_std


    def standardize_indicators(self):
        #standardize sets under 3 senarios seperately
        train_std_low, test_std_low, mean_low, std_low = self.standardize(self.train_sets[0], self.test_sets[0])
        train_std_mid, test_std_mid, mean_mid, std_mid = self.standardize(self.train_sets[1], self.test_sets[1])
        train_std_high, test_std_high, mean_high, std_high = self.standardize(self.train_sets[2], self.test_sets[2])
        #record standardized data sperately
        self.train_sets = [train_std_low, train_std_mid, train_std_high]
        self.test_sets = [test_std_low, test_std_mid, test_std_high]
        #record statistics used to do standardization
        self.train_means = [mean_low, mean_mid, mean_high]
        self.train_stds = [std_low, std_mid, std_high]
        self.do_standardization = True
        #update train_set and test_set
        self.train_set = pd.concat(self.train_sets, axis = 0).sort_index()
        self.test_set = pd.concat(self.test_sets, axis=0).sort_index()


    def summary(self, Display_in_Excel):
        if Display_in_Excel:
            content = []
            content.append('Model is trained on training period from {0} to {1} containing {2} days'.format(
                self.train_set.index[0].strftime("%m/%d/%Y"),
                self.train_set.index[-1].strftime("%m/%d/%Y"),
                self.train_set.shape[0]))
            content.append(
                'to optimize F1 score of testing period from {0} to {1} containing {2} days'.format(
                    self.test_set.index[0].strftime("%m/%d/%Y"),
                    self.test_set.index[-1].strftime("%m/%d/%Y"),
                    self.test_set.shape[0]))
            content.append('')
            content.append('Bucketing criteria is {0} with lower threshold {1} and upper threshold {2}'.format(self.criteria,
                                                                                                      self.low,
                                                                                                      self.high))
            content.append('In training period,')
            content.append('Low senario has {0} observations'.format(self.train_sets[0].shape[0]))
            content.append('Mid senario has {0} observations'.format(self.train_sets[1].shape[0]))
            content.append('High senario has {0} observations'.format(self.train_sets[2].shape[0]))
            content.append('')
            names = [name for name in self.train_set.columns if name != 'Y']
            name_split = [names[i:i + 5] for i in range(0, len(names), 5)]
            content.append('Feature universe includes {0} indicators:'.format(len(names)))
            for name_piece in name_split:
                content.append(name_piece)
            if self.do_standardization:
                content.append('Continuous features have been standardized before modeling!')
            content.append('')
            return content
        else:
            print('')
            print('**** Data Summary ****')
            print('Model is trained on training period from {0} to {1} containing {2} days'.format(self.train_set.index[0].strftime("%m/%d/%Y"),
                                                                                                   self.train_set.index[-1].strftime("%m/%d/%Y"),
                                                                                                   self.train_set.shape[0]))
            print(
                'to optimize F1 score of testing period from {0} to {1} containing {2} days'.format(self.test_set.index[0].strftime("%m/%d/%Y"),
                                                                                                    self.test_set.index[-1].strftime("%m/%d/%Y"),
                                                                                                    self.test_set.shape[0]))
            print('')
            print('Bucketing criteria is {0} with lower threshold {1} and upper threshold {2}'.format(self.criteria,
                                                                                                      self.low,
                                                                                                      self.high))
            print('In training period,')
            print('Low scenario has {0} observations'.format(self.train_sets[0].shape[0]))
            print('Mid scenario has {0} observations'.format(self.train_sets[1].shape[0]))
            print('High scenario has {0} observations'.format(self.train_sets[2].shape[0]))
            print('')
            names = [name for name in self.train_set.columns if name != 'Y']
            name_split = [names[i:i + 5] for i in range(0, len(names), 5)]
            print('Feature universe includes {0} indicators:'.format(len(names)))
            for name_piece in name_split:
                print(name_piece)
            if self.do_standardization:
                print('Continuous features have been standardized before modeling!')
            print('***********')
            print('')

