import numpy as np
import pandas as pd
import threading
import matplotlib.pyplot as plt
import datetime as dt
import Data_Preparation
from itertools import compress, product
import time as t
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")
gbl_lock = threading.Lock()


class Modeling:
    def __init__(self, Data_object, use_MNlogit):
        # Model type
        self.use_MNlogit = use_MNlogit  # if True: Multinomial Logistic Regression, False: Random Forest

        # Bucketing parameters
        self.criteria = Data_object.criteria
        self.low = Data_object.low
        self.high = Data_object.high

        # Standardization Parameters
        self.train_means = Data_object.train_means
        self.train_stds = Data_object.train_stds
        self.do_standardization = Data_object.do_standardization

        # Model Content
        self.model_low = None
        self.model_mid = None
        self.model_high = None
        self.selected_features = None
        self.f1_score = None  # F1 score of model in test set
        self.cm = None  # Confusion matrix in test set
        self.fs_list = None
        self.cm_list = None

    def bucketing(self, data):
        '''
        This function will split given data based on bucketing criteria and low/high bound
        :param data: pd.Dataframe
        :return: data_low, data_mid, data_high, split dataset
        '''
        if self.criteria not in data.columns:
            raise Exception('Data to split does not have required indicator which serves as criteria!!')
        data = data.copy()
        data_low = data[data[self.criteria] < self.low]
        data_mid = data[(data[self.criteria] >= self.low) & (data[self.criteria] <= self.high)]
        data_high = data[data[self.criteria] > self.high]
        return data_low, data_mid, data_high

    def fit(self, X):
        '''
        :param X: np.array like, (number of observations, number of features)
        :return Y_pred: np.array like, predicted Y
        '''
        if (self.use_MNlogit == None) or (self.model_low == None) or (self.model_mid == None) or (
                self.model_high == None):
            raise Exception(
                'Model has not been estimated yet. Use Backward/Forward stepwise search or Globle search first!')

        X_low, X_mid, X_high = self.bucketing(X)
        X_low = X_low[self.selected_features]
        X_mid = X_mid[self.selected_features]
        X_high = X_high[self.selected_features]

        if self.do_standardization:
            std_inds_low = [x for x in self.selected_features if x in self.train_means[0].index]
            std_inds_mid = [x for x in self.selected_features if x in self.train_means[1].index]
            std_inds_high = [x for x in self.selected_features if x in self.train_means[2].index]

            X_low.loc[:, std_inds_low] = (X_low[std_inds_low] - self.train_means[0][std_inds_low]) / self.train_stds[0][
                std_inds_low]
            X_mid.loc[:, std_inds_mid] = (X_mid[std_inds_mid] - self.train_means[1][std_inds_mid]) / self.train_stds[1][
                std_inds_mid]
            X_high.loc[:, std_inds_high] = (X_high[std_inds_high] - self.train_means[2][std_inds_high]) / \
                                           self.train_stds[2][std_inds_high]

        Y_pred = pd.Series(np.full(X.shape[0], np.nan), index=X.index)

        if self.use_MNlogit:
            # Multinomial Logistic Model
            Y_pred_low = self.logis_model_pred(self.model_low, X_low)
            Y_pred_mid = self.logis_model_pred(self.model_mid, X_mid)
            Y_pred_high = self.logis_model_pred(self.model_high, X_high)
        else:
            # Random Forest
            Y_pred_low = self.rf_model_pred(self.model_low, X_low)
            Y_pred_mid = self.rf_model_pred(self.model_mid, X_mid)
            Y_pred_high = self.rf_model_pred(self.model_high, X_high)

        X_ls = [X_low, X_mid, X_high]
        pred_ls = [Y_pred_low, Y_pred_mid, Y_pred_high]
        for i in range(len(X_ls)):
            if pred_ls[i] is not None:
                Y_pred.loc[X_ls[i].index] = pred_ls[i]

        return Y_pred


    def logis_model_train(self, train):
        train = train.copy()
        Y = train['Y']
        X = sm.add_constant(train.drop(columns='Y'), has_constant='raise')
        if len(Y):
            MLmodel = sm.MNLogit(Y, X)
            MLresult = MLmodel.fit(disp=0)
            return MLresult
        else:
            raise ValueError('No samples!')

    def rf_model_train(self, train):
        train = train.copy()
        Y = train['Y']
        X = train.drop(columns='Y')
        clf_rf = RandomForestClassifier(n_estimators=50, random_state=1234)
        if len(Y):
            rf_model = clf_rf.fit(X, Y)
            return rf_model
        else:
            raise ValueError('No sample!')

    def logis_model_pred(self, logis_fit, X):
        X = X.copy()
        if X.shape[0]:
            X_test = sm.add_constant(X, has_constant='add')
            Y_pred = logis_fit.predict(X_test).idxmax(axis=1).values
            return Y_pred

    def rf_model_pred(self, rf_fit, X):
        if X.shape[0]:
            Y_pred = rf_fit.predict(X)
            return Y_pred

    def get_f1_score(self, train_sets, test_sets, selected_inds):
        train_low = train_sets[0]
        train_mid = train_sets[1]
        train_high = train_sets[2]
        test_low = test_sets[0]
        test_mid = test_sets[1]
        test_high = test_sets[2]

        train_low = train_low[selected_inds + ['Y']]
        train_mid = train_mid[selected_inds + ['Y']]
        train_high = train_high[selected_inds + ['Y']]

        test = pd.concat([test_low, test_mid, test_high], axis=0).sort_index()
        test_low = test_low[selected_inds]
        test_mid = test_mid[selected_inds]
        test_high = test_high[selected_inds]

        Y_pred = pd.Series(np.full(test.shape[0], np.nan), index=test.index)

        gbl_lock.acquire()
        if self.use_MNlogit:
            gbl_lock.release()
            model_low, model_mid, model_high = self.logis_model_train(train_low), self.logis_model_train(
                train_mid), self.logis_model_train(train_high)
            if np.isnan(model_low.llf):
                raise Exception('Too few samples in low senario for model training!')
            if np.isnan(model_mid.llf):
                raise Exception('Too few samples in middle senario for model training!')
            if np.isnan(model_high.llf):
                raise Exception('Too few samples in high senario for model training!')

            Y_pred_low, Y_pred_mid, Y_pred_high = self.logis_model_pred(model_low, test_low), self.logis_model_pred(
                model_mid, test_mid), self.logis_model_pred(model_high, test_high)
        else:
            gbl_lock.release()
            model_low, model_mid, model_high = self.rf_model_train(train_low), self.rf_model_train(
                train_mid), self.rf_model_train(train_high)
            Y_pred_low, Y_pred_mid, Y_pred_high = self.rf_model_pred(model_low, test_low), self.rf_model_pred(model_mid,
                                                                                                              test_mid), self.rf_model_pred(
                model_high, test_high)

        test_ls = [test_low, test_mid, test_high]
        pred_ls = [Y_pred_low, Y_pred_mid, Y_pred_high]
        for i in range(len(test_ls)):
            if pred_ls[i] is not None:
                Y_pred.loc[test_ls[i].index] = pred_ls[i]
        fs = f1_score(test['Y'].values, Y_pred.values, labels=[0, 1, 2], average='macro')
        cm = confusion_matrix(test['Y'].values, Y_pred.values, labels = [0,1,2])
        fs_list, cm_list = self.score_trend(test['Y'].values, Y_pred.values)
        return fs, cm, fs_list, cm_list, model_low, model_mid, model_high


    def score_trend(self, Y_real, Y_pred, interval_days=5):
        if len(Y_real) != len(Y_pred):
            raise Exception('Length of real value and predicted value should be the same!')

        Y_real_split = [Y_real[i:i + interval_days] for i in range(0, len(Y_real), interval_days)]
        Y_pred_split = [Y_pred[i:i + interval_days] for i in range(0, len(Y_pred), interval_days)]
        score_list = []
        cm_list = []
        for j in range(len(Y_real_split)):
            fs = f1_score(Y_real_split[j], Y_pred_split[j], labels=[0, 1, 2], average='macro')
            cm = confusion_matrix(Y_real_split[j], Y_pred_split[j],labels = [0,1,2])
            score_list.append(fs)
            cm_list.append(cm)
        return score_list, cm_list

    ### Feature Selection and Model training

    def Backward_Stepwise(self, train_sets, test_sets):
        print('Backward Stepwise Selection Running....')
        # Backward Stepwise Selection
        inds_back = list(train_sets[1].columns)
        inds_back.remove('Y')

        # train and predict with total variables
        f1_back_max = self.get_f1_score(train_sets, test_sets, inds_back)[0]

        while True:
            remove_ind = 0  # The indicator to be removed in selection below

            for ind in inds_back:
                try:
                    fs = self.get_f1_score(train_sets, test_sets, [n for n in inds_back if n != ind])[0]
                    if fs > f1_back_max:
                        f1_back_max = fs
                        remove_ind = ind
                except:
                    continue
            if remove_ind == 0:
                print('Backforward stepwise selection finished!')
                break
            else:
                inds_back.remove(remove_ind)
                if inds_back == []:
                    print('No invariable has been left in backforward selection!')
                    print('Last variable was {0} with F1 score equal to {1}'.format(remove_ind, f1_back_max))

        fs, cm, fs_list, cm_list, model_low, model_mid, model_high = self.get_f1_score(train_sets, test_sets, inds_back)
        self.model_low = model_low
        self.model_mid = model_mid
        self.model_high = model_high
        self.selected_features = inds_back
        self.f1_score = fs
        self.cm = cm
        self.fs_list = fs_list
        self.cm_list = cm_list

    def Forward_Stepwise(self, train_sets, test_sets):
        print('Forward Stepwise Selection Running....')
        inds_fwd = []
        f1_fwd_max = 0
        unselected_ind = list(train_sets[1].columns)
        unselected_ind.remove('Y')
        while True:
            add_ind = 0  # The indicator to be added in selection below
            for ind in unselected_ind:
                try:
                    fs = self.get_f1_score(train_sets, test_sets, inds_fwd + [ind])[0]

                    if fs > f1_fwd_max:
                        f1_fwd_max = fs
                        add_ind = ind
                except:
                    continue
            if add_ind == 0:
                print('Forward stepwise selection finished!')
                if inds_fwd == []:
                    print('No indicator has been selected!')
                break
            else:
                inds_fwd.append(add_ind)
                unselected_ind.remove(add_ind)

        fs, cm, fs_list,cm_list, model_low, model_mid, model_high = self.get_f1_score(train_sets, test_sets, inds_fwd)
        self.model_low = model_low
        self.model_mid = model_mid
        self.model_high = model_high
        self.selected_features = inds_fwd
        self.f1_score = fs
        self.cm = cm
        self.fs_list = fs_list
        self.cm_list = cm_list

    def combinations(self, items):
        return (set(compress(items, mask)) for mask in product(*[[0, 1]] * len(items)))

    def thread(self, train_sets, test_sets, inds):
        global gbl_lock
        # for inds in inds_part:
        try:
            fs = self.get_f1_score(train_sets, test_sets, list(inds))[0]
            gbl_lock.acquire()
            self.__result[tuple(inds)] = fs
            gbl_lock.release()
        except:
            pass

    def Pseudo_Global_Search(self, train_sets, test_sets, inds_to_add=None):
        print('Pseudo Global Search Running....')
        reduced_universe_size = 7 #number of indicators to run global search
        self.__result = {}
        start_time = t.time()
        # Global Selection
        tot_inds = list(train_sets[1].columns)
        tot_inds.remove('Y')

        # Preliminary Selection by Forward
        fs_dict = {}
        for ind in tot_inds:
            try:
                fs = self.get_f1_score(train_sets, test_sets, [ind])[0]
                fs_dict[ind] = fs
            except:
                pass
        if inds_to_add is not None:
            add_ls = list(inds_to_add) if isinstance(inds_to_add, list) else [inds_to_add]
            sort_inds = sorted(fs_dict, key=fs_dict.get, reverse=True)
            fs_list = [x for x in sort_inds if x not in add_ls][:reduced_universe_size-len(add_ls)] + add_ls
        else:
            fs_list = sorted(fs_dict, key=fs_dict.get, reverse=True)[:reduced_universe_size]
        print('Global Candidates are {0}'.format(fs_list))

        inds_combo = self.combinations(fs_list)
        inds_combo = [sorted(x) for x in inds_combo][1:]
        print('Total number of combinations is {0}...'.format(len(inds_combo)))

        thread_ls = []
        for inds in inds_combo:
            o = threading.Thread(target=self.thread, args=(train_sets.copy(), test_sets.copy(), inds))
            thread_ls.append(o)
        for x in thread_ls:
            x.start()
        for x in thread_ls:
            x.join()

        inds_opt = list(sorted(self.__result, key=self.__result.get, reverse=True)[0])  # optimal set of indicators
        print('Pseudo Global Search finished!')
        print('Total time used is {0} sec'.format(t.time() - start_time))

        fs, cm, fs_list, cm_list, model_low, model_mid, model_high = self.get_f1_score(train_sets, test_sets, inds_opt)
        self.model_low = model_low
        self.model_mid = model_mid
        self.model_high = model_high
        self.selected_features = inds_opt
        self.f1_score = fs
        self.cm = cm
        self.fs_list = fs_list
        self.cm_list = cm_list



    def summary(self, Display_in_Excel):
        if Display_in_Excel:
            content = []
            if self.use_MNlogit:
                content.append('Multinomial Logistic Regression Model')
            else:
                content.append('Random Forest Model')
            if self.do_standardization:
                content.append('with continuous data standardized under different senarios')
            content.append('')
            features = self.selected_features
            features_split = [features[i:i + 4] for i in range(0, len(features), 4)]
            content.append('Features of Final Model include {0} indicators:'.format(len(features)))
            for feature_piece in features_split:
                content.append(feature_piece)
            content.append('')
            content.append('F1 score of test period')
            content.append(str(np.round(self.f1_score,decimals =4)))
            content.append('Confusion Matrix')
            content.append(self.cm)
            # content.append('*Time-varying Accuracy Analysis*')
            # content.append('F1 scores every 5 days in test period:')
            # content.append(np.round(self.fs_list, decimals=3))
            # content.append('Confusion matrix of the largest F1 score,{0}:'.format(np.round(max(self.fs_list), decimals=3)))
            # content.append(self.cm_list[pd.Series(self.fs_list).idxmax()])
            # content.append('Confusion matrix of the smallest F1 score,{0}:'.format(np.round(min(self.fs_list), decimals=3)))
            # content.append(self.cm_list[pd.Series(self.fs_list).idxmin()])
            return content
        else:
            print('')
            print('**** Model Summary ****')
            if self.use_MNlogit:
                print('Multinomial Logistic Regression Model')
            else:
                print('Random Forest Model')
            if self.do_standardization:
                print('with continuous data standardized under different senarios')
            print('')
            features = self.selected_features
            features_split = [features[i:i + 5] for i in range(0, len(features), 5)]
            print('Features of Final Model include {0} indicators:'.format(len(features)))
            for feature_piece in features_split:
                print(feature_piece)
            print('')
            print('F1 score of test period reaches ' + str(self.f1_score))
            print('Confusion Matrix of test period:')
            print(self.cm)
            print('')
            print('*Time-varying Accuracy Analysis*')
            print('F1 scores every 5 days in test period:')
            print(np.round(self.fs_list, decimals=3))
            print('Confusion matrix of the largest F1 score,{0}:'.format(np.round(max(self.fs_list),decimals = 3)))
            print(self.cm_list[pd.Series(self.fs_list).idxmax()])
            print('Confusion matrix of the smallest F1 score,{0}:'.format(np.round(min(self.fs_list),decimals = 3)))
            print(self.cm_list[pd.Series(self.fs_list).idxmin()])
            print('********')
            print('')

