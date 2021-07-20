import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
import xlwings as xw
from Data_Preparation import Data_Prep
from Model_Training import Modeling


def save_model(model_obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(model_obj, output, pickle.HIGHEST_PROTOCOL)
        print('Model Saved!')

def read_model(filename):
    with open(filename, 'rb') as input:
        model = pickle.load(input)
    return model

#Modeling Training
def ES_model_training(excel_path='../',Run_in_Excel = True):
    cell_idx = 0
    if Run_in_Excel:
        wb = xw.Book.caller().sheets['Main']
        wb.range('L3').offset(row_offset = cell_idx).value = 'Preparing Data...'

    ####Data Preparation###
    file_path = excel_path + 'ES Predicting Tool.xlsm'
    Data = Data_Prep(file_path)
    Data.create_regressand(chg_days=5)
    Data.add_new_indicators()
    Data.add_time_lag(lag_inds=['VIX'], lag_list=[1, 2])
    Data.stationary_filter()  # delete unstationary series
    Data.data_split(train_len=2000, test_len=60)  # split training data and testing data
    Data.bucketing(criteria='VIX', low=15, high=25)  # bucketing data to 3 senarios using VIX
    # Data.bucketing(criteria='Skew_senario', low=0, high=0,delete_criteria= True)  # bucketing data using Skew
    Data.standardize_indicators()  # standardize continuous series under each senario
    summary = Data.summary(Display_in_Excel=Run_in_Excel)
    if Run_in_Excel:
        wb.range('L3').offset(row_offset=cell_idx).value = 'Data Preparation finished!'
        cell_idx +=1
        for i in range(len(summary)):
            wb.range('L14').offset(row_offset = i).value = summary[i]

    ####Modeling####
    MNlogit = True # If True, use Multinomial Logistic Regression Model; False, use Random Forest Model
    model_backward_MN = Modeling(Data, use_MNlogit=MNlogit)
    model_forward_MN = Modeling(Data, use_MNlogit=MNlogit)
    model_global_MN = Modeling(Data, use_MNlogit=MNlogit)
    MNlogit = False  # If True, use Multinomial Logistic Regression Model; False, use Random Forest Model
    model_backward_RF = Modeling(Data, use_MNlogit=MNlogit)
    model_forward_RF = Modeling(Data, use_MNlogit=MNlogit)
    model_global_RF = Modeling(Data, use_MNlogit=MNlogit)


    if Run_in_Excel:
        wb.range('L3').offset(row_offset=cell_idx).value = 'Modeling Starts'
        cell_idx +=1
        wb.range('L3').offset(row_offset=cell_idx).value = 'Backward Stepwise Selection is running...'
 
    start = time.time()
     # Backward Selection
    model_backward_MN.Backward_Stepwise(Data.train_sets, Data.test_sets)
    model_backward_RF.Backward_Stepwise(Data.train_sets, Data.test_sets)
    end1 = time.time()
    print('Backward Selections took {0} secs\n'.format(end1-start))
    if Run_in_Excel:
        wb.range('L3').offset(row_offset=cell_idx).value = 'Backward Stepwise ' \
                                                            'Selection finished using {0} secs'.format(
                                                         np.round(end1-start, decimals =2))
 
        wb.range('L3').offset(row_offset=cell_idx+1).value = 'Forward Stepwise Selection is running...'
        cell_idx += 1
 
     #Forward Selection
    model_forward_MN.Forward_Stepwise(Data.train_sets, Data.test_sets)
    model_forward_RF.Forward_Stepwise(Data.train_sets, Data.test_sets)
    end2 = time.time()
    print('Forward Selections took {0} secs\n'.format(end2-end1))
    if Run_in_Excel:
        wb.range('L3').offset(row_offset=cell_idx).value = 'Forward Stepwise ' \
                                                            'Selection finished using {0} secs'.format(
                                                         np.round(end2-end1, decimals = 2))
 
        wb.range('L3').offset(row_offset=cell_idx+1).value = 'Pseudo Global Search is running...'
        cell_idx += 1
 
     #Global Search
    model_global_MN.Pseudo_Global_Search(Data.train_sets, Data.test_sets,inds_to_add=['VIX','Skewind'])
    model_global_RF.Pseudo_Global_Search(Data.train_sets, Data.test_sets,inds_to_add=['VIX','Skewind'])
    end3 = time.time()
    print('Pseudo Global Searches took {0} secs\n'.format(end3 - start))
    if Run_in_Excel:
        wb.range('L3').offset(row_offset=cell_idx).value = 'Pseudo Global Search finished using {0} secs'.format(
                                                         np.round(end3 - start, decimals=2))
        cell_idx += 1
 
 
    fs_list = [model_backward_MN.f1_score, model_backward_RF.f1_score,
            model_forward_MN.f1_score, model_forward_RF.f1_score,
            model_global_MN.f1_score,model_global_RF.f1_score]
    model_list = [model_backward_MN, model_backward_RF,
               model_forward_MN, model_forward_RF,
               model_global_MN, model_global_RF]
    #Choose model with maximal F1 score
    model_best = model_list[pd.Series(fs_list).idxmax()]

    # Test each kind of model
#    model_backward_MN.Backward_Stepwise(Data.train_sets, Data.test_sets)
#    model_best = model_backward_MN

    summary = model_best.summary(Display_in_Excel=Run_in_Excel)
    if Run_in_Excel:
        i = 0
        while i < len(summary):
            content = summary[i]
            if content == 'F1 score of test period':
                wb.range('H15').value = summary[i+1]
                i +=2
                continue
            if content == 'Confusion Matrix':
                cm = summary[i+1]
                for row in range(3):
                    for col in range(3):
                        wb.range('H19').offset(row_offset=row, column_offset = col).value = cm[row][col]
                i +=2
                continue
            wb.range('A15').offset(row_offset = i).value = content
            i +=1

        #plot F1 score every 5 days
        fig = plt.figure(figsize=(3, 2))
        plt.plot(model_best.fs_list)
        plt.xticks(np.arange(0, len(model_best.fs_list), 2), np.arange(0, len(model_best.fs_list), 2) * 5)
        plt.title('F1 score every 5 days')
        plt.ylabel('F1 score')
        wb.pictures.add(fig, name='F1_Score', update=True, left=wb.range('F23').left, top=wb.range('F23').top)

        #Print out function coefficients or feature importance
        if model_best.use_MNlogit:
            wb.range('A15').offset(row_offset=i - 4).value = "Coefficients of Function under Each Scenario"
            wb.range('A15').offset(row_offset=i - 3).value = "Low Scenario"
            para = np.round(model_best.model_low.params, decimals=3)
            para.columns = ['Prob(Dump)/Prob(Flat)', 'Prob(Rally)/Prob(Flat)']
            para.index.name = 'Coeff'
            wb.range('A15').offset(row_offset=i - 2).value = para
            i = i+  para.shape[0] + 3

            wb.range('A15').offset(row_offset=i - 3).value = "Mid Scenario"
            para = np.round(model_best.model_mid.params, decimals=3)
            para.columns = ['Prob(Dump)/Prob(Flat)', 'Prob(Rally)/Prob(Flat)']
            para.index.name = 'Coeff'
            wb.range('A15').offset(row_offset=i - 2).value = para

            if i<24:
                i = i + para.shape[0] + 3
                wb.range('A15').offset(row_offset=i - 3).value = "High Scenario"
                para = np.round(model_best.model_high.params, decimals=3)
                para.columns = ['Prob(Dump)/Prob(Flat)', 'Prob(Rally)/Prob(Flat)']
                para.index.name = 'Coeff'
                wb.range('A15').offset(row_offset=i - 2).value = para
            else:
                wb.range('A15').offset(row_offset=i - 3,column_offset = 5).value = "High Scenario"
                para = np.round(model_best.model_high.params, decimals=3)
                para.columns = ['Prob(Dump)/Prob(Flat)', 'Prob(Rally)/Prob(Flat)']
                para.index.name = 'Coeff'
                wb.range('A15').offset(row_offset=i - 2,column_offset = 5).value = para

        else:
            # plot feature importance
            # Low Scenario
            fig_low, ax_low = plt.subplots(figsize=(3, 2))
            imp_feature = pd.DataFrame([model_best.model_low.feature_importances_], columns=model_best.selected_features,
                                       index=['imp'])
            imp_feature.T.sort_values(by='imp', ascending=False).plot(kind='bar', color='steelblue', ax=ax_low)
            plt.ylabel('Feature Importance')
            plt.title('Low Scenario')
            plt.legend().set_visible(False)
            wb.pictures.add(fig_low, name='Low_importance', update=True, left= wb.range('A15').offset(row_offset = i-4).left,
                            top=wb.range('A15').offset(row_offset = i-4).top)
            # Mid Scenario
            fig_mid, ax_mid = plt.subplots(figsize=(3, 2))
            imp_feature = pd.DataFrame([model_best.model_mid.feature_importances_], columns=model_best.selected_features,
                                       index=['imp'])
            imp_feature.T.sort_values(by='imp', ascending=False).plot(kind='bar', color='steelblue', ax=ax_mid)
            plt.ylabel('Feature Importance')
            plt.title('Mid Scenario')
            plt.legend().set_visible(False)
            wb.pictures.add(fig_mid, name='Mid_importance', update=True, left=wb.range('A15').offset(row_offset = i+11).left,
                            top=wb.range('A15').offset(row_offset = i+11).top)
            # High Scenario
            fig_high, ax_high = plt.subplots(figsize=(3, 2))
            imp_feature = pd.DataFrame([model_best.model_high.feature_importances_],
                                       columns=model_best.selected_features,
                                       index=['imp'])
            imp_feature.T.sort_values(by='imp', ascending=False).plot(kind='bar', color='steelblue', ax=ax_high)
            plt.ylabel('Feature Importance')
            plt.title('High Scenario')
            plt.legend().set_visible(False)
            wb.pictures.add(fig_high, name='High_importance', update=True, left=wb.range('A15').offset(row_offset = i+11,
                                                                                                       column_offset =5).left,
                            top=wb.range('A15').offset(row_offset = i+11, column_offset = 5).top)

    #Save trained model to a file
    if Run_in_Excel:
        save_model(model_best, excel_path + 'Supportive_Files/Model_Storage.pkl')
    else:
        save_model(model_best, 'Model_Storage.pkl')
    print('Model Training Finished!!')
    if Run_in_Excel:
        wb.range('L3').offset(row_offset=cell_idx).value = 'Model is saved'
        wb.range('L3').offset(row_offset=cell_idx+1).value = 'Model Training finished!'




#Predict based on model
def ES_model_predict(excel_path='../',Run_in_Excel = True):
    if Run_in_Excel:
        wb = xw.Book.caller().sheets['Main']
        wb.range("L3").value = "Predicting..."
    file_path = excel_path + 'ES Predicting Tool.xlsm'
    xlsx = pd.ExcelFile(file_path)
    df = pd.read_excel(xlsx, 'Summary')
    df.set_index('Dates', inplace=True)
    df = df.loc[~df.index.isna()]
    df.fillna(method = 'ffill', inplace=True)
    df.dropna(inplace = True)
    latest_X = df.iloc[-30:, :]  #latest data used to make prediction
    latest_X.reset_index(inplace = True)
    
    # Transform latest X, for example, add new indicators, time lags, do standardization etc.
    Data = Data_Prep(file_path,external_data=latest_X)
    Data.create_regressand(chg_days=5)
    Data.add_new_indicators()
    Data.add_time_lag(lag_inds=['VIX'], lag_list=[1, 2])
    latest_X = Data.latest_data

    #Import existing model
    if Run_in_Excel:
        model_best = read_model(excel_path + 'Supportive_Files/Model_Storage.pkl')
    else:
        model_best = read_model('Model_Storage.pkl')
    content = model_best.summary(Display_in_Excel = Run_in_Excel)

    #Make prediction
    Y_pred = model_best.fit(latest_X)
    print('')
    print('Predicting result:')
    print(Y_pred)

    #Export predicting result to excel
    if Run_in_Excel:
        wb.range('A7').value = Y_pred.index.strftime("%m/%d/%Y")
        wb.range('B7').value = Y_pred.values
        wb.range('L4').value = "Prediction finished!"


if __name__ == '__main__':
    #Test fuctions here
    click_train = True

    if click_train:
        ES_model_training(Run_in_Excel=False)
    else:
        ES_model_predict(Run_in_Excel=False)
