# -*- coding: utf-8 -*-
"""
@author: Hossein Bagheri and Mohammad Ganjirad
Gmail: h.bagheri.en@gmail.com
Description: 
Python 3, (env: base)
"""

from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd 
from sklearn import preprocessing
import os
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, f1_score,recall_score,roc_curve,auc,RocCurveDisplay
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import time
import warnings
#########################################################################################
def hyperopt(param_space, X_train, y_train, num_eval):
    
    start = time.time()
    
    def objective_function(params):
        reg = XGBClassifier(**params)
        score = cross_val_score(reg, Xtrain, ytrain, cv=5).mean()
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best_param = fmin(objective_function, 
                      param_space, 
                      algo=tpe.suggest, 
                      max_evals=num_eval, 
                      trials=trials,
                      rstate= np.random.RandomState(1))
    loss = [x['result']['loss'] for x in trials.trials]
    

   
    
    Classifier_best = XGBClassifier(
                            eta=best_param['eta'],
                            max_depth=int(best_param['max_depth']),
                            n_estimators=int(best_param['n_estimators']),
                            min_child_weight=best_param['min_child_weight'],
                            colsample_bytree=best_param['colsample_bytree'],
                            subsample=best_param['subsample'],
                            gamma=best_param['gamma'],
                            reg_lambda=best_param['reg_lambda']
                            )
    Classifier_best.fit(Xtrain, ytrain)
    TimeElapsed= time.time() - start
    print("")
    print("##### Results")
    print("Score best parameters: ", min(loss)*-1)
    print("Best parameters: ", best_param)
    print("Time elapsed: ",TimeElapsed)
    print("Parameter combinations evaluated: ", num_eval)
    
    return trials, Classifier_best,TimeElapsed

#########################################################################################
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)
#########################################################################################
path = r'.\FinalDS_Clean.csv'
df = pd.read_csv(path)
#########################################################################################
Featuers_S1=['CR','DPSVI','DPSVIo','Sigma0_VH','Sigma0_VV',
          'C_CR','C_DPSVI','C_DPSVIo','C_Sigma0_VH','C_Sigma0_VV'];

Featuers_S2=['NR' ,'GSAVI' ,'SWIR21' ,'S1G', 'B8A' ,'NDVI', 'S1N',
 'S1R','NBR','B11','B12','SAVI','S2G','MSR','MSAVI',
 'S2N','B2','GDVII','B3','S2R' ,'NDII', 'B4' ,'B5', 'B6' ,'RG' ,'B7' ,'DVI' ,'B8',
 'CVI','NG','NDGI','GNDVI','C_B11','C_B12','C_B2','C_B3','C_B4','C_B5','C_B6'
 ,'C_B7','C_B8','C_B8A','C_CVI','C_DVI','C_GDVII','C_GNDVI','C_GSAVI',
 'C_MSAVI','C_MSR','C_NBR','C_NDGI','C_NDII','C_NDVI','C_NG','C_NR' ,'C_RG',
 'C_S1G' ,'C_S1N' ,'C_S1R' ,'C_S2G' ,'C_S2N', 'C_S2R' ,'C_SAVI' ,'C_SWIR21'];

Featuers_Canopy=['Median','Mean','STD','MAX','MIN','Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10']

Featuers_S1S2=['NR' ,'GSAVI' ,'SWIR21' ,'S1G', 'B8A' ,'NDVI', 'S1N',
 'S1R','NBR','B11','B12','SAVI','S2G','MSR','MSAVI',
 'S2N','B2','GDVII','B3','S2R' ,'NDII', 'B4' ,'B5', 'B6' ,'RG' ,'B7' ,'DVI' ,'B8',
 'CVI','NG','NDGI','GNDVI','CR','DPSVI','DPSVIo','Sigma0_VH','Sigma0_VV',
 'C_B11','C_B12','C_B2','C_B3','C_B4','C_B5','C_B6','C_B7','C_B8','C_B8A',
 'C_CR','C_CVI','C_DPSVI','C_DPSVIo','C_DVI','C_GDVII','C_GNDVI','C_GSAVI',
 'C_MSAVI','C_MSR','C_NBR','C_NDGI','C_NDII','C_NDVI','C_NG','C_NR' ,'C_RG',
 'C_S1G' ,'C_S1N' ,'C_S1R' ,'C_S2G' ,'C_S2N', 'C_S2R' ,'C_SAVI' ,'C_Sigma0_VH'
 ,'C_Sigma0_VV' ,'C_SWIR21'];

Featuers_S1S2Canopy=['NR' ,'GSAVI' ,'SWIR21' ,'S1G', 'B8A' ,'NDVI', 'S1N',
 'S1R','NBR','B11','B12','SAVI','S2G','MSR','MSAVI',
 'S2N','B2','GDVII','B3','S2R' ,'NDII', 'B4' ,'B5', 'B6' ,'RG' ,'B7' ,'DVI' ,'B8',
 'CVI','NG','NDGI','GNDVI','CR','DPSVI','DPSVIo','Sigma0_VH','Sigma0_VV',
 'C_B11','C_B12','C_B2','C_B3','C_B4','C_B5','C_B6','C_B7','C_B8','C_B8A',
 'C_CR','C_CVI','C_DPSVI','C_DPSVIo','C_DVI','C_GDVII','C_GNDVI','C_GSAVI',
 'C_MSAVI','C_MSR','C_NBR','C_NDGI','C_NDII','C_NDVI','C_NG','C_NR' ,'C_RG',
 'C_S1G' ,'C_S1N' ,'C_S1R' ,'C_S2G' ,'C_S2N', 'C_S2R' ,'C_SAVI' ,'C_Sigma0_VH'
 ,'C_Sigma0_VV' ,'C_SWIR21','Median','Mean','STD','MAX','MIN','Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10']
#########################################################################################
TargetPmt=['l0','l1','l2','l3']
Mode=['S1','S2','Canopy','S1S2','S1S2Canopy']
#########################################################################################
for i in TargetPmt:
    Df_Test=pd.DataFrame({i:['Accuracy','Kappa','PW','PMI','PMA','RW','RMI','RW','F1W','F1MI','F1MA','TimesElapsed']})
    Df_Train=pd.DataFrame({i:['Accuracy','Kappa','PW','RW','F1W']})
    
    for j in Mode:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(f"Stage {j} ,{i}")
        ###############################################
        if j=='S1': featuers=Featuers_S1
        if j=='S2': featuers=Featuers_S2
        if j=='Canopy': featuers=Featuers_Canopy
        if j=='S1S2': featuers=Featuers_S1S2 
        if j=='S1S2Canopy': featuers=Featuers_S1S2Canopy  
        ################################################
        X=df[featuers]
        print(X.columns)
        y = df[[i]]
        y = y.to_numpy()

        scaler = MinMaxScaler()
        Xs = scaler.fit_transform(X)
        print(Xs.shape)
        print(y.shape)



        Xtrain = Xs[df['SPLIT'] == 'train']
        ytrain = y[df['SPLIT'] == 'train']

        Xtest = Xs[df['SPLIT'] == 'test']
        ytest = y[df['SPLIT'] == 'test']


        #  For the New Version of XGB
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        ytrain = le.fit_transform(ytrain)
        ytest = le.fit_transform(ytest)


        param_hyperopt= {
            # learning rate
        'eta': hp.loguniform('eta', np.log(0.01), np.log(1)),
        'max_depth': scope.int(hp.quniform('max_depth', 5, 35, 1)),
        'n_estimators' : scope.int(hp.quniform('n_estimators', 2000, 2500, 100)),
        'min_child_weight' : hp.quniform('min_child_weight', 1, 10, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'gamma': hp.uniform('gamma', 0.0, 10.0),
        'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0)
        }

        results_hyperopt, calssifier_best,TimeElapsed = hyperopt(param_hyperopt, Xtrain, ytrain, 20)
        ########################################################################################
        # Prediction (Training)
        y_train_pred=calssifier_best.predict(Xtrain)
        # Calculate metrics (Train)
        accuracyTrain = accuracy_score(ytrain, y_train_pred)
        kappaTrain = cohen_kappa_score(ytrain, y_train_pred)
        precisionTrain = precision_score(ytrain, y_train_pred, average='weighted')
        recallTrain=recall_score(ytrain, y_train_pred, average='weighted')
        f1Train = f1_score(ytrain, y_train_pred, average='weighted')
        print("Accuracy (Train):", accuracyTrain)
        print("Kappa (Train):", kappaTrain)
        print("Precision (Train):", precisionTrain)
        print("Recall (Train):", recallTrain)
        print("F1 Score (Train):", f1Train)
        Computed_Metric_Train=[accuracyTrain,kappaTrain,precisionTrain,recallTrain,f1Train]
        #########################################################################################
        # make predictions on the test set
        y_test_pred= calssifier_best.predict(Xtest)
        accuracy = accuracy_score(ytest, y_test_pred)
        kappa = cohen_kappa_score(ytest, y_test_pred)
        precision_weighted = precision_score(ytest, y_test_pred, average='weighted')
        precision_micro =precision_score(ytest, y_test_pred, average='micro')
        precision_macro =precision_score(ytest, y_test_pred, average='macro')
        recall_wighted=recall_score(ytest, y_test_pred, average='weighted')
        recall_micro=recall_score(ytest, y_test_pred, average='micro')
        recall_macro=recall_score(ytest, y_test_pred, average='macro')
        f1_weighted =f1_score(ytest, y_test_pred, average='weighted')
        f1_micro =f1_score(ytest, y_test_pred, average='micro')
        f1_macro=f1_score(ytest, y_test_pred, average='macro')
        print("########################")
        print("Accuracy (Test):", accuracy)
        print("Kappa (Test):", kappa)
        print("Precision (Test):", precision_weighted)
        print("Recall (Test):", recall_wighted)
        print("F1 Score (Test):", f1_weighted)
        Computed_Metric_Test=[accuracy,kappa,
            precision_weighted,precision_micro,precision_macro,
            recall_wighted,recall_micro,recall_macro,
            f1_weighted,f1_micro,f1_macro,TimeElapsed]
        #########################################################################################
        Df_Test.insert(len(Df_Test.columns),j,Computed_Metric_Test)
        Df_Train.insert(len(Df_Train.columns),j,Computed_Metric_Train)
        #########################################################################################
    Df_Test.to_csv(f'XGB_Test_{i}.csv')
    Df_Train.to_csv(f'XGB_Train_{i}.csv')
    #########################################################################################
    
print("Finished")
    