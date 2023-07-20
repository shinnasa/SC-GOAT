# %%
import pandas as pd
import numpy as np
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.single_table import CopulaGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
import xgboost as xgb
import time
import utilities
import random
import sys
import os
import warnings

warnings.filterwarnings('ignore')

# %% [markdown]
# # Load data
# Load data and create train test split from the smaller dataset that contains 10% of the full data

# %%
arguments = sys.argv
print('arguments: ', arguments)
assert len(arguments) > 3
print(os.getcwd())
data_set_name = arguments[1]
target = 'income'
if data_set_name == 'credit_card':
    target = 'Class'

optimization_itr = int(arguments[2])

encode = False

if len(arguments) > 3:
    encode = eval(arguments[3])

balanced = False
if len(arguments) > 4:
    balanced = eval(arguments[4])

prefix = ''
if data_set_name == 'credit_card':
    if balanced:
        prefix = 'balanced_'
    else:
        prefix = 'unbalanced_'

if encode:
    prefix =  "encoded_" + prefix 

print('data_set_name: ', data_set_name)
print('target: ', target)
print('optimization_itr: ', optimization_itr)
print('encode: ', encode)
print('balanced: ', balanced)
print('prefix: ', prefix)

data_set_name_temp = prefix + data_set_name

train = True

df_original = utilities.load_data_original(data_set_name, balanced)

# %%
df = df_original.copy()
if len(df) > 50000:
    df = df.sample(50000, replace = False, random_state = 5)
# %%

categorical_columns = []

for column in df.columns:
    if (column != target) & (df[column].dtype == 'object'):
        categorical_columns.append(column)

encoder = utilities.MultiColumnTargetEncoder(categorical_columns, target)

initial_columns_ordering = df.columns.values
print('initial_columns_ordering: ', initial_columns_ordering)

def get_train_validation_test_data(df, encode):
    df_train_original, df_test_original = train_test_split(df, test_size = 0.3,  random_state = 5) #70% is training and 30 to test
    df_test_original, df_val_original = train_test_split(df_test_original, test_size = 1 - 0.666,  random_state = 5)# out of 30, 20 is test and 10 for validation

    if encode:
        df_train = encoder.transform(df_train_original)

        # HERE ENCODING HAPPENS FOR ONLY FOR SYNTHESIZERS
        df_val = df_val_original
        df_test = df_test_original

        # HERE ENCODING HAPPENS FOR BOTH XGBOOST AND SYNTHESIZERS
        # df_val = encoder.transform_test_data(df_val_original)
        # df_test = encoder.transform_test_data(df_test_original)

        return df_train, df_val, df_test
    else:
        return df_train_original, df_val_original, df_test_original

class_0_ratio = len(df[df[target] == 0]) / len(df)
class_1_ratio = len(df[df[target] == 1]) / len(df)

df_train, df_val, df_test = get_train_validation_test_data(df, encode)

# print(len(df[df[target] == 0]) /len(df))
# print(len(df_train))
# print(len(df_test))

# utilities.save_test_train_data(data_set_name, df_train, df_test, balanced)
# %%
x_train = df_train.loc[:, df_train.columns != target]
y_train = df_train[target]

x_val= df_val.loc[:, df_val.columns != target]
y_val = df_val[target]

x_test = df_test.loc[:, df_test.columns != target]
y_test = df_test[target]

# %%
x_train

# %%
x_val

# %%
x_test

# %% [markdown]
# ## Create Supervised Synthesizers

# %%
params_xgb = {
        'eval_metric' : 'auc',
        'objective' : 'binary:logistic',
        'seed' : 5,
        'base_score' : len(df_train[df_train[target] == 1]) / len(df_train),
        'max_delta_step' : 1.0
        # 'scale_pos_weight' : len(df_train[df_train[target] == 0]) / len(df_train[df_train[target] == 1])
}

def fit_synth(df, method, epoch):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)
    if method == "GaussianCopula":
        synth = GaussianCopulaSynthesizer(metadata=metadata)
    elif method == "CTGAN" or method =="CopulaGAN":
        if method == "CTGAN":
            synth = CTGANSynthesizer(metadata=metadata, epochs=epoch)
        if method == "CopulaGAN":
            synth = CopulaGANSynthesizer(metadata=metadata, epochs=epoch)
    elif method == "TVAE":
        synth = TVAESynthesizer(metadata=metadata, epochs=epoch)
    else:
        raise ValueError("Invalid model name: " + method)
    return synth

def downstream_loss(sampled, df_val, target, classifier):
    x_samp = sampled.loc[:, sampled.columns != target]
    y_samp = sampled[target]
    x_val = df_val.loc[:, sampled.columns != target]
    y_val = df_val[target]

    if (sum(y_samp) / len(y_samp) > 0) & (sum(y_samp) / len(y_samp) < 1):
        params_xgb['base_score'] =  sum(y_samp) / len(y_samp)
    else:
        #HERE ONLY ONE CLASS PRESENTS IN THE DATA
        return 0, 0, None
    # params_xgb['base_score'] =  len(sampled[sampled[target] == 1]) / len(sampled)
    if classifier == "XGB":
        for column in x_samp.columns:
            if x_samp[column].dtype == 'object':
                x_samp[column] = x_samp[column].astype('category')
                x_val[column] = x_val[column].astype('category')
        dtrain = xgb.DMatrix(data=x_samp, label=y_samp, enable_categorical=True)
        dval = xgb.DMatrix(data=x_val, label=y_val, enable_categorical=True)
        clf = xgb.train(params_xgb, dtrain, 1000, verbose_eval=False)

        clf_probs_train = clf.predict(dtrain)
        clf_auc_train = roc_auc_score(y_samp.values.astype(float), clf_probs_train)
        clf_probs_val = clf.predict(dval)
        clf_auc_val = roc_auc_score(y_val.values.astype(float), clf_probs_val)
        return clf_auc_train, clf_auc_val, clf
    else:
        raise ValueError("Invalid classifier: " + classifier)

 
if train:
    # %% [markdown]
    # # Generate Synthetic Data from untuned models

    # %% [markdown]
    # ## GaussianCopula

    print("==========Executing GaussianCopula==========")

    #Starting time count
    start_time_GaussianCopula = time.time()

    # %%
    method = "GaussianCopula"
    epoch = 150
    synth = fit_synth(df_train, method, epoch)
    synth.fit(df_train)
    N_sim = 10000

    sampled_gaussain_copula = None

    if data_set_name_temp == 'unbalanced_credit_card':
        class0 = Condition(
            num_rows=round(N_sim * class_0_ratio),
            column_values={'Class': 0}
        )
        class1 = Condition(
            num_rows=round(N_sim * class_1_ratio),
            column_values={'Class': 1}
        )
        sampled_gaussain_copula = synth.sample_from_conditions(
            conditions=[class0, class1]
        )
    else:
        sampled_gaussain_copula = synth.sample(num_rows = N_sim)

    print('sampled: ', sampled_gaussain_copula)
    end_time_GaussianCopula = time.time()

    # HERE ENCODING HAPPENS FOR ONLY FOR SYNTHESIZERS
    if encode:
        sampled_gaussain_copula = encoder.inverse_transform(sampled_gaussain_copula)[initial_columns_ordering]

    # %%
    print(sampled_gaussain_copula.head())

    # %%
    x_gauss = sampled_gaussain_copula.loc[:, sampled_gaussain_copula.columns != target]
    y_gauss = sampled_gaussain_copula[target]  

    # %%
    sampled_gaussain_copula.to_csv('../data/' + data_set_name + "/" + prefix + data_set_name + "_sampled_untuned_gaussain_copula.csv", index=False)

    # %% [markdown]
    # ## CTGAN

    print("==========Executing CTGAN==========")

    start_time_CTGAN = time.time()

    # %%
    method = "CTGAN"
    epoch = 150
    synth = fit_synth(df_train, method, epoch)
    synth.fit(df_train)
    N_sim = 10000
    sampled_ct_gan = synth.sample(num_rows = N_sim)

    # if data_set_name_temp == 'unbalanced_credit_card':
    #     class0 = Condition(
    #         num_rows=round(N_sim * class_0_ratio),
    #         column_values={'Class': 0}
    #     )
    #     class1 = Condition(
    #         num_rows=round(N_sim * class_1_ratio),
    #         column_values={'Class': 1}
    #     )
    #     # conditions = pd.DataFrame.from_dict
    #     sampled_ct_gan = synth.sample_from_conditions(
    #         conditions=[class1], max_tries_per_batch = 10000, batch_size = 100 * N_sim
    #     )
    #     print('sampled: ', sampled_ct_gan)

    end_time_CTGAN = time.time()

    # HERE ENCODING HAPPENS FOR ONLY FOR SYNTHESIZERS
    if encode:
        sampled_ct_gan = encoder.inverse_transform(sampled_ct_gan)[initial_columns_ordering]

    # %%
    print(sampled_ct_gan.head())

    # %%
    x_ctgan = sampled_ct_gan.loc[:, sampled_ct_gan.columns != target]
    y_ctgan = sampled_ct_gan[target]   

    # %%
    sampled_ct_gan.to_csv('../data/' + data_set_name + "/" + prefix + data_set_name + "_sampled_untuned_ct_gan.csv", index=False)

    # %% [markdown]
    # ## CopulaGAN

    print("==========Executing CopulaGAN==========")

    start_time_CopulaGAN = time.time()

    # %%
    method = "CopulaGAN"
    epoch = 150
    synth = fit_synth(df_train, method, epoch)
    synth.fit(df_train)
    N_sim = 10000
    sampled_copula_gan = synth.sample(num_rows = N_sim)

    end_time_CopulaGAN = time.time()

    # HERE ENCODING HAPPENS FOR ONLY FOR SYNTHESIZERS
    if encode:
        sampled_copula_gan = encoder.inverse_transform(sampled_copula_gan)[initial_columns_ordering]

    # %%
    print(sampled_copula_gan.head())

    # %%
    x_copgan = sampled_copula_gan.loc[:, sampled_copula_gan.columns != target]
    y_copgan = sampled_copula_gan[target]       

    # %%
    sampled_copula_gan.to_csv('../data/' + data_set_name + "/" + prefix + data_set_name + "_sampled_untuned_copula_gan.csv", index=False)

    # %% [markdown]
    # ## TVAE

    print("==========Executing TVAE==========")

    start_time_TVAE = time.time()

    # %%
    method = "TVAE"
    epoch = 150
    synth = fit_synth(df_train, method, epoch)
    synth.fit(df_train)
    N_sim = 10000
    sampled_tvae= synth.sample(num_rows = N_sim)

    end_time_TVAE = time.time()
    
    # HERE ENCODING HAPPENS FOR ONLY FOR SYNTHESIZERS
    if encode:
        sampled_tvae = encoder.inverse_transform(sampled_tvae)[initial_columns_ordering]

    # %%
    print(sampled_tvae.head())

    # %%
    x_tvae = sampled_tvae.loc[:, sampled_tvae.columns != target]
    y_tvae = sampled_tvae[target]       

    # %%
    sampled_tvae.to_csv('../data/' + data_set_name + "/" + prefix + data_set_name + "_sampled_untuned_tvae.csv", index=False)

    # %% [markdown]
    # # Train Downstream Task

    # sampled_gaussain_copula = pd.read_csv('../data/' + data_set_name + "/" + prefix + data_set_name + "_sampled_untuned_gaussain_copula.csv")
    # sampled_ct_gan = pd.read_csv('../data/' + data_set_name + "/" + prefix + data_set_name + "_sampled_untuned_ct_gan.csv")
    # sampled_copula_gan = pd.read_csv('../data/' + data_set_name + "/" + prefix + data_set_name + "_sampled_untuned_copula_gan.csv")
    # sampled_tvae = pd.read_csv('../data/' + data_set_name + "/" + prefix + data_set_name + "_sampled_untuned_tvae.csv")

    # x_gauss = sampled_gaussain_copula.loc[:, sampled_gaussain_copula.columns != target]
    # y_gauss = sampled_gaussain_copula[target]  

    # x_ctgan = sampled_ct_gan.loc[:, sampled_ct_gan.columns != target]
    # y_ctgan = sampled_ct_gan[target]  

    # x_copgan= sampled_copula_gan.loc[:, sampled_copula_gan.columns != target]
    # y_copgan = sampled_copula_gan[target]  

    # x_tvae = sampled_tvae.loc[:, sampled_tvae.columns != target]
    # y_tvae = sampled_tvae[target]  

    # %%
    params_range = {
                'alpha_1':  hp.uniform('alpha_1', 0, 1),
                'alpha_2':  hp.uniform('alpha_2', 0, 1),
                'alpha_3':  hp.uniform('alpha_3', 0, 1),
                'alpha_4':  hp.uniform('alpha_4', 0, 1),
                'generated_data_size': 10000
            }
    
    # num_boost_round = 1000

    # %%
    def objective_maximize_roc(params):
        # Keep track of the best iteration records
        global output 
        global best_val_roc 
        global train_roc
        global best_params
        global best_X_synthetic
        global best_y_synthetic
        global best_classifier_model

        # Scale the alphas so that their sum adds up to 1
        alpha_temp = [params['alpha_1'], params['alpha_2'], params['alpha_3'], params['alpha_4']]
        scale = sum(alpha_temp)
        alpha = [(1 / scale) * alpha_temp[i] for i in range(len(alpha_temp))]
        index = np.argmax(alpha)
        params['alpha_1'] = alpha[0]
        params['alpha_2'] = alpha[1]
        params['alpha_3'] = alpha[2]
        params['alpha_4'] = alpha[3]

        # Combine all the data into a single list
        X_temp = [x_gauss, x_ctgan, x_copgan, x_tvae]
        y_temp = [y_gauss, y_ctgan, y_copgan, y_tvae]
        # Randomly select the data from each source
        random.seed = 5
        randomRows = random.sample(list(y_temp[0].index.values), int(alpha[0] * len(y_temp[0].index.values)))

        X_new = X_temp[0].loc[randomRows]
        y_new = y_temp[0].loc[randomRows].values

        x_val_real = x_val.copy()
        y_val_real = y_val.copy()

        generated_data_size = params['generated_data_size']

        size = [int(alpha[i] * len(y_temp[i].index.values)) for i in range(4)]
        size[index] += (generated_data_size - sum(size))
        
        # Randomly select the data from each source based on the alpha values
        for i in range(1, len(y_temp)):
            n = size[i]
            randomRows = random.sample(list(y_temp[i].index.values), n)
            X_new = pd.concat([X_new, X_temp[i].loc[randomRows]])
            y_new = np.concatenate((y_new, y_temp[i].loc[randomRows].values))


        X_synthetic = X_new.copy()
        y_synthetic = y_new.copy()
        
        # Train classifier
        for column in X_new.columns:
            if X_new[column].dtype == 'object':
                X_new[column] = X_new[column].astype('category')
                x_val_real[column] = x_val_real[column].astype('category')
        
        dtrain = xgb.DMatrix(data=X_new, label=y_new, enable_categorical=True)
        if (sum(y_new) / len(y_new) > 0) & (sum(y_new) / len(y_new) < 1): 
            params_xgb['base_score'] =  sum(y_new) / len(y_new)
        # params_xgb['scale_pos_weight'] =  sum(y_new) / (len(y_new) - sum(y_new))

        dval = xgb.DMatrix(data=x_val_real, label=y_val_real, enable_categorical=True)
        clf = xgb.train(params = params_xgb, dtrain=dtrain, verbose_eval=False)

        # Evaluate the performance of the classifier
        clf_probs = clf.predict(dval)
        clf_auc = roc_auc_score(y_val_real.astype(float), clf_probs)
        
        clf_probs_train = clf.predict(dtrain)
        clf_auc_train = roc_auc_score(y_new.astype(float), clf_probs_train)
        params['train_roc']        = clf_auc_train
        params['val_roc']        = clf_auc

        output = output._append(params, ignore_index = True)
        
        # Update best record of the loss function and the alpha values based on the optimization
        if params['val_roc'] > best_val_roc:
            best_val_roc = params['val_roc']
            train_roc = params['train_roc']
            best_params = alpha
            best_X_synthetic = X_synthetic
            best_y_synthetic = y_synthetic
            best_classifier_model = clf
        
        # Loss function is to maximize the test roc score
        return {
            'loss' : 1 - clf_auc,
            'status' : STATUS_OK,
            'eval_time ': time.time(),
            'test_roc' : clf_auc,
            }

    # %%
    def trainDT(max_evals:int):
        # Keep track of the best iteration records
        global output 
        output = pd.DataFrame()
        global train_roc
        global best_val_roc
        global best_params
        global best_X_synthetic
        global best_y_synthetic
        global best_classifier_model
        best_val_roc = 0
        train_roc = 0
        best_params = []
        trials = Trials()
        start_time_BO = time.time()
        clf_best_param = fmin(fn=objective_maximize_roc,
                        space=params_range,
                        max_evals=max_evals,
                    # rstate=np.random.default_rng(42),
                        algo=tpe.suggest,
                        trials=trials)
        print(clf_best_param)
        end_time_BO = time.time()
        total_time_BO = end_time_BO - start_time_BO
        return best_val_roc, train_roc, best_params, best_X_synthetic, best_y_synthetic, clf_best_param, output, total_time_BO, best_classifier_model

    # %%

    print("==========Executing Baseyan Optimization==========")

    best_val_roc, train_roc, best_params, best_X_synthetic, best_y_synthetic, clf_best_param, params_history, total_time_BO, best_classifier_model = trainDT(optimization_itr)

    end_time = time.time()

    # %%
    best_val_roc

    # %%
    best_y_synthetic

    # %%
    best_X_synthetic

    # %%
    best_params

    # %%
    params_history

    # %%
    def save_synthetic_data(data_set_name:str, best_X_synthetic, best_y_synthetic, balanced:bool=False):
        synthetic_data = best_X_synthetic
        prefix = ''
        if data_set_name == 'credit_card':
            if balanced:
                prefix = 'balanced_'
            else:
                prefix = 'unbalanced_'

        if encode:
            prefix =  "encoded_" + prefix 
        if data_set_name == 'adult':
            target = 'income'
            synthetic_data[target] = best_y_synthetic
            synthetic_data.loc[synthetic_data[target] == True, target] = " <=50K"
            synthetic_data.loc[synthetic_data[target] == False, target] = " >50K"
            synthetic_data.to_csv("../data/output/" + prefix + data_set_name + "_untuned_models_synthetic_data_xgboost.csv", index=False)
        elif data_set_name == 'credit_card':
            target = 'Class'
            synthetic_data[target] = best_y_synthetic
            synthetic_data.to_csv("../data/output/" + prefix + data_set_name + "_untuned_models_synthetic_data_xgboost.csv", index=False)
        else:
            raise ValueError("Invalid data set name: " + data_set_name)
        return synthetic_data

    # HERE ENCODING HAPPENS FOR BOTH XGBOOST AND SYNTHESIZERS
    # if encode:
    #     best_X_synthetic = encoder.inverse_transform(best_X_synthetic)

    # %%
    save_synthetic_data(data_set_name, best_X_synthetic, best_y_synthetic, balanced)

    #Compute test ROC
    for column in x_test.columns:
        if x_test[column].dtype == 'object':
            x_test[column] = x_test[column].astype('category')
    dtest = xgb.DMatrix(data=x_test, label=y_test, enable_categorical=True)
    clf_probs = best_classifier_model.predict(dtest)
    test_roc = roc_auc_score(y_test.astype(float), clf_probs)

    print('train_roc: ', train_roc)
    print('best_val_roc: ', best_val_roc)
    print('test_roc: ', test_roc)
    clf_best_param["train_roc"] = train_roc
    clf_best_param["val_roc"] = best_val_roc
    clf_best_param["test_roc"] = test_roc
    clf_best_param["total_time_GaussianCopula"] = end_time_GaussianCopula - start_time_GaussianCopula
    clf_best_param["total_time_CTGAN"] = end_time_CTGAN - start_time_CTGAN
    clf_best_param["total_time_CopulaGAN"] = end_time_CopulaGAN - start_time_CopulaGAN
    clf_best_param["total_time_TVAE"] = end_time_TVAE - start_time_TVAE
    clf_best_param["total_time_BO"] = total_time_BO

    print('clf_best_param: ', clf_best_param)

    clf_best_param_df = pd.DataFrame()
    clf_best_param_df = clf_best_param_df._append(clf_best_param, ignore_index = True)
    clf_best_param_df.to_csv("../data/output/" + prefix + data_set_name + "_untuned_models_clf_best_param_xgboost.csv", index=False)

    params_history.to_csv("../data/history/" + prefix + data_set_name + "_untuned_models_params_alpha_history.csv", index=False)

else:

    params_xgb = {
        'eval_metric' : 'auc',
        'objective' : 'binary:logistic',
        'seed' : 5,
        'max_delta_step' : 1.0
        # 'scale_pos_weight' : len(df_train[df_train[target] == 0]) / len(df_train[df_train[target] == 1])
    }

    print("==========Computing XGBOOST Perforamce on " + data_set_name + "==========")
    sampled_gaussain_copula = pd.read_csv('../data/' + data_set_name + "/" + prefix + data_set_name + "_sampled_untuned_gaussain_copula.csv")
    sampled_ct_gan = pd.read_csv('../data/' + data_set_name + "/" + prefix + data_set_name + "_sampled_untuned_ct_gan.csv")
    sampled_copula_gan = pd.read_csv('../data/' + data_set_name + "/" + prefix + data_set_name + "_sampled_untuned_copula_gan.csv")
    sampled_tvae = pd.read_csv('../data/' + data_set_name + "/" + prefix + data_set_name + "_sampled_untuned_tvae.csv")

    print(len(sampled_gaussain_copula[sampled_gaussain_copula[target] == 0]) /len(sampled_gaussain_copula))
    print(len(sampled_ct_gan[sampled_ct_gan[target] == 0]) /len(sampled_ct_gan))
    print(len(sampled_copula_gan[sampled_copula_gan[target] == 0]) /len(sampled_copula_gan))
    print(len(sampled_tvae[sampled_tvae[target] == 0]) /len(sampled_tvae))

    synthetic_data = pd.read_csv("../data/output/" + prefix + data_set_name + "_untuned_models_synthetic_data_xgboost.csv")
    if data_set_name == 'adult':
        print(len(synthetic_data[synthetic_data[target] == ' >50K']) /len(synthetic_data))
        print(len(synthetic_data[synthetic_data[target] == ' <=50K']) /len(synthetic_data))
    else:
        # print(synthetic_data.columns)
        print(len(synthetic_data[synthetic_data[target] == 0]) /len(synthetic_data))

    start_time_GaussianCopula = time.time()
    clf_auc_train_gaussain_copula, clf_auc_val_gaussain_copula, clf_gaussain_copula = downstream_loss(sampled_gaussain_copula, df_val, target, 'XGB')
    end_time_GaussianCopula = time.time()
    start_time_CTGAN = time.time()
    clf_auc_train_ct_gan, clf_auc_val_ct_gan, clf_ct_gan= downstream_loss(sampled_ct_gan, df_val, target, 'XGB')
    end_time_CTGAN = time.time()
    start_time_CopulaGAN = time.time()
    clf_auc_train_copula_gan, clf_auc_val_copula_gan, clf_copula_gan = downstream_loss(sampled_copula_gan, df_val, target, 'XGB')
    end_time_CopulaGAN = time.time()
    start_time_TVAE = time.time()
    clf_auc_train_tvae, clf_auc_val_tvae, clf_tvae = downstream_loss(sampled_tvae, df_val, target, 'XGB')
    end_time_TVAE = time.time()

    for column in x_test.columns:
        if x_test[column].dtype == 'object':
            x_test[column] = x_test[column].astype('category')
    dtest = xgb.DMatrix(data=x_test, label=y_test, enable_categorical=True)
    
    clf_auc_test_gaussain_copula = 0
    if clf_gaussain_copula != None:
        clf_probs_test_gaussain_copula = clf_gaussain_copula.predict(dtest)
        clf_auc_test_gaussain_copula = roc_auc_score(y_test.astype(float), clf_probs_test_gaussain_copula)

    clf_auc_test_ct_gan = 0
    if clf_ct_gan != None:
        clf_probs_test_ct_gan = clf_ct_gan.predict(dtest)
        clf_auc_test_ct_gan = roc_auc_score(y_test.astype(float), clf_probs_test_ct_gan)

    clf_auc_test_copula_gan = 0
    if clf_copula_gan != None:
        clf_probs_test_copula_gan = clf_copula_gan.predict(dtest)
        clf_auc_test_copula_gan = roc_auc_score(y_test.astype(float), clf_probs_test_copula_gan)

    clf_auc_test_tvae = 0
    if clf_tvae != None:
        clf_probs_test_tvae = clf_tvae.predict(dtest)
        clf_auc_test_tvae = roc_auc_score(y_test.astype(float), clf_probs_test_tvae)

    individual_clf_auc = {'clf_auc_train_gaussain_copula' : clf_auc_train_gaussain_copula,
                        'clf_auc_val_gaussain_copula' : clf_auc_val_gaussain_copula,
                        'clf_auc_test_gaussain_copula' : clf_auc_test_gaussain_copula,
                        'clf_auc_train_ct_gan' : clf_auc_train_ct_gan,
                        'clf_auc_val_ct_gan' : clf_auc_val_ct_gan,
                        'clf_auc_test_ct_gan' : clf_auc_test_ct_gan, 
                        'clf_auc_train_copula_gan' : clf_auc_train_copula_gan,
                        'clf_auc_val_copula_gan' : clf_auc_val_copula_gan,
                        'clf_auc_test_copula_gan' : clf_auc_test_copula_gan,
                        'clf_auc_train_tvae' : clf_auc_train_tvae,
                        'clf_auc_val_tvae' : clf_auc_val_tvae,
                        'clf_auc_test_tvae' : clf_auc_test_tvae,
                        'train' : len(sampled_tvae),
                        'val' : len(df_val),
                        'test' : len(df_test)}

    individual_clf_auc["total_time_GaussianCopula"] = end_time_GaussianCopula - start_time_GaussianCopula
    individual_clf_auc["total_time_CTGAN"] = end_time_CTGAN - start_time_CTGAN
    individual_clf_auc["total_time_CopulaGAN"] = end_time_CopulaGAN - start_time_CopulaGAN
    individual_clf_auc["total_time_TVAE"] = end_time_TVAE - start_time_TVAE

    print('individual_clf_auc: ', individual_clf_auc)

    individual_clf_auc_df = pd.DataFrame()
    individual_clf_auc_df = individual_clf_auc_df._append(individual_clf_auc, ignore_index = True)

    individual_clf_auc_df.to_csv("../data/output/" + prefix + data_set_name + "_untuned_models_clf_auc_score_and_time_per_each_individual_model.csv", index=False)

# %%
