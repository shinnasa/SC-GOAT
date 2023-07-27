import pandas as pd
import numpy as np
from scipy.special import softmax
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.fmin import generate_trials_to_calculate
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
random.seed(5)
warnings.filterwarnings('ignore')

################################################################################################
# Get user defined values
################################################################################################
arguments = sys.argv
print('arguments: ', arguments)
print(os.getcwd())

if len(arguments) > 2:
    encode = False
    data_set_name = arguments[1]
    
    optimization_itr = int(arguments[2])
    encode = eval(arguments[3])

    balanced = False
    if len(arguments) > 4:
        balanced = eval(arguments[4])

    tuned = False
    if len(arguments) > 5:
        tuned = eval(arguments[5])

    augment_data_percentage = 0
    augment_data = False
    if len(arguments) > 6:
        augment_data = True
        augment_data_percentage = float(arguments[6])
        assert 0 <= augment_data_percentage <= 1
else:
    data_set_name = 'adult'
    optimization_itr = 100
    encode = True
    balanced = False
    tuned = True
    augment_data = False
    augment_data_percentage = 0

target = 'income'
if data_set_name == 'credit_card':
    target = 'Class'

prefix = ''
if data_set_name == 'credit_card':
    if balanced:
        prefix = 'balanced_'
    else:
        prefix = 'unbalanced_'

outpath = "data/output/ES10/"
outpath_data_augmentation = "data/outputDataAugmentation/"
historypath_data_augmentation = "data/historyDataAugmentation/"

print('data_set_name: ', data_set_name)
print('target: ', target)
print('optimization_itr: ', optimization_itr)
print('encode: ', encode)
print('balanced: ', balanced)
print('tuned: ', tuned)
print('augment_data: ', augment_data)
print('augment_data_percentage: ', augment_data_percentage)
print('prefix: ', prefix)

################################################################################################
# Load data
################################################################################################
df_original = utilities.load_data_original(data_set_name, balanced)

df = df_original.copy()
if len(df) > 50000:
    df = df.sample(50000, replace = False)

categorical_columns = []

for column in df.columns:
    if (column != target) & (df[column].dtype == 'object'):
        categorical_columns.append(column)

initial_columns_ordering = df.columns.values


################################################################################################
# Define functions. These will be moved to utilities.py eventually
################################################################################################
def get_train_validation_test_data(df, encode):
    df_train_original, df_test_original = train_test_split(df, test_size = 0.3) #70% is training and 30 to test
    df_val_original, df_test_original = train_test_split(df_test_original, test_size = 1 - 0.666)# out of 30, 20 is test and 10 for validation
    if encode:
        # HERE ENCODING HAPPENS FOR ONLY FOR SYNTHESIZERS
        df_train = encoder.transform(df_train_original)
        df_val = df_val_original
        df_test = df_test_original

        # HERE ENCODING HAPPENS FOR BOTH XGBOOST AND SYNTHESIZERS
        # df_train = encoder.transform(df_train_original)
        # df_val = encoder.transform_test_data(df_val_original)
        # df_test = encoder.transform_test_data(df_test_original)

        return df_train, df_val, df_test
    else:
        return df_train_original, df_val_original, df_test_original

# df_train, df_val, df_test = get_train_validation_test_data(df, encode)

# df_train_original = pd.read_csv("data/input/" + prefix + data_set_name + "_train.csv", index_col=0)
# df_val_original = pd.read_csv("data/input/" + prefix + data_set_name + "_validation.csv", index_col=0)
# df_test_original = pd.read_csv("data/input/" + prefix + data_set_name + "_test.csv", index_col=0)

# df_real = df_train.sample(10000, replace = False)

# categorical_columns = []

# for column in df_train_original.columns:
#     if (column != target) & (df_train_original[column].dtype == 'object'):
#         categorical_columns.append(column)

# initial_columns_ordering = df_train_original.columns.values
# encoder = utilities.MultiColumnTargetEncoder(categorical_columns, target)

if encode:
    prefix =  "encoded_" + prefix 

# df_train = df_train_original.copy()
# df_val = df_val_original.copy()
# df_test = df_test_original.copy()
# if encode:
#     df_train = encoder.transform(df_train_original)


params_xgb = {
        'eval_metric' : 'auc',
        'objective' : 'binary:logistic',
        'seed': 5,
        # 'base_score' :  len(df_train[df_train[target] == 1]) / len(df_train)
}

def downstream_loss(sampled, df_val, target, classifier):
    x_samp = sampled.loc[:, sampled.columns != target]
    y_samp = sampled[target]
    x_val = df_val.loc[:, sampled.columns != target]
    y_val = df_val[target]

    if (sum(y_samp) / len(y_samp) > 0) & (sum(y_samp) / len(y_samp) < 1):
        pass
    else:
        #HERE ONLY ONE CLASS PRESENTS IN THE DATA
        return 0.5, 0.5, None
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

sampled_gaussain_copula = None
sampled_ct_gan = None
sampled_copula_gan = None
sampled_tvae = None

if encode:
    m_name = prefix + data_set_name
else:
    m_name = prefix + data_set_name

if tuned:
    tuning = "_untuned_"
else:
    tuning = "_tuned_"
N_sim = 10000

# sampled_gaussain_copula = pd.read_csv(outpath + m_name + "_untuned_GaussianCopula_synthetic_data_xgboost.csv")
# sampled_ct_gan = pd.read_csv(outpath + m_name + tuning + "CTGAN_synthetic_data_xgboost.csv")
# sampled_copula_gan = pd.read_csv(outpath + m_name + tuning + "CopulaGAN_synthetic_data_xgboost.csv")
# sampled_tvae = pd.read_csv(outpath + m_name + tuning + "TVAE_synthetic_data_xgboost.csv")

# print(len(sampled_gaussain_copula[sampled_gaussain_copula[target] == 0]) /len(sampled_gaussain_copula))
# print(len(sampled_ct_gan[sampled_ct_gan[target] == 0]) /len(sampled_ct_gan))
# print(len(sampled_copula_gan[sampled_copula_gan[target] == 0]) /len(sampled_copula_gan))
# print(len(sampled_tvae[sampled_tvae[target] == 0]) /len(sampled_tvae))


# x_gauss = sampled_gaussain_copula.loc[:, sampled_gaussain_copula.columns != target]
# y_gauss = sampled_gaussain_copula[target]
# x_ctgan = sampled_ct_gan.loc[:, sampled_ct_gan.columns != target]
# y_ctgan = sampled_ct_gan[target]
# x_copgan = sampled_copula_gan.loc[:, sampled_copula_gan.columns != target]
# y_copgan = sampled_copula_gan[target]
# x_tvae = sampled_tvae.loc[:, sampled_tvae.columns != target]
# y_tvae = sampled_tvae[target]


def getParams(augment_data_percentage:float):
    # print('----augment_data_percentage: ', augment_data_percentage)

    if augment_data_percentage > 0:
        params_range = {
            'alpha_1':  hp.uniform('alpha_1', 0, 1),
            'alpha_2':  hp.uniform('alpha_2', 0, 1),
            'alpha_3':  hp.uniform('alpha_3', 0, 1),
            'alpha_4':  hp.uniform('alpha_4', 0, 1),
            'alpha_5':  augment_data_percentage,
            'generated_data_size': 10000
           } 
        return params_range
    else:
        params_range = {
            'alpha_1':  hp.uniform('alpha_1', 0, 1),
            'alpha_2':  hp.uniform('alpha_2', 0, 1),
            'alpha_3':  hp.uniform('alpha_3', 0, 1),
            'alpha_4':  hp.uniform('alpha_4', 0, 1),
            'alpha_5':  0,
            'generated_data_size': 10000
           } 
        return params_range

generated_data_size = 10000


#Compute initial aphpas for warm start
lmname =['GaussianCopula', 'CTGAN', 'CopulaGAN', 'TVAE']
# def computeInitialAlphaForWarmStart(augment_data_percentage:float):
#     val_auc = []
#     for method_name in lmname:
#         fname = outpath + m_name + "_" + method_name + "_clf_best_param_xgboost.csv"
#         if not os.path.exists(fname):
#             print(f"File '{fname}' does not exist. Skipping...")
#             continue
#         df = pd.read_csv(fname, index_col=0)
#         luntuned = [df.loc['untuned_val_roc', 'Value']]
#         if len(val_auc) == 0:
#             val_auc = luntuned
#         else:
#             val_auc.append(float(luntuned[0]))
        
#     val_auc_temp = [val_auc[i] - min(val_auc) + 1e-3 for i in range(len(val_auc))]
#     scale = sum(val_auc_temp) / (1 - augment_data_percentage)
#     alphas = [(1 / scale) * val_auc_temp[i] for i in range(len(val_auc_temp))]
        
#     return alphas

def computeInitialAlphaForWarmStart(augment_data_percentage:float, val_auc):
    # val_auc = []
    # for method_name in lmname:
    #     fname = outpath + m_name + "_" + method_name + "_clf_best_param_xgboost.csv"
    #     if not os.path.exists(fname):
    #         print(f"File '{fname}' does not exist. Skipping...")
    #         continue
    #     df = pd.read_csv(fname, index_col=0)
    #     luntuned = [df.loc['untuned_val_roc', 'Value']]
    #     if len(val_auc) == 0:
    #         val_auc = luntuned
    #     else:
    #         val_auc.append(float(luntuned[0]))
    
    val_auc_temp = [val_auc[i] - min(val_auc) + 1e-3 for i in range(len(val_auc))]
    scale = sum(val_auc_temp) / (1 - augment_data_percentage)
    alphas = [(1 / scale) * val_auc_temp[i] for i in range(len(val_auc_temp))]
        
    return alphas

def objective_maximize_roc(params):
    # Keep track of the best iteration records
    global output 
    global best_val_roc 
    global train_roc
    global best_params
    global best_X_synthetic
    global best_y_synthetic
    global best_classifier_model
    global X_temp
    global y_temp
    # Scale the alphas so that their sum adds up to 1
    alpha_temp = [params['alpha_1'], params['alpha_2'], params['alpha_3'], params['alpha_4'], params['alpha_5']]
    # alpha_temp = [params['alpha_1'], params['alpha_2'], params['alpha_3'], params['alpha_4']]
    # alpha_modified = [alpha_temp[i] - min(alpha_temp) + 1e-3 for i in range(len(alpha_temp))]
    scale = (sum(alpha_temp)-augment_data_percentage)  / (1 - augment_data_percentage)
    # alpha = softmax(alpha_temp)
    # alpha = [(1 / scale) * alpha_modified[i] for i in range(len(alpha_modified))]
    alpha = [(1 / scale) * alpha_temp[i] for i in range(len(alpha_temp) - 1)]
    alpha.append(augment_data_percentage)
    # print("Scaled alpha: ",alpha)
    # print("Sum of alphas: ", sum(alpha))
    index = np.argmax(alpha)
    params['alpha_1'] = alpha[0]
    params['alpha_2'] = alpha[1]
    params['alpha_3'] = alpha[2]
    params['alpha_4'] = alpha[3]
    params['alpha_5'] = alpha[4]

    # print('alpha: ', alpha)
    # print('alpha: ', sum(alpha))

    # Combine all the data into a single list
    # X_temp = [x_gauss, x_ctgan, x_copgan, x_tvae, x_real]
    # y_temp = [y_gauss, y_ctgan, y_copgan, y_tvae, y_real]
    
    # Randomly select the data from each source
    randomRows = random.sample(list(y_temp[0].index.values), int(alpha[0] * len(y_temp[0].index.values)))

    X_new = X_temp[0].loc[randomRows]
    y_new = y_temp[0].loc[randomRows].values

    x_val_real = x_val.copy()
    y_val_real = y_val.copy()

    generated_data_size = params['generated_data_size']

    size = [int(alpha[i] * len(y_temp[i].index.values)) for i in range(len(alpha))]
    size[index] += (generated_data_size - sum(size))
    
    # Randomly select the data from each source based on the alpha values
    for i in range(1, len(y_temp)):
        n = size[i]
        if n > 0:
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
    dval = xgb.DMatrix(data=x_val_real, label=y_val_real, enable_categorical=True)
    clf = xgb.train(params = params_xgb, dtrain=dtrain, verbose_eval=False)

    # Evaluate the performance of the classifier
    clf_probs = clf.predict(dval)
    clf_auc = roc_auc_score(y_val_real.astype(float), clf_probs)
    
    clf_auc_train = 0.5
    if (sum(y_new) / len(y_new) > 0) & (sum(y_new) / len(y_new) < 1):
        clf_probs_train = clf.predict(dtrain)
        clf_auc_train = roc_auc_score(y_new.astype(float), clf_probs_train)
    params['train_roc']        = clf_auc_train
    params['val_roc']        = clf_auc

    output = output._append(params, ignore_index = True)
    
    # Update best record of the loss function and the alpha values based on the optimization
    if params['val_roc'] > best_val_roc:
        best_val_roc = params['val_roc']
        train_roc = params['train_roc']
        best_params = params
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

def trainDT(max_evals:int, X_temp_arg, y_temp_arg, val_auc_arg):
    # Keep track of the best iteration records
    global output 
    output = pd.DataFrame()
    global train_roc
    global best_val_roc
    global best_params
    global best_X_synthetic
    global best_y_synthetic
    global best_classifier_model
    global X_temp
    global y_temp
    global val_auc
    X_temp = X_temp_arg
    y_temp = y_temp_arg
    val_auc = val_auc_arg
    best_val_roc = 0
    train_roc = 0
    best_params = []
    initial_alphas = computeInitialAlphaForWarmStart(augment_data_percentage, val_auc)
    initial_alphas_dict = {'alpha_1' : initial_alphas[0], 
                           'alpha_2' : initial_alphas[1], 
                           'alpha_3' : initial_alphas[2] ,
                           'alpha_4' : initial_alphas[3], 
                           'alpha_5' : augment_data_percentage
                           }
    trials = generate_trials_to_calculate([initial_alphas_dict])
    start_time_BO = time.time()
    params_range = getParams(augment_data_percentage)
    # print('getParams: ', getParams(augment_data_percentage))
    clf_best_param = fmin(fn=objective_maximize_roc,
                    space=params_range,
                    max_evals=max_evals,
                # rstate=np.random.default_rng(42),
                    algo=tpe.suggest,
                    trials=trials)
    # print('best_params: ', best_params)
    # print('clf_best_param: ', clf_best_param)
    end_time_BO = time.time()
    total_time_BO = end_time_BO - start_time_BO
    return best_val_roc, train_roc, best_params, best_X_synthetic, best_y_synthetic, best_params, output, total_time_BO, best_classifier_model

# print("==========Executing Baseyan Optimization==========")

# best_val_roc, train_roc, best_params, best_X_synthetic, best_y_synthetic, clf_best_param, params_history, total_time_BO, best_classifier_model = trainDT(optimization_itr)


def save_synthetic_data(data_set_name:str, best_X_synthetic, best_y_synthetic, balanced, encode, tuned, augment_data_percentage, experiment):
    synthetic_data = best_X_synthetic
    prefix = ''
    if data_set_name == 'credit_card':
        if balanced:
            prefix = 'balanced_'
        else:
            prefix = 'unbalanced_'
    str_tuned ='_untuned'
    if tuned:
        str_tuned = '_tuned'
    if encode:
        prefix =  "encoded_" + prefix 
    if augment_data_percentage > 0:
        prefix = 'augmented_' + str(augment_data_percentage) + "_" + prefix
    prefix = 'experiment' + str(experiment) + '_' + prefix
    if data_set_name == 'adult':
        target = 'income'
        synthetic_data[target] = best_y_synthetic
        synthetic_data.loc[synthetic_data[target] == False, target] = " <=50K"
        synthetic_data.loc[synthetic_data[target] == True, target] = " >50K"
        synthetic_data.to_csv(outpath_data_augmentation + prefix + data_set_name + str_tuned + "_models_synthetic_data_xgboost.csv", index=False)
    elif data_set_name == 'credit_card':
        target = 'Class'
        synthetic_data[target] = best_y_synthetic
        synthetic_data.to_csv(outpath_data_augmentation + prefix + data_set_name + str_tuned + "_models_synthetic_data_xgboost.csv", index=False)
    else:
        raise ValueError("Invalid data set name: " + data_set_name)
    return synthetic_data

# HERE ENCODING HAPPENS FOR BOTH XGBOOST AND SYNTHESIZERS
# if encode:
#     best_X_synthetic = encoder.inverse_transform(best_X_synthetic)[initial_columns_ordering]


################################################################################################
# Start experiments
################################################################################################

number_experiments = 10
str_tuned ='_untuned'
if tuned:
    str_tuned = '_tuned'

if augment_data_percentage > 0:
    prefix = 'augmented_' + str(augment_data_percentage) + "_" + prefix

# class_0_ratio = len(df[df[target] == 0]) / len(df)
# class_1_ratio = len(df[df[target] == 1]) / len(df)


all_test_AUC = []
all_val_AUC = []
data_set_name_temp = prefix + data_set_name

all_clf_auc_test_gaussain_copula = []
all_clf_auc_test_ct_gan = []
all_clf_auc_test_copula_gan = []
all_clf_auc_test_tvae = []

all_clf_auc_val_gaussain_copula = []
all_clf_auc_val_ct_gan = []
all_clf_auc_val_copula_gan = []
all_clf_auc_val_tvae = []

for experiment in range(number_experiments):
    prefix_temp = 'experiment' + str(experiment) + '_' + prefix

    encoder = utilities.MultiColumnTargetEncoder(categorical_columns, target)

    df_train, df_val, df_test = get_train_validation_test_data(df, encode)
    df_real = df_train.sample(10000, replace = False)

    class_0_ratio = len(df_train[df_train[target] == 0]) / len(df_train)
    class_1_ratio = len(df_train[df_train[target] == 1]) / len(df_train)

    x_real = df_real.loc[:, df_real.columns != target]
    y_real = df_real[target]

    x_val= df_val.loc[:, df_val.columns != target]
    y_val = df_val[target]

    x_test = df_test.loc[:, df_test.columns != target]
    y_test = df_test[target]

    for column in x_test.columns:
        if x_test[column].dtype == 'object':
            x_test[column] = x_test[column].astype('category')
    dtest = xgb.DMatrix(data=x_test, label=y_test, enable_categorical=True)


    print("==========Executing GaussianCopula==========")

    method = "GaussianCopula"
    params_ = utilities.get_init_params(method)
    if tuned:
        params_ = utilities.get_tuned_params(method, data_set_name, outpath)
    synth_GaussianCopula = utilities.fit_synth(df_train, params_)
    synth_GaussianCopula.fit(df_train)

    sampled_gaussain_copula = None
    print('==========Generating GaussianCopula Data==========')
    if data_set_name_temp == 'unbalanced_credit_card':
        class0 = Condition(
            num_rows=round(N_sim * class_0_ratio),
            column_values={'Class': 0}
        )
        class1 = Condition(
            num_rows=round(N_sim * class_1_ratio),
            column_values={'Class': 1}
        )
        sampled_gaussain_copula = synth_GaussianCopula.sample_from_conditions(
            conditions=[class0, class1]
        )
    else:
        sampled_gaussain_copula = synth_GaussianCopula.sample(num_rows = N_sim)


    print("==========Executing CTGAN==========")
    method = "CTGAN"
    params_ = utilities.get_init_params(method)
    if tuned:
        params_ = utilities.get_tuned_params(method, data_set_name, outpath)
    synth_CTGAN = utilities.fit_synth(df_train, params_)
    synth_CTGAN.fit(df_train)

    print('==========Generating CTGAN Data==========')
    sampled_ct_gan = synth_CTGAN.sample(num_rows = N_sim)

    print("==========Executing CopulaGAN==========")
    method = "CopulaGAN"
    params_ = utilities.get_init_params(method)
    if tuned:
        params_ = utilities.get_tuned_params(method, data_set_name, outpath)
    synth_CopulaGAN = utilities.fit_synth(df_train, params_)
    synth_CopulaGAN.fit(df_train)

    print('==========Generating CopulaGAN Data==========')
    sampled_copula_gan = synth_CopulaGAN.sample(num_rows = N_sim)

    print("==========Executing TVAE==========")
    method = "TVAE"
    params_ = utilities.get_init_params(method)
    if tuned:
        params_ = utilities.get_tuned_params(method, data_set_name, outpath)
    synth_TVAE = utilities.fit_synth(df_train, params_)
    synth_TVAE.fit(df_train)

    print('==========Generating TVAE Data==========')
    sampled_tvae= synth_TVAE.sample(num_rows = N_sim)


    # HERE ENCODING HAPPENS FOR ONLY FOR SYNTHESIZERS
    if encode:
        sampled_gaussain_copula = encoder.inverse_transform(sampled_gaussain_copula)[initial_columns_ordering]
    if encode:
        sampled_ct_gan = encoder.inverse_transform(sampled_ct_gan)[initial_columns_ordering]
    if encode:
        sampled_copula_gan = encoder.inverse_transform(sampled_copula_gan)[initial_columns_ordering]
    if encode:
        sampled_tvae = encoder.inverse_transform(sampled_tvae)[initial_columns_ordering]

    x_gauss = sampled_gaussain_copula.loc[:, sampled_gaussain_copula.columns != target]
    y_gauss = sampled_gaussain_copula[target]
    x_ctgan = sampled_ct_gan.loc[:, sampled_ct_gan.columns != target]
    y_ctgan = sampled_ct_gan[target]
    x_copgan = sampled_copula_gan.loc[:, sampled_copula_gan.columns != target]
    y_copgan = sampled_copula_gan[target]
    x_tvae = sampled_tvae.loc[:, sampled_tvae.columns != target]
    y_tvae = sampled_tvae[target]

    clf_auc_train_gaussain_copula, clf_auc_val_gaussain_copula, clf_gaussain_copula = downstream_loss(sampled_gaussain_copula, df_val, target, 'XGB')
    clf_auc_train_ct_gan, clf_auc_val_ct_gan, clf_ct_gan = downstream_loss(sampled_ct_gan, df_val, target, 'XGB')
    clf_auc_train_copula_gan, clf_auc_val_copula_gan, clf_copula_gan = downstream_loss(sampled_copula_gan, df_val, target, 'XGB')
    clf_auc_train_tvae, clf_auc_val_tvae, clf_tvae = downstream_loss(sampled_tvae, df_val, target, 'XGB')
    
    # lmname =['GaussianCopula', 'CTGAN', 'CopulaGAN', 'TVAE']
    val_auc = [clf_auc_val_gaussain_copula, clf_auc_val_ct_gan, clf_auc_val_copula_gan, clf_auc_val_tvae]

    clf_auc_test_gaussain_copula = 0.5
    if clf_gaussain_copula != None:
        clf_probs_test_gaussain_copula = clf_gaussain_copula.predict(dtest)
        clf_auc_test_gaussain_copula = roc_auc_score(y_test.astype(float), clf_probs_test_gaussain_copula)

    clf_auc_test_ct_gan = 0.5
    if clf_ct_gan != None:
        clf_probs_test_ct_gan = clf_ct_gan.predict(dtest)
        clf_auc_test_ct_gan = roc_auc_score(y_test.astype(float), clf_probs_test_ct_gan)

    clf_auc_test_copula_gan = 0.5
    if clf_copula_gan != None:
        clf_probs_test_copula_gan = clf_copula_gan.predict(dtest)
        clf_auc_test_copula_gan = roc_auc_score(y_test.astype(float), clf_probs_test_copula_gan)

    clf_auc_test_tvae = 0.5
    if clf_tvae != None:
        clf_probs_test_tvae = clf_tvae.predict(dtest)
        clf_auc_test_tvae = roc_auc_score(y_test.astype(float), clf_probs_test_tvae)

    if len(all_clf_auc_test_gaussain_copula) == 0:
        all_clf_auc_test_gaussain_copula = [clf_auc_test_gaussain_copula]
        all_clf_auc_test_ct_gan = [clf_auc_test_ct_gan]
        all_clf_auc_test_copula_gan = [clf_auc_test_copula_gan]
        all_clf_auc_test_tvae = [clf_auc_test_tvae]

        all_clf_auc_val_gaussain_copula = [clf_auc_val_gaussain_copula]
        all_clf_auc_val_ct_gan = [clf_auc_val_ct_gan]
        all_clf_auc_val_copula_gan = [clf_auc_val_copula_gan]
        all_clf_auc_val_tvae = [clf_auc_val_tvae]
    else:
        all_clf_auc_test_gaussain_copula.append(clf_auc_test_gaussain_copula)
        all_clf_auc_test_ct_gan.append(clf_auc_test_ct_gan)
        all_clf_auc_test_copula_gan.append(clf_auc_test_copula_gan)
        all_clf_auc_test_tvae.append(clf_auc_test_tvae)
    
        all_clf_auc_val_gaussain_copula.append(clf_auc_val_gaussain_copula)
        all_clf_auc_val_ct_gan.append(clf_auc_val_ct_gan)
        all_clf_auc_val_copula_gan.append(clf_auc_val_copula_gan)
        all_clf_auc_val_tvae.append(clf_auc_val_tvae)

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
    
    print("==========Executing Baseyan Optimization==========")
    # Combine all the data into a single list
    X_temp = [x_gauss, x_ctgan, x_copgan, x_tvae, x_real]
    y_temp = [y_gauss, y_ctgan, y_copgan, y_tvae, y_real]

    best_val_roc, train_roc, best_params, best_X_synthetic, best_y_synthetic, clf_best_param, params_history, total_time_BO, best_classifier_model = trainDT(optimization_itr, X_temp, y_temp, val_auc)
    save_synthetic_data(data_set_name, best_X_synthetic, best_y_synthetic, balanced, encode, tuned, augment_data_percentage, experiment)
    clf_probs = best_classifier_model.predict(dtest)
    test_roc = roc_auc_score(y_test.astype(float), clf_probs)
    clf_best_param["train_roc"] = train_roc
    clf_best_param["val_roc"] = best_val_roc
    clf_best_param["test_roc"] = test_roc
    clf_best_param["total_time_BO"] = total_time_BO
    if len(all_test_AUC) == 0:
        all_test_AUC = [test_roc]
        all_val_AUC = [best_val_roc]
    else:
        all_test_AUC.append(test_roc)
        all_val_AUC.append(best_val_roc)
    # print('clf_best_param: ', clf_best_param)

    alpha_temp = [clf_best_param['alpha_1'], clf_best_param['alpha_2'], clf_best_param['alpha_3'], clf_best_param['alpha_4'], clf_best_param['alpha_5']]
    assert abs(sum(alpha_temp) - 1) < 0.001

    individual_clf_auc_df = pd.DataFrame()
    individual_clf_auc_df = individual_clf_auc_df._append(individual_clf_auc, ignore_index = True)

    individual_clf_auc_df.to_csv(outpath_data_augmentation + prefix_temp + data_set_name + str_tuned + "_models_clf_auc_score_and_time_per_each_individual_model.csv", index=False)

    clf_best_param_df = pd.DataFrame()
    clf_best_param_df = clf_best_param_df._append(clf_best_param, ignore_index = True)
    clf_best_param_df.to_csv(outpath_data_augmentation + prefix_temp + data_set_name + str_tuned + "_models_clf_best_param_xgboost.csv", index=False)

    params_history.to_csv(historypath_data_augmentation + prefix_temp + data_set_name + str_tuned + "_models_params_alpha_history.csv", index=False)


print('all_test_AUC: ', all_test_AUC)
print('all_val_AUC: ', all_val_AUC)

print('Average test AUC: ', np.average(all_test_AUC))
print('Average test AUC: ', np.average(all_val_AUC))

print('Std test AUC: ', np.std(all_test_AUC))
print('Std val AUC: ', np.std(all_val_AUC))

print('all_clf_auc_test_gaussain_copula: ', all_clf_auc_test_gaussain_copula)
print('all_clf_auc_test_ct_gan: ', all_clf_auc_test_ct_gan)
print('all_clf_auc_test_copula_gan: ', all_clf_auc_test_copula_gan)
print('all_clf_auc_test_tvae: ', all_clf_auc_test_tvae)

pd_all_results = pd.DataFrame({'our_method_test_AUC' : all_test_AUC, 
              'our_method_val_AUC' : all_val_AUC,
              'gaussain_copula_test_AUC' : all_clf_auc_test_gaussain_copula,
              'gaussain_copula_val_AUC' : all_clf_auc_val_gaussain_copula,
              'ct_gan_test_AUC' : all_clf_auc_test_ct_gan,
              'ct_gan_val_AUC' : all_clf_auc_val_ct_gan,
              'copula_gan_test_AUC' : all_clf_auc_test_copula_gan,
              'copula_gan_val_AUC' : all_clf_auc_val_copula_gan,
              'tvae_test_AUC' : all_clf_auc_test_tvae,
              'tvae_val_AUC' : all_clf_auc_val_tvae,
              })

print('Average test AUC gaussain_copula: ', np.average(all_clf_auc_test_gaussain_copula))
print('Average test AUC ct_gan: ', np.average(all_clf_auc_test_ct_gan))
print('Average test AUC copula_gan: ', np.average(all_clf_auc_test_copula_gan))
print('Average test AUC tvae: ', np.average(all_clf_auc_test_tvae))

print('Std test AUC gaussain_copula: ', np.std(all_clf_auc_test_gaussain_copula))
print('Std test AUC ct_gan: ', np.std(all_clf_auc_test_ct_gan))
print('Std test AUC copula_gan: ', np.std(all_clf_auc_test_copula_gan))
print('Std test AUC tvae: ', np.std(all_clf_auc_test_tvae))

methods = ['our method', 'gaussain_copula', 'ct_gan', 'copula_gan', 'tvae']

average_auc_test = [np.average(all_test_AUC), np.average(all_clf_auc_test_gaussain_copula), np.average(all_clf_auc_test_ct_gan),
                   np.average(all_clf_auc_test_copula_gan), np.average(all_clf_auc_test_tvae)]
average_auc_val = [np.average(all_val_AUC), np.average(all_clf_auc_val_gaussain_copula), np.average(all_clf_auc_val_ct_gan),
                   np.average(all_clf_auc_val_copula_gan), np.average(all_clf_auc_val_tvae)]

std_auc_test = [np.std(all_test_AUC), np.std(all_clf_auc_test_gaussain_copula), np.std(all_clf_auc_test_ct_gan),
                   np.std(all_clf_auc_test_copula_gan), np.std(all_clf_auc_test_tvae)]
std_auc_val = [np.std(all_val_AUC), np.std(all_clf_auc_val_gaussain_copula), np.std(all_clf_auc_val_ct_gan),
                   np.std(all_clf_auc_val_copula_gan), np.std(all_clf_auc_val_tvae)]

pd_results_average_srt = pd.DataFrame({'methods' : methods,
                                       'average_auc_test' : average_auc_test,
                                       'average_auc_val' : average_auc_val,
                                       'std_auc_test' : std_auc_test,
                                       'std_auc_val' : std_auc_val

})

outpath_data_results = "data/outputResults/"
all_results_AUC_file_name = outpath_data_results + prefix + data_set_name + str_tuned +  '_all_results_AUC_file_name.csv'
average_std_AUC_file_name = outpath_data_results + prefix + data_set_name + str_tuned +  '_average_std_AUC_file_name.csv'

pd_all_results.to_csv(all_results_AUC_file_name, index=False)
pd_results_average_srt.to_csv(average_std_AUC_file_name, index=False)

# #Compute test ROC
# for column in x_test.columns:
#     if x_test[column].dtype == 'object':
#         x_test[column] = x_test[column].astype('category')
# dtest = xgb.DMatrix(data=x_test, label=y_test, enable_categorical=True)
# clf_probs = best_classifier_model.predict(dtest)
# test_roc = roc_auc_score(y_test.astype(float), clf_probs)

# clf_best_param["train_roc"] = train_roc
# clf_best_param["val_roc"] = best_val_roc
# clf_best_param["test_roc"] = test_roc
# clf_best_param["total_time_BO"] = total_time_BO

# print('clf_best_param: ', clf_best_param)

# alpha_temp = [clf_best_param['alpha_1'], clf_best_param['alpha_2'], clf_best_param['alpha_3'], clf_best_param['alpha_4'], clf_best_param['alpha_5']]
# print('abs(sum(alpha_temp) - 1) : ', abs(sum(alpha_temp) - 1))
# assert abs(sum(alpha_temp) - 1) < 0.001

# str_tuned ='_untuned'
# if tuned:
#     str_tuned = '_tuned'

# if augment_data_percentage > 0:
#     prefix = 'augmented_' + str(augment_data_percentage) + "_" + prefix

# clf_best_param_df = pd.DataFrame()
# clf_best_param_df = clf_best_param_df._append(clf_best_param, ignore_index = True)
# clf_best_param_df.to_csv(outpath_data_augmentation + prefix + data_set_name + str_tuned + "_models_clf_best_param_xgboost.csv", index=False)

# params_history.to_csv(historypath_data_augmentation + prefix + data_set_name + str_tuned + "_models_params_alpha_history.csv", index=False)


# start_time_GaussianCopula = time.time()
# clf_auc_train_gaussain_copula, clf_auc_val_gaussain_copula, clf_gaussain_copula = downstream_loss(sampled_gaussain_copula, df_val, target, 'XGB')
# end_time_GaussianCopula = time.time()
# start_time_CTGAN = time.time()
# clf_auc_train_ct_gan, clf_auc_val_ct_gan, clf_ct_gan= downstream_loss(sampled_ct_gan, df_val, target, 'XGB')
# end_time_CTGAN = time.time()
# start_time_CopulaGAN = time.time()
# clf_auc_train_copula_gan, clf_auc_val_copula_gan, clf_copula_gan = downstream_loss(sampled_copula_gan, df_val, target, 'XGB')
# end_time_CopulaGAN = time.time()
# start_time_TVAE = time.time()
# clf_auc_train_tvae, clf_auc_val_tvae, clf_tvae = downstream_loss(sampled_tvae, df_val, target, 'XGB')
# end_time_TVAE = time.time()

# for column in x_test.columns:
#     if x_test[column].dtype == 'object':
#         x_test[column] = x_test[column].astype('category')
# dtest = xgb.DMatrix(data=x_test, label=y_test, enable_categorical=True)

# clf_auc_test_gaussain_copula = 0.5
# if clf_gaussain_copula != None:
#     clf_probs_test_gaussain_copula = clf_gaussain_copula.predict(dtest)
#     clf_auc_test_gaussain_copula = roc_auc_score(y_test.astype(float), clf_probs_test_gaussain_copula)

# clf_auc_test_ct_gan = 0.5
# if clf_ct_gan != None:
#     clf_probs_test_ct_gan = clf_ct_gan.predict(dtest)
#     clf_auc_test_ct_gan = roc_auc_score(y_test.astype(float), clf_probs_test_ct_gan)

# clf_auc_test_copula_gan = 0.5
# if clf_copula_gan != None:
#     clf_probs_test_copula_gan = clf_copula_gan.predict(dtest)
#     clf_auc_test_copula_gan = roc_auc_score(y_test.astype(float), clf_probs_test_copula_gan)

# clf_auc_test_tvae = 0.5
# if clf_tvae != None:
#     clf_probs_test_tvae = clf_tvae.predict(dtest)
#     clf_auc_test_tvae = roc_auc_score(y_test.astype(float), clf_probs_test_tvae)

# individual_clf_auc = {'clf_auc_train_gaussain_copula' : clf_auc_train_gaussain_copula,
#                     'clf_auc_val_gaussain_copula' : clf_auc_val_gaussain_copula,
#                     'clf_auc_test_gaussain_copula' : clf_auc_test_gaussain_copula,
#                     'clf_auc_train_ct_gan' : clf_auc_train_ct_gan,
#                     'clf_auc_val_ct_gan' : clf_auc_val_ct_gan,
#                     'clf_auc_test_ct_gan' : clf_auc_test_ct_gan, 
#                     'clf_auc_train_copula_gan' : clf_auc_train_copula_gan,
#                     'clf_auc_val_copula_gan' : clf_auc_val_copula_gan,
#                     'clf_auc_test_copula_gan' : clf_auc_test_copula_gan,
#                     'clf_auc_train_tvae' : clf_auc_train_tvae,
#                     'clf_auc_val_tvae' : clf_auc_val_tvae,
#                     'clf_auc_test_tvae' : clf_auc_test_tvae,
#                     'train' : len(sampled_tvae),
#                     'val' : len(df_val),
#                     'test' : len(df_test)}

# individual_clf_auc["total_time_GaussianCopula"] = end_time_GaussianCopula - start_time_GaussianCopula
# individual_clf_auc["total_time_CTGAN"] = end_time_CTGAN - start_time_CTGAN
# individual_clf_auc["total_time_CopulaGAN"] = end_time_CopulaGAN - start_time_CopulaGAN
# individual_clf_auc["total_time_TVAE"] = end_time_TVAE - start_time_TVAE

# print('individual_clf_auc: ', individual_clf_auc)

# individual_clf_auc_df = pd.DataFrame()
# individual_clf_auc_df = individual_clf_auc_df._append(individual_clf_auc, ignore_index = True)

# individual_clf_auc_df.to_csv("data/output/" + prefix + data_set_name + str_tuned + "_models_clf_auc_score_and_time_per_each_individual_model.csv", index=False)