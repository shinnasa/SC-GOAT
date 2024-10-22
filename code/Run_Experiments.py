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
import os
os.chdir("code/")
from utilities import *
os.chdir("..")
import random
import sys
import warnings

warnings.filterwarnings('ignore')

##################################################################################################
# Define arguments for the python script
##################################################################################################

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--DATA_SET_NAME", help="DATA_SET_NAME can be ['adult', 'unbalanced_credit_card', 'balanced_credit_card']",  default='adult', choices=["adult", "unbalanced_credit_card", "balanced_credit_card"])
parser.add_argument("-e", "--ENCODE", help="ENCODE can be [True, False]", default="False", choices = ["True", "False"])
parser.add_argument("-i", "--ITR", help="Number of Optimization Iterations", type=int, default=350)
parser.add_argument("-g", "--GENERATED_DATA_SIZE", help="Number of Generated Data", type=int, default=10000)
parser.add_argument("-o", "--OUTPUT_DIR", help="Output directory", required=True)
parser.add_argument("-id", "--ID", help="Experiment ID", type=int, default=1)
parser.add_argument("-a", "--AUGMENT_DATA_PERCENTAGE", help="Augment Data Percentage", type=float, default=0, choices=np.arange(0, 1.0, 0.01))
args = parser.parse_args()

##################################################################################################
# Get user defined arguments
##################################################################################################
print("arguments: ", args)
data_set_name = args.DATA_SET_NAME
encode = eval(args.ENCODE)
optimization_itr = args.ITR
shortrpoch = False
output_dir = args.OUTPUT_DIR
experiment = args.ID
assert experiment >= 1
augment_data_percentage = args.AUGMENT_DATA_PERCENTAGE
assert 0 <= augment_data_percentage <= 1
augment_data = augment_data_percentage > 0
generated_data_size = args.GENERATED_DATA_SIZE
assert generated_data_size > 0

balanced = False
if data_set_name == "balanced_credit_card":
    balanced = True    

target = 'income'
if data_set_name == 'credit_card':
    target = 'Class'

prefix = ''
if encode:
    prefix =  "encoded_" + prefix 

outpath_tuned_params = output_dir + "/output/"
outpath_experiments = output_dir + "/outputDatExperiments/"
historypath_experiments = output_dir + "/historyDataExperiments/"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    os.mkdir(outpath_tuned_params)
    os.mkdir(outpath_experiments)
    os.mkdir(historypath_experiments)

if not os.path.exists(outpath_tuned_params):
    os.mkdir(outpath_tuned_params)

if not os.path.exists(outpath_experiments):
    os.mkdir(outpath_experiments)

if not os.path.exists(historypath_experiments):
    os.mkdir(historypath_experiments)

print('data_set_name: ', data_set_name)
print('target: ', target)
print('optimization_itr: ', optimization_itr)
print('encode: ', encode)
print('balanced: ', balanced)
print('augment_data: ', augment_data)
print('augment_data_percentage: ', augment_data_percentage)
print('prefix: ', prefix)

################################################################################################
# Load data
################################################################################################
df_original = load_data_original(data_set_name, balanced)

df = df_original.copy()
Nrows = 50000
if len(df) > Nrows:
    df = df.sample(Nrows, replace = False)

categorical_columns = []

for column in df.columns:
    if (column != target) & (df[column].dtype == 'object'):
        categorical_columns.append(column)

initial_columns_ordering = df.columns.values


################################################################################################
# Define functions. These will be moved to utilities.py eventually
################################################################################################


lmname =['GaussianCopula', 'CTGAN', 'CopulaGAN', 'TVAE']

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
    scale = (sum(alpha_temp)-augment_data_percentage)  / (1 - augment_data_percentage)
    alpha = [(1 / scale) * alpha_temp[i] for i in range(len(alpha_temp) - 1)]
    alpha.append(augment_data_percentage)
    index = np.argmax(alpha)
    params['alpha_1'] = alpha[0]
    params['alpha_2'] = alpha[1]
    params['alpha_3'] = alpha[2]
    params['alpha_4'] = alpha[3]
    params['alpha_5'] = alpha[4]
    # Randomly select the data from each source
    x_val_real = x_val.copy()
    y_val_real = y_val.copy()

    generated_data_size = params['generated_data_size']

    size = [int(alpha[i] * len(y_temp[i].index.values)) for i in range(len(alpha))]
    size[index] += (generated_data_size - sum(size))
    randomRows = random.sample(list(y_temp[0].index.values), size[0])
    randomRows = np.sort(randomRows)
    X_new = X_temp[0].loc[randomRows]
    y_new = y_temp[0].loc[randomRows].values
    # Randomly select the data from each source based on the alpha values
    for i in range(1, len(y_temp)):
        n = size[i]
        if n > 0:
            randomRows = random.sample(list(y_temp[i].index.values), n)
            randomRows = np.sort(randomRows)
            X_new = pd.concat([X_new, X_temp[i].loc[randomRows]])
            y_new = np.concatenate((y_new, y_temp[i].loc[randomRows].values))

    X_synthetic = X_new.copy()
    y_synthetic = y_new.copy()
    assert X_synthetic.shape[0] == 10000
    # Train classifier
    for column in X_new.columns:
        if X_new[column].dtype == 'object':
            X_new[column] = X_new[column].astype('category')
            x_val_real[column] = x_val_real[column].astype('category')
    dtrain = xgb.DMatrix(data=X_new, label=y_new, enable_categorical=True)
    # dval = xgb.DMatrix(data=x_val_real, label=y_val_real, enable_categorical=True)
    # clf = xgb.train(params = params_xgb, dtrain=dtrain, verbose_eval=False)

    data_temp_train = X_synthetic.copy()
    data_temp_train[target] = y_synthetic.copy()
    data_temp_val = x_val_real.copy()
    data_temp_val[target] = y_val_real.copy()
    clf, clf_auc = downstream_loss(data_temp_train, data_temp_val, target)

    # Evaluate the performance of the classifier
    # clf_probs = clf.predict(dval)
    # clf_auc = roc_auc_score(y_val_real.astype(float), clf_probs)
    
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
    initial_alphas_dict_auc = {'alpha_1' : initial_alphas[0], 
                           'alpha_2' : initial_alphas[1], 
                           'alpha_3' : initial_alphas[2] ,
                           'alpha_4' : initial_alphas[3], 
                           'alpha_5' : augment_data_percentage
                           }
    trials = generate_trials_to_calculate([initial_alphas_dict_auc])
    start_time_BO = time.time()
    params_range = get_alpha_Params(augment_data_percentage)
    # print('getParams: ', getParams(augment_data_percentage))

    #GET the starting params value including the corner solutions
    clf_best_param_initialization = fmin(fn=objective_maximize_roc,
                    space=params_range,
                    max_evals=1,
                    algo=tpe.suggest,
                    trials=trials)
    initial_alphas_dict = {}
    print('clf_best_param_initialization: ', clf_best_param_initialization)
    print('best_val_roc: ', best_val_roc)
    print('val_auc_arg: ', val_auc_arg)
    if best_val_roc > max(val_auc_arg):
        initial_alphas_dict = initial_alphas_dict_auc
    else:
        i = np.argmax(val_auc_arg)
        print('i: ', i)
        initial_alphas = [0 for i in range(len(initial_alphas))]
        initial_alphas[i] = 1
        initial_alphas_dict = {'alpha_1' : initial_alphas[0], 
                           'alpha_2' : initial_alphas[1], 
                           'alpha_3' : initial_alphas[2] ,
                           'alpha_4' : initial_alphas[3], 
                           'alpha_5' : augment_data_percentage
                           }
    print('initial_alphas_dict: ', initial_alphas_dict)
    
    output = pd.DataFrame()
    train_roc = 0
    best_val_roc = 0
    best_params = []
    global best_X_synthetic
    global best_y_synthetic
    global best_classifier_model
    X_temp = X_temp_arg
    y_temp = y_temp_arg
    val_auc = val_auc_arg
    best_val_roc = 0
    train_roc = 0

    trials = generate_trials_to_calculate([initial_alphas_dict])

    clf_best_param = fmin(fn=objective_maximize_roc,
                    space=params_range,
                    max_evals=max_evals,
                    early_stop_fn=no_progress_loss(1, 5),
                    algo=tpe.suggest,
                    trials=trials)
    
    print('best_val_roc: ', best_val_roc)

    end_time_BO = time.time()
    total_time_BO = end_time_BO - start_time_BO
    return best_val_roc, train_roc, best_params, best_X_synthetic, best_y_synthetic, best_params, output, total_time_BO, best_classifier_model

################################################################################################
# Start experiments
################################################################################################

if augment_data_percentage > 0:
    prefix = 'augmented_' + str(augment_data_percentage) + "_" + prefix

prefix_temp = 'experiment' + str(experiment) + '_' + prefix

df_train, df_val, df_test, encoder = get_train_validation_test_data(df, encode, target)

N_sim = 10000
df_real = df_train.sample(N_sim, replace = False)

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

params_xgb = {
        'eval_metric': 'auc', 'objective':'binary:logistic', 'seed': 5,
    }
lmethods = ["GaussianCopula", "CTGAN", "CopulaGAN", "TVAE"]


lres = []
X_temp_u = []
X_temp_t = []
y_temp_u = []
y_temp_t = []
# Provide the root directory where you want to start the search
root_directory = 'data/temp/'
# Provide the target string you want to search for in folder names
target_string = prefix + data_set_name + "_"+ str(experiment)
remove_folders_with_string(root_directory, target_string)

for method_name in lmethods:
    params_u = get_init_params(method_name)
    if method_name == "GaussianCopula":
        params_t = {}
    else:
        params_t = get_tuned_params(method_name, prefix + data_set_name, outpath_tuned_params)
    params_u['method'] = method_name
    params_t['method'] = method_name
    if 'batch_size' not in params_t:
        params_t['batch_size'] = 5
    
    if shortrpoch:
        params_t['epochs'] = 1
        params_u['epochs'] = 1
    else:
        params_t['epochs'] = 150
        params_u['epochs'] = 150
    print("params_t: ", params_t)
    print("params_u: ", params_u)
    syn_u = fit_synth(df_train, params_u)
    syn_t = fit_synth(df_train, params_t)
    syn_u.fit(df_train)
    syn_t.fit(df_train)
    
    if method_name == "GaussianCopula" and (prefix + data_set_name) == 'unbalanced_credit_card':
        class_0_ratio = len(df_train[df_train[target] == 0]) / len(df_train)
        class_1_ratio = len(df_train[df_train[target] == 1]) / len(df_train)
        class0 = Condition(
            num_rows=round(N_sim * class_0_ratio),
            column_values={'Class': 0}
        )
        class1 = Condition(
            num_rows=round(N_sim * class_1_ratio),
            column_values={'Class': 1}
        )
        sampled_u = syn_u.sample_from_conditions(conditions=[class0, class1], output_file_path="data/temp/untuned_" + prefix + data_set_name + "_"+ str(experiment) + ".csv")
        sampled_t = syn_t.sample_from_conditions(conditions=[class0, class1], output_file_path="data/temp/tuned_" + prefix + data_set_name + "_"+ str(experiment) + ".csv")
    else:
        print("data/temp/untuned_" + prefix + data_set_name + "_"+ str(experiment) + ".csv")
        sampled_u = syn_u.sample(N_sim, output_file_path="data/temp/untuned_" + prefix + data_set_name + "_"+ str(experiment) + ".csv")
        sampled_t = syn_t.sample(N_sim, output_file_path='data/temp/tuned_' + prefix + data_set_name + "_"+ str(experiment) + ".csv")
    os.remove("data/temp/untuned_" + prefix + data_set_name + "_"+ str(experiment) + ".csv")
    os.remove("data/temp/tuned_" + prefix + data_set_name + "_"+ str(experiment) + ".csv")
    if encode:
        sampled_u = encoder.inverse_transform(sampled_u)[initial_columns_ordering]
        sampled_t = encoder.inverse_transform(sampled_t)[initial_columns_ordering]
    clf_, val_loss_u = downstream_loss(sampled_u, df_val, target)
    clf_, val_loss_t = downstream_loss(sampled_t, df_val, target)
    clf_, test_loss_u = downstream_loss(sampled_u, df_test, target)
    clf_, test_loss_t = downstream_loss(sampled_t, df_test, target)
    lrow = [data_set_name, method_name, val_loss_u, val_loss_t, test_loss_u, test_loss_t]
    lres.append(lrow)
    X_temp_u.append(sampled_u.loc[:, sampled_u.columns != target])
    X_temp_t.append(sampled_t.loc[:, sampled_t.columns != target])
    y_temp_u.append(sampled_u[target])
    y_temp_t.append(sampled_t[target])
    print("========== " + method_name + " Done ==========")

X_temp_u.append(x_real)
X_temp_t.append(x_real)
y_temp_u.append(y_real)
y_temp_t.append(y_real)
dfres = pd.DataFrame(lres, columns=['data', 'method', 'untuned_val_auc', 'tuned_val_auc', 'untuned_test_auc', 'tuned_test_auc'])
val_auc_u = dfres['untuned_val_auc'].tolist()
val_auc_t = dfres['tuned_val_auc'].tolist()
print("==========Executing Baseyan Optimization==========")
# Combine all the data into a single list


best_val_roc_u, train_roc_u, best_params_u, best_X_synthetic_u, best_y_synthetic_u, clf_best_param_u, params_history_u, total_time_BO_u, best_classifier_model_u = trainDT(optimization_itr, X_temp_u, y_temp_u, val_auc_u)
best_val_roc_t, train_roc_t, best_params_t, best_X_synthetic_t, best_y_synthetic_t, clf_best_param_t, params_history_t, total_time_BO_t, best_classifier_model_t = trainDT(optimization_itr, X_temp_t, y_temp_t, val_auc_t)


clf_probs_u = best_classifier_model_u.predict(dtest)
clf_probs_t = best_classifier_model_t.predict(dtest)
test_roc_u = roc_auc_score(y_test.astype(float), clf_probs_u)
test_roc_t = roc_auc_score(y_test.astype(float), clf_probs_t)
lrow = [data_set_name, 'SC-GOAT', best_val_roc_u, best_val_roc_t, test_roc_u, test_roc_t]
lres.append(lrow)
dfres = pd.DataFrame(lres, columns=['data', 'method', 'untuned_val_auc', 'tuned_val_auc', 'untuned_test_auc', 'tuned_test_auc'])
dfres['experiment'] = experiment
dfres.to_csv(outpath_experiments + prefix_temp + data_set_name + "_val_test_auc.csv", index=False)

save_synthetic_data(data_set_name, best_X_synthetic_u, best_y_synthetic_u, balanced, encode, False, augment_data_percentage, experiment, outpath_experiments)
save_synthetic_data(data_set_name, best_X_synthetic_t, best_y_synthetic_t, balanced, encode, True, augment_data_percentage, experiment, outpath_experiments)

clf_best_param_u["untuned_train_roc"] = train_roc_u
clf_best_param_u["untuned_val_roc"] = best_val_roc_u
clf_best_param_u["untuned_test_roc"] = test_roc_u
clf_best_param_u["untuned_total_time_BO"] = total_time_BO_u

clf_best_param_t["tuned_train_roc"] = train_roc_t
clf_best_param_t["tuned_val_roc"] = best_val_roc_t
clf_best_param_t["tuned_test_roc"] = test_roc_t
clf_best_param_t["tuned_total_time_BO"] = total_time_BO_t

clf_best_param_df_u = pd.DataFrame()
clf_best_param_df_u = clf_best_param_df_u._append(clf_best_param_u, ignore_index = True)
clf_best_param_df_u.to_csv(outpath_experiments + prefix_temp + data_set_name + "_untuned_" + "_models_clf_best_param_xgboost.csv", index=False)

clf_best_param_df_t = pd.DataFrame()
clf_best_param_df_t = clf_best_param_df_t._append(clf_best_param_t, ignore_index = True)
clf_best_param_df_t.to_csv(outpath_experiments + prefix_temp + data_set_name + "_tuned_" + "_models_clf_best_param_xgboost.csv", index=False)

params_history_u.to_csv(historypath_experiments + prefix_temp + data_set_name + "_untuned_" + "_models_params_alpha_history.csv", index=False)
params_history_u.to_csv(historypath_experiments + prefix_temp + data_set_name + "_tuned_" + "_models_params_alpha_history.csv", index=False)