import pandas as pd
import numpy as np
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.single_table import CopulaGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from hyperopt.early_stop import no_progress_loss
import time
from sdv.sampling import Condition
import sys
# Create class for encoding
class MultiColumnTargetEncoder:
    def __init__(self,columns, target_column):
        self.columns = columns # array of column names to encode
        self.target_column = target_column
        
    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        Target Encoding If no columns specified, transforms all
        categorical columns in X.
        '''
        output = X.copy()
        self.column_value_to_column_encoding = {}
        self.encoded_columns = []
        self.column_encoding_to_column_value = {}
        if self.columns is not None:
            for col in self.columns:
                if len(self.encoded_columns) == 0:
                    self.encoded_columns = [col + '_target_encoded']
                else:
                    self.encoded_columns.append(col + '_target_encoded')
                self.column_value_to_column_encoding[col] = X[self.target_column].groupby(X[col]).agg(['count', 'mean'])
                
                self.column_encoding_to_column_value[col] = {}
                for column_unique_val in self.column_value_to_column_encoding[col].index:
                    self.column_encoding_to_column_value[col][self.column_value_to_column_encoding[col].loc[column_unique_val]['mean']] = column_unique_val
                    output.loc[output[col] == column_unique_val, col + '_target_encoded'] = self.column_value_to_column_encoding[col].loc[column_unique_val]['mean']
       
        else:
            for column in X.columns:
                if X[column].dtype != 'object':
                    if len(self.encoded_columns) == 0:
                        self.encoded_columns = [column + '_target_encoded']
                    else:
                        self.encoded_columns.append(column + '_target_encoded')
                    self.column_value_to_column_encoding[column] = X[self.target_column].groupby(X[column]).agg(['count', 'mean'])
                    self.column_encoding_to_column_value[column] = {}
                    for column_unique_val in self.column_value_to_column_encoding[column].index:
                        self.column_encoding_to_column_value[col][self.column_value_to_column_encoding[col].loc[column_unique_val]['mean']] = column_unique_val
                        output.loc[output[column] == column_unique_val, column + '_target_encoded'] = self.column_value_to_column_encoding[column].loc[column_unique_val]['mean']
        output = output.drop(columns=self.columns)
        return output
    
    def transform_test_data(self, X):
        output = X.copy()
        for col in self.columns:
            for column_unique_val in self.column_value_to_column_encoding[col].index:
                self.column_encoding_to_column_value[col][self.column_value_to_column_encoding[col].loc[column_unique_val]['mean']] = column_unique_val
                output.loc[output[col] == column_unique_val, col + '_target_encoded'] = self.column_value_to_column_encoding[col].loc[column_unique_val]['mean']
                
        output = output.drop(columns=self.columns)
        
        return output
    
    def findMatchingValue(self, column_index, search_value):
        min_difference = float('inf')
        possible_values = self.column_encoding_to_column_value[self.columns[column_index]].keys()
        
        if search_value in possible_values:
            return search_value

        closest_value = search_value

        for value in possible_values:
            if abs(value - search_value) < min_difference:
                min_difference = abs(value - search_value)
                closest_value = value
        return closest_value

    def inverse_transform(self,X):
        output = X.copy()
        for index, column in enumerate(self.encoded_columns):
            for column_unique_val in output[column].unique():
                closest_value = self.findMatchingValue(index, column_unique_val)
                output.loc[output[column] == column_unique_val, self.columns[index]] = self.column_encoding_to_column_value[self.columns[index]][closest_value]
        output = output.drop(columns=self.encoded_columns)
        return output 
        

def function_test_MultiColumnTargetEncoder():
    df_original = pd.read_csv('../data/adult/adult.csv')
    categorical_columns = ['workclass', 'education', 'native-country']
    target_column = 'income'
    df_original.loc[df_original[target_column] == "<=50K", target_column] = 0
    df_original.loc[df_original[target_column] == ">50K", target_column] = 1
    df_original.replace('?', np.NaN,inplace=True)
    df_original.dropna(axis=0,how='any',inplace=True)
    encoder_ = MultiColumnTargetEncoder(categorical_columns, target_column)
    print(df_original)
    df_modified = encoder_.transform(df_original)
    print(df_modified)
    df_new =  encoder_.inverse_transform(df_modified)
    print(df_new)

    print(df_new['workclass'] == df_original['workclass'])
    print(df_new['education'] == df_original['education'])
    print(df_new['native-country'] == df_original['native-country'])

# function_test_MultiColumnTargetEncoder()

# Function to load different datasets
def load_data(data_set_name:str):
    adult_data_set_dir = "data/adult"
    credit_card_data_set_dir = "data/credit_card"
    adult_data_set_csv_file_name = 'adult'
    balaned_credit_card_data_set_csv_file_name = 'credit_card_balanced'
    unbalaned_credit_card_data_set_csv_file_name = 'credit_card_unbalanced'

    df_original = pd.DataFrame()
    if data_set_name == 'adult':
        df_original = pd.read_csv(adult_data_set_dir + '/' + adult_data_set_csv_file_name + '.csv')
        target = 'income'
        df_original.loc[df_original[target] == "<=50K", target] = 0
        df_original.loc[df_original[target] == ">50K", target] = 1
        df_original.replace('?', np.NaN,inplace=True)
        df_original.dropna(axis=0,how='any',inplace=True)
    elif data_set_name == 'balanced_credit_card':
        df_original = pd.read_csv(credit_card_data_set_dir + '/' + balaned_credit_card_data_set_csv_file_name + '.csv')
    elif data_set_name == 'unbalanced_credit_card':
            df_original = pd.read_csv(credit_card_data_set_dir + '/' + unbalaned_credit_card_data_set_csv_file_name + '.csv')
    else:
        raise ValueError("Invalid data set name: " + data_set_name)
    return df_original

def get_train_validation_test_data(df, encode, target):
    df_train_original, df_test_original = train_test_split(df, test_size = 0.3,) #70% is training and 30 to test
    df_val_original, df_test_original = train_test_split(df_test_original, test_size = 1 - 0.6666666)# out of 30, 20 is test and 10 for validation

    if encode:
        categorical_columns = []
        for column in df.columns:
            if (column != target) & (df[column].dtype == 'object'):
                categorical_columns.append(column)
        encoder = MultiColumnTargetEncoder(categorical_columns, target)
        df_train = encoder.transform(df_train_original)
        df_val = df_val_original
        df_test = df_test_original

        return df_train, df_val, df_test, encoder
    else:
        return df_train_original, df_val_original, df_test_original, None

def load_data_original(data_set_name:str, balanced:bool=False):
    adult_data_set_dir = "data/adult"
    credit_card_data_set_dir = "data/credit_card"
    adult_data_set_csv_file_name = 'adult'
    balaned_credit_card_data_set_csv_file_name = 'credit_card_balanced'
    unbalaned_credit_card_data_set_csv_file_name = 'credit_card_unbalanced'
    df_original = pd.DataFrame()
    if data_set_name == 'adult':
        df_original = pd.read_csv(adult_data_set_dir + '/' + adult_data_set_csv_file_name + '.csv')
        target = 'income'
        df_original.loc[df_original[target] == "<=50K", target] = 0
        df_original.loc[df_original[target] == ">50K", target] = 1
        df_original.replace('?', np.NaN,inplace=True)
        df_original.dropna(axis=0,how='any',inplace=True)
    elif data_set_name == 'credit_card':
        if balanced:
            df_original = pd.read_csv(credit_card_data_set_dir + '/' + balaned_credit_card_data_set_csv_file_name + '.csv')
        else:
            df_original = pd.read_csv(credit_card_data_set_dir + '/' + unbalaned_credit_card_data_set_csv_file_name + '.csv')
    else:
        raise ValueError("Invalid data set name: " + data_set_name)
    return df_original

def save_test_train_data(data_set_name, df_train, df_test, balanced:bool=False):
    adult_data_set_dir = "../data/adult"
    credit_card_data_set_dir = "../data/credit_card"
    adult_data_set_csv_file_name = 'adult'
    balaned_credit_card_data_set_csv_file_name = 'credit_card_balanced'
    unbalaned_credit_card_data_set_csv_file_name = 'credit_card_unbalanced'
    if data_set_name == 'adult':
        df_train.to_csv(adult_data_set_dir + '/' + adult_data_set_csv_file_name + '_train.csv', index=False)
        df_test.to_csv(adult_data_set_dir + '/' + adult_data_set_csv_file_name + '_testcsv', index=False)
    elif data_set_name == 'credit_card':
        if balanced:
            df_train.to_csv(credit_card_data_set_dir + '/' + balaned_credit_card_data_set_csv_file_name + '_train.csv')
            df_test.to_csv(credit_card_data_set_dir + '/' + balaned_credit_card_data_set_csv_file_name + '_test.csv')
        else:
            df_train.to_csv(credit_card_data_set_dir + '/' + unbalaned_credit_card_data_set_csv_file_name + '_train.csv')
            df_test.to_csv(credit_card_data_set_dir + '/' + unbalaned_credit_card_data_set_csv_file_name + '_test.csv')
    else:
        raise ValueError("Invalid data set name: " + data_set_name)
    
# Function to fit the synthesizers
def fit_synth(df, params):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)
    method = params['method']
    # print(params, method)
    if method == "GaussianCopula":
        synth = GaussianCopulaSynthesizer(metadata=metadata)
    elif method == "CTGAN" or method =="CopulaGAN":
        epoch = params['epochs']
        batch_size = int(params['batch_size'])*100
        if int(params["g_dim3"]) != 0:
            generator_dim = (128*int(params['g_dim1']), 128*int(params['g_dim2']), 128*int(params['g_dim3']))
        else:
            generator_dim = (128*int(params['g_dim1']), 128*int(params['g_dim2']))
        if int(params["d_dim3"]) != 0:
            discriminator_dim = (128*int(params['d_dim1']), 128*int(params['d_dim2']), 128*int(params['d_dim3']))
        else:
            discriminator_dim = (128*int(params['d_dim1']), 128*int(params['d_dim2']))
        # discriminator_lr = params['d_lr']
        # generator_lr = params['g_lr']
        if method == "CTGAN":
            synth = CTGANSynthesizer(metadata=metadata, epochs=epoch, batch_size=batch_size, generator_dim=generator_dim, 
                                     discriminator_dim=discriminator_dim)#, generator_lr=generator_lr, discriminator_lr=discriminator_lr, verbose=False)
        if method == "CopulaGAN":
            synth = CopulaGANSynthesizer(metadata=metadata, epochs=epoch, batch_size=batch_size, generator_dim=generator_dim,
                                         discriminator_dim=discriminator_dim) #, generator_lr=generator_lr, discriminator_lr=discriminator_lr, verbose=False)
    elif method == "TVAE":
        epoch = params['epochs']
        batch_size = int(params['batch_size'])*100
        if int(params["c_dim3"]) != 0:
            compress_dims = (64*int(params['c_dim1']), 64*int(params['c_dim2']), 64*int(params['c_dim3']))
        else:
            compress_dims = (64*int(params['c_dim1']), 64*int(params['c_dim2']))
        if int(params["d_dim3"]) != 0:
            decompress_dims = (64*int(params['d_dim1']), 64*int(params['d_dim2']), 64*int(params['d_dim3']))
        else:
            decompress_dims = (64*int(params['d_dim1']), 64*int(params['d_dim2']))
        synth = TVAESynthesizer(metadata=metadata, epochs=epoch, batch_size=batch_size, compress_dims=compress_dims, 
                                 decompress_dims=decompress_dims)
    else:
        raise ValueError("Invalid model name: " + method)
    return synth

# Function for downstream loss calculation
def downstream_loss(sampled, df_val, target, classifier = "XGB"):
    params_xgb = {
        'eval_metric': 'auc', 'objective':'binary:logistic', 'seed': 5,
    }
    
    x_samp = sampled.loc[:, sampled.columns != target]
    y_samp = sampled[target]
    x_val = df_val.loc[:, sampled.columns != target]
    y_val = df_val[target]
    if classifier == "XGB":
        for column in x_samp.columns:
            if x_samp[column].dtype == 'object':
                x_samp[column] = x_samp[column].astype('category')
                x_val[column] = x_val[column].astype('category')
        dtrain = xgb.DMatrix(data=x_samp, label=y_samp, enable_categorical=True)
        dval = xgb.DMatrix(data=x_val, label=y_val, enable_categorical=True)
        clf = xgb.train(params_xgb, dtrain, 1000, verbose_eval=False)
        clf_probs = clf.predict(dval)
        clf_auc = roc_auc_score(y_val.values.astype(float), clf_probs)
        return clf, clf_auc
    else:
        raise ValueError("Invalid classifier: " + classifier)
        



# Get parameters depending on the synthesizer
def getparams(method_name):
    epoch = 150
    if method_name == 'GaussianCopula':
        return {}
    elif method_name == 'CTGAN':
        params_range = {
        'N_sim': 10000,
        'loss': 'ROCAUC',
        'method': method_name,
        'epochs':  epoch,  
        'batch_size': hp.randint('batch_size',1, 5), # multiple of 100
        'g_dim1':  hp.randint('g_dim1',1, 3), # multiple of 128
        'g_dim2':  hp.randint('g_dim2',1, 2), # multiple of 128
        'g_dim3':  hp.randint('g_dim3',0, 1), # multiple of 128
        'd_dim1':  hp.randint('d_dim1',1, 3), # multiple of 128
        'd_dim2':  hp.randint('d_dim2',1, 2), # multiple of 128
        'd_dim3':  hp.randint('d_dim3',0, 1), # multiple of 128
        'd_lr': 2e-4, #hp.uniform('d_lr', 5e-5, 1e-2),
        "g_lr": 2e-4, #hp.uniform('g_lr', 5e-5, 1e-2),
        } 
        return params_range
    elif method_name == "CopulaGAN":
        params_range = {
        'N_sim': 10000,
        'loss': 'ROCAUC',
        'method': method_name,
        'epochs':  epoch,  
        'batch_size': hp.randint('batch_size',1, 5), # multiple of 100
        'g_dim1':  hp.randint('g_dim1',1, 3), # multiple of 128
        'g_dim2':  hp.randint('g_dim2',1, 2), # multiple of 128
        'g_dim3':  hp.randint('g_dim3',0, 1), # multiple of 128
        'd_dim1':  hp.randint('d_dim1',1, 3), # multiple of 128
        'd_dim2':  hp.randint('d_dim2',1, 2), # multiple of 128
        'd_dim3':  hp.randint('d_dim3',0, 1), # multiple of 128
        'd_lr': 2e-4,
        "g_lr": 2e-4,
        } 
        return params_range
    else:
        params_range = {
        'N_sim': 10000,
        'loss': 'ROCAUC',
        'method': method_name,
        'epochs':  epoch,
        'batch_size':  5, #hp.randint('batch_size',1, 5), # multiple of 100
        'c_dim1':  hp.randint('c_dim1',1, 3), # multiple of 64
        'c_dim2':  hp.randint('c_dim2',1, 2), # multiple of 64
        'c_dim3':  hp.randint('c_dim3',0, 1), # multiple of 64
        'd_dim1':  hp.randint('d_dim1',1, 3), # multiple of 64
        'd_dim2':  hp.randint('d_dim2',1, 2), # multiple of 64
        'd_dim3':  hp.randint('d_dim3',0, 1), # multiple of 64
        } 
        return params_range

# Get inirial params
def get_init_params(method_name):
    epoch = 150
    if method_name == 'GaussianCopula':
        params = {
        'N_sim': 10000,
        'method': method_name
        } 
        return params
    elif method_name == 'CTGAN':
        params = {
        'N_sim': 10000,
        'loss': 'ROCAUC',
        'method': method_name,
        'epochs':  epoch,  
        'batch_size':  5,
        'g_dim1':  2,
        'g_dim2':  2,
        'g_dim3':  0,
        'd_dim1':  2,
        'd_dim2':  2,
        'd_dim3':  0,
        'd_lr': 2e-4,
        "g_lr": 2e-4,
        } 
        return params
    elif method_name == "CopulaGAN":
        params = {
        'N_sim': 10000,
        'loss': 'ROCAUC',
        'method': method_name,
        'epochs':  epoch,  
        'batch_size':  5, #hp.randint('batch_size',1, 5), # multiple of 100
        'g_dim1':  2,
        'g_dim2':  2,
        'g_dim3':  0,
        'd_dim1':  2,
        'd_dim2':  2,
        'd_dim3':  0,
        'd_lr': 2e-4,
        "g_lr": 2e-4,
        } 
        return params
    else:
        params = {
        'N_sim': 10000,
        'loss': 'ROCAUC',
        'method': method_name,
        'epochs':  epoch,
        'batch_size':  5, #hp.randint('batch_size',1, 5), # multiple of 100
        'c_dim1':  2,
        'c_dim2':  2,
        'c_dim3':  0,
        'd_dim1':  2,
        'd_dim2':  2,
        'd_dim3':  0,
        } 
        return params

def get_tuned_params(method_name, data_set_name,outpath):
    fname = outpath + data_set_name + "_" + method_name + "_clf_best_hp_xgboost.csv"
    dfp = pd.read_csv(fname, index_col=0)
    params = dfp['Value'].to_dict()
    # Keys to remove
    keys_to_remove = ['tuned_val_roc', 'untuned_val_roc', 'tuned_test_roc', 'untuned_test_roc', 'elapsed_time']
    params = {key: value for key, value in params.items() if key not in keys_to_remove}
    return params

def get_alpha_Params(augment_data_percentage:float):
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

def computeInitialAlphaForWarmStart(augment_data_percentage:float, val_auc):    
    val_auc_temp = [val_auc[i] - min(val_auc) + 1e-3 for i in range(len(val_auc))]
    scale = sum(val_auc_temp) / (1 - augment_data_percentage)
    alphas = [(1 / scale) * val_auc_temp[i] for i in range(len(val_auc_temp))]
    return alphas


def save_synthetic_data(data_set_name:str, best_X_synthetic, best_y_synthetic, balanced, encode, tuned, augment_data_percentage,
 experiment, outpath_data_augmentation):
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

import os

import os

def remove_folders_with_string(root_directory, target_string):
    for root, dirs, files in os.walk(root_directory, topdown=False):
        for folder in dirs:
            if target_string in folder:
                folder_path = os.path.join(root, folder)
                print("Removing folder:", folder_path)
                try:
                    os.rmdir(folder_path)
                    print("Folder removed successfully.")
                except Exception as e:
                    print("Error occurred while removing folder:", str(e))




