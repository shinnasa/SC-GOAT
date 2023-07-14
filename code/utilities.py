import pandas as pd
import numpy as np

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
    
    def inverse_transform(self,X):
        output = X.copy()
        for index, column in enumerate(self.encoded_columns):
            for column_unique_val in output[column].unique():
                output.loc[output[column] == column_unique_val, self.columns[index]] = self.column_encoding_to_column_value[self.columns[index]][column_unique_val]
        output = output.drop(columns=self.encoded_columns)
        return output 
        

adult_data_set_dir = "../data/adult"
credit_card_data_set_dir = "../data/credit_card"
adult_data_set_csv_file_name = 'adult'
balaned_credit_card_data_set_csv_file_name = 'credit_card_balanced'
unbalaned_credit_card_data_set_csv_file_name = 'credit_card_unbalanced'

def load_data(data_set_name:str, balanced:bool=False):
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
    
