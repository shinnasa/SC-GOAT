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
                    self.encoded_columns = self.encoded_columns.append(col + '_target_encoded')
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
                        self.encoded_columns = self.encoded_columns.append(column + '_target_encoded')
                    self.column_value_to_column_encoding[column] = X[self.target_column].groupby(X[column]).agg(['count', 'mean'])
                    self.column_encoding_to_column_value[column] = {}
                    for column_unique_val in self.column_value_to_column_encoding[column].index:
                        self.column_encoding_to_column_value[col][self.column_value_to_column_encoding[col].loc[column_unique_val]['mean']] = column_unique_val
                        output.loc[output[column] == column_unique_val, column + '_target_encoded'] = self.column_value_to_column_encoding[column].loc[column_unique_val]['mean']
        output = output.drop(columns=self.columns)
        return output
    def inverse_transform(self,X):
        output = X.copy()
        for index, column in enumerate(self.encoded_columns):
            for column_unique_val in output[column].unique():
                output.loc[output[column] == column_unique_val, self.columns[index]] = self.column_encoding_to_column_value[self.columns[index]][column_unique_val]
        output = output.drop(columns=self.encoded_columns)
        return output 
        
