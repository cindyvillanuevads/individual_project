

import pandas as pd
import numpy as np
import os
import seaborn as sns

# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")




import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer





def miss_dup_values(df):
    '''
    takes in a dataframe of observations and attributes and returns a dataframe where each row is an atttribute name, 
    the first column is the number of rows with missing values for that attribute, 
    and the second column is percent of total rows that have missing values for that attribute and 
    duplicated rows.
    '''
        # Total missing values
    mis_val = df.isnull().sum()
        # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
        #total of duplicated
    dup = df.duplicated().sum()  
        # Percentage of missing values
    dup_percent = 100 * dup / len(df)
        # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
    mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
        # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
           "There are " + str(mis_val_table_ren_columns.shape[0]) +
           " columns that have missing values.")
    print( "  ")
    print (f"** There are {dup} duplicate rows that represents {round(dup_percent, 2)}% of total Values**")
        # Return the dataframe with missing information
    return mis_val_table_ren_columns


def split_data(df, target):
    '''
        This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes
    
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate[target])
    
    
    print(f'complete df -> {df.shape}')
    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')

    return train, validate, test


def impute_mode(train, validate, test):
    '''
    take in train, validate, and test DataFrames, impute mode for 'loanamount',
    and return train, validate, and test DataFrames
    '''
    imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
    train[['loanamount']] = imputer.fit_transform(train[['loanamount']])
    validate[['loanamount']] = imputer.transform(validate[['loanamount']])
    test[['loanamount']] = imputer.transform(test[['loanamount']])
    return train, validate, test


# plot distributions
def distribution (df):
    '''
    takes in a df and plot individual variable distributions excluding object type
    '''

    cols =df.columns.to_list()
    for col in cols:
        if df[col].dtype != 'object':
            plt.hist(df[col])
            plt.title(f'Distribution of {col}')
            plt.xlabel('values')
            plt.ylabel('Counts of customers')
            plt.show()




def distribution_boxplot (df):
    '''
    takes in a df and boxplot variable distributions excluding object type
    '''
    cols =df.columns.to_list()
    for col in cols:
        if df[col].dtype != 'object':
            plt.figure(figsize=(8, 6))
            sns.boxplot(x= col, data=df)
            plt.title(f'Distribution of {col}')
            plt.xlabel('values')
            plt.show()
    return
