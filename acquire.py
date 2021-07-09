
import pandas as pd
import numpy as np
import os


# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")


# ************************************ unique values ***************************************************************

def report_unique_val (df):
    '''
    takes in a df and gives you a report of number of unique values and count values <15 (categorical)
    count values <15 (numerical)
    '''
    num_cols = df.select_dtypes(exclude = 'O').columns.to_list()
    cat_cols = df.select_dtypes(include = 'O').columns.to_list()
    for col in df.columns:
            print(f'**{col}**')
            le = df[col].nunique()
            print ('Unique Values : ', df[col].nunique())
            print(' ')
            if col in cat_cols and le < 15:
                print(df[col].value_counts())
            if col in num_cols and  le < 23:
                 print(df[col].value_counts().sort_index(ascending=True)) 
            elif col in num_cols and le <150:
                print(df[col].value_counts(bins=10, sort=False).sort_index(ascending=True))
            elif col in num_cols and le <1001:
                print(df[col].value_counts(bins=100, sort=False).sort_index(ascending=True))

            print('=====================================================')


# ************************************ sumarize ***************************************************************
def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe) 
    and output to console various statistics on said dataframe, including:
    # shape
    # .head()
    # .info()
    # .describe()
    # value_counts()
    # observation of nulls in the dataframe
    '''
    print('=====================================================')
    print('Dataframe shape: ')
    print(df.shape)
    print('=====================================================')
    print('Dataframe head: ')
    print(df.head(3))
    print('=====================================================')
    print('Dataframe info: ')
    print(df.info())
    print('=====================================================')
    print('Dataframe Description: ')
    print(df.describe().T)
    print('=====================================================')
    report_unique_val (df)


def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on churn.
    
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.churn)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.churn)
    return train, validate, test