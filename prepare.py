

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
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler





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







def scaled_df ( train_df , validate_df, test_df,columns,  scaler):
    '''
    Take in a 3 df and a type of scaler that you  want to  use. it will scale all columns
    except object type. Fit a scaler only in train and tramnsform in train, validate and test.
    returns  new dfs with the scaled columns.
    scaler : MinMaxScaler() or RobustScaler(), StandardScaler() 
    Example:
    scaled_df( X_train , X_validate , X_test, RobustScaler())
    
    '''
    
    # fit our scaler
    scaler.fit(train_df[columns])
    # get our scaled arrays
    train_scaled = scaler.transform(train_df[columns])
    validate_scaled= scaler.transform(validate_df[columns])
    test_scaled= scaler.transform(test_df[columns])

    # convert arrays to dataframes
    train_scaled_df = pd.DataFrame(train_scaled, columns=columns).set_index([train_df.index.values])
    validate_scaled_df = pd.DataFrame(validate_scaled, columns=columns).set_index([validate_df.index.values])
    test_scaled_df = pd.DataFrame(test_scaled, columns=columns).set_index([test_df.index.values])

    #add the columns that are not scaled
    train_scaled_df = pd.concat([train_scaled_df, train_df.drop(columns = columns) ], axis= 1 )
    validate_scaled_df = pd.concat([validate_scaled_df, validate_df.drop(columns = columns) ], axis= 1 )
    test_scaled_df = pd.concat([test_scaled_df, test_df.drop(columns = columns) ], axis= 1 )
    #plot
    for col in columns: 
        plt.figure(figsize=(13, 6))
        plt.subplot(121)
        plt.hist(train_df[col], ec='black')
        plt.title('Original')
        plt.xlabel(col)
        plt.ylabel("counts")
        plt.subplot(122)
        plt.hist(train_scaled_df[col],  ec='black')
        plt.title('Scaled')
        plt.xlabel(col)
        plt.ylabel("counts")



    return train_scaled_df, validate_scaled_df, test_scaled_df






def split_Xy (train, validate, test, target):
    '''
    This function takes in three dataframe (train, validate, test) and a target  and splits each of the 3 samples
    into a dataframe with independent variables and a series with the dependent, or target variable.
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test.
    Example:
    X_train, y_train, X_validate, y_validate, X_test, y_test = split_Xy (train, validate, test, 'Fertility' )
    '''
    
    #split train
    X_train = train.drop(columns= [target])
    y_train= train[target]
    #split validate
    X_validate = validate.drop(columns= [target])
    y_validate= validate[target]
    #split validate
    X_test = test.drop(columns= [target])
    y_test= test[target]

    print(f'X_train -> {X_train.shape}               y_train->{y_train.shape}')
    print(f'X_validate -> {X_validate.shape}         y_validate->{y_validate.shape} ')        
    print(f'X_test -> {X_test.shape}                  y_test>{y_test.shape}') 
    return  X_train, y_train, X_validate, y_validate, X_test, y_test


# ************************* acquire _clean() *************************

def aquire_clean ():
    '''
    this functions read df1_loan.csv and clean the df. renames columns, chanfe type of columns, fill missing values
    create dummies and creates new columns
    
    return df
    '''
    #read the file that is  already in my directory
    df = pd.read_csv('df1_loan.csv')
    #drop unnamed column
    df = df.drop(columns= ['Unnamed: 0'])
    #rename columns (lowercase)
    df.columns = map(str.lower, df.columns)
    #change total_income to float
    df['total_income']= df.total_income.replace({'\$':''}, regex = True).astype('float')
    #change type in dependents
    df['dependents']= df.dependents.replace({'\+':''}, regex = True).astype('float')
    #fill missing values of the columns with 0
    df[['credit_history','self_employed','dependents', 'married']] =df[['credit_history','self_employed','dependents', 'married']].fillna(0)
    #fill  missing values of gender
    df.gender.fillna('Male', inplace = True)
    #fill  missing values of loan_amount_term
    df.loan_amount_term.fillna(360, inplace = True)
    #loanamount values are thousands 
    df['loanamount'] = df.loanamount * 1000
    # encode yes/no columns : loan_status, self_employed, married
    df[['loan_status', 'self_employed', 'married']] = df[['loan_status', 'self_employed', 'married']].replace({'Yes': 1, 'No': 0, 'Y': 1, 'N': 0})
    #encode gender female = 0 , male = 1
    df.gender.replace({'Male': 1, 'Female': 0}, inplace = True)
    #create dummies
    dummy_df = pd.get_dummies(df[['education','property_area']])
    #lowercase the columns
    dummy_df.columns = map(str.lower, dummy_df.columns)
    #concat 2dfs
    df = pd.concat([df, dummy_df], axis=1).drop(columns = ['education', 'property_area','education_not graduate'])
    #create a new columns
    df['has_coapplicant'] = 0 
    #fill new column with 1 for that condition
    df.loc[df.coapplicantincome > 0, ['has_coapplicant']] = 1
    #  total_income divide by  dependents +1(applicant)
    df['income_portion_dependents'] = df.total_income/ (df.dependents +1) 
    
    return df

# ************************* split_and_scaled() *************************

def split_and_scaled (df, cols_tobe_scaled, scaler ):
    '''
    takes in a df, a list of columns to be scaled, and a scale type. split df into train, validate and test.
    scale the selected columns using the selected scale.
    returns train, validate, test, train_scaled, validate_scaled, test_scaled
    df: dataframe
    cols_tobe_scaled : [ 'applicantincome', 'coapplicantincome', 'loanamount','loan_amount_term', 'total_income', 'income_portion_dependents']
    scaler : MinMaxScaler() or RobustScaler(), StandardScaler()
    Example
    split_and_scaled (df1, cols_tobe_scaled, RobustScaler())
    '''
    #using my function that is in prepare.py
    train, validate, test = split_data(df, 'loan_status')
    #using a function to impute the mean
    train, validate, test = impute_mode(train, validate, test)
    train_scaled, validate_scaled, test_scaled = scaled_df(train, validate, test, cols_tobe_scaled , scaler)
    return (train, validate, test, train_scaled, validate_scaled, test_scaled)  