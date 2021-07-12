import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from io import StringIO
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from IPython.display import display, display_html 

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE



def select_rfe (X_df, y_df, n_features, method):
    '''
    Takes in the predictors, the target, and the number of features to select (k) ,
    and returns the names of the top k selected features based on the Recursive Feature Elimination (RFE)
    
    X_df : the predictors
    y_df : the target
    n_features : the number of features to select (k)
    method : LinearRegression, LassoLars, TweedieRegressor
    Example
    select_rfe(X_train_scaled, y_train, 2, LinearRegression())
    '''
    lm = method
    rfe = RFE(estimator=lm, n_features_to_select= n_features)
    rfe.fit(X_df, y_df)
    top_rfe = list(X_df.columns[rfe.support_])
    print(f'The top {n_features} selected feautures based on the the RFE class class are: {top_rfe}' )
    print(pd.Series(dict(zip(X_df.columns, rfe.ranking_))).sort_values())
    return top_rfe

def select_kbest  (X_df, y_df, n_features):
    '''
    Takes in the predictors, the target, and the number of features to select (k) ,
    and returns the names of the top k selected features based on the SelectKBest class
    
    X_df : the predictors
    y_df : the target
    n_features : the number of features to select (k)
    Example
    select_kbest(X_train_scaled, y_train, 2)
    '''
    
    f_selector = SelectKBest(score_func=f_classif, k= n_features)
    f_selector.fit(X_df, y_df)
    mask = f_selector.get_support()
    X_df.columns[mask]
    top = list(X_df.columns[mask])
    print(f'The top {n_features} selected feautures based on the SelectKBest class are: {top}' )
    return top


def model_performs (X_df, y_df, model):
    '''
    Take in a X_df, y_df and model  and fit the model , make a prediction, calculate score (accuracy), 
    confusion matrix, rates, clasification report.
    X_df: train, validate or  test. Select one
    y_df: it has to be the same as X_df.
    model: name of your model that you prevously created 
    
    Example:
    mmodel_performs (X_train, y_train, model1)
    '''

    #prediction
    pred = model.predict(X_df)

    #score = accuracy
    acc = model.score(X_df, y_df)

    #conf Matrix
    conf = confusion_matrix(y_df, pred)
    mat =  pd.DataFrame ((confusion_matrix(y_df, pred )),index = ['actual_no_approved','actual_approved'], columns =['pred_no_approved','pred_approved' ])
    rubric_df = pd.DataFrame([['True Negative', 'False positive'], ['False Negative', 'True Positive']], columns=mat.columns, index=mat.index)
    cf = rubric_df + ': ' + mat.values.astype(str)

    #assign the values
    tp = conf[1,1]
    fp =conf[0,1] 
    fn= conf[1,0]
    tn =conf[0,0]

    #calculate the rate
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    tnr = tn/(tn+fp)
    fnr = fn/(fn+tp)

    #classification report
    clas_rep =pd.DataFrame(classification_report(y_df, pred, output_dict=True)).T
    clas_rep.rename(index={'0': "No Aproved", '1': "Approved"}, inplace = True)
    print(f'''
    The accuracy for our model is {acc:.4%}
    The True Positive Rate is {tpr:.3%},    The False Positive Rate is {fpr:.3%},
    The True Negative Rate is {tnr:.3%},    The False Negative Rate is {fnr:.3%}
    ________________________________________________________________________________
    ''')
    print('''
    The positive is  'Loan Approved '
    Confusion Matrix
    ''')
    display(cf)
    print('''
    ________________________________________________________________________________
    
    Classification Report:
    ''')
    display(clas_rep)
   


def compare (model1, model2, X_df1,X_df2, y_df):
    '''
    Take in two models to compare their performance metrics.
    X_df: train, validate or  test. Select one
    y_df: it has to be the same as X_df.
    model1: name of your first model that you want to compare  
    model2: name of your second model that you want to compare
    Example: 
    compare(logit2, logit4, X_validate, y_validate)
    '''



    #prediction
    pred1 = model1.predict(X_df1)
    pred2 = model2.predict(X_df2)

    #score = accuracy
    acc1 = model1.score(X_df1, y_df)
    acc2 = model2.score(X_df2, y_df)


    #conf Matrix
    #model 1
    conf1 = confusion_matrix(y_df, pred1)
    mat1 =  pd.DataFrame ((confusion_matrix(y_df, pred1 )),index = ['actual_no_churn','actual_churn'], columns =['pred_no_churn','pred_churn' ])
    rubric_df = pd.DataFrame([['True Negative', 'False positive'], ['False Negative', 'True Positive']], columns=mat1.columns, index=mat1.index)
    cf1 = rubric_df + ': ' + mat1.values.astype(str)
    
    #model2
    conf2 = confusion_matrix(y_df, pred2)
    mat2 =  pd.DataFrame ((confusion_matrix(y_df, pred2 )),index = ['actual_no_churn','actual_churn'], columns =['pred_no_churn','pred_churn' ])
    cf2 = rubric_df + ': ' + mat2.values.astype(str)
    #model 1
    #assign the values
    tp = conf1[1,1]
    fp =conf1[0,1] 
    fn= conf1[1,0]
    tn =conf1[0,0]

    #calculate the rate
    tpr1 = tp/(tp+fn)
    fpr1 = fp/(fp+tn)
    tnr1 = tn/(tn+fp)
    fnr1 = fn/(fn+tp)

    #model 2
    #assign the values
    tp = conf2[1,1]
    fp =conf2[0,1] 
    fn= conf2[1,0]
    tn =conf2[0,0]

    #calculate the rate
    tpr2 = tp/(tp+fn)
    fpr2 = fp/(fp+tn)
    tnr2 = tn/(tn+fp)
    fnr2 = fn/(fn+tp)

    #classification report
    #model1
    clas_rep1 =pd.DataFrame(classification_report(y_df, pred1, output_dict=True)).T
    clas_rep1.rename(index={'0': "no_churn", '1': "churn"}, inplace = True)

    #model2
    clas_rep2 =pd.DataFrame(classification_report(y_df, pred2, output_dict=True)).T
    clas_rep2.rename(index={'0': "no_churn", '1': "churn"}, inplace = True)
    print(f'''
    ******       Model 1  ******                                ******     Model 2  ****** 
    The accuracy for our model 1 is {acc1:.4%}            |   The accuracy for our model 2 is {acc2:.4%}  
                                                        |
    The True Positive Rate is {tpr1:.3%}                   |   The True Positive Rate is {tpr2:.3%}  
    The False Positive Rate is {fpr1:.3%}                  |   The False Positive Rate is {fpr2:.3%} 
    The True Negative Rate is {tnr1:.3%}                   |   The True Negative Rate is {tnr2:.3%} 
    The False Negative Rate is {fnr1:.3%}                  |   The False Negative Rate is {fnr2:.3%}
    _____________________________________________________________________________________________________________
    ''')
    print('''
    The positive is  'churn'
    Confusion Matrix
    ''')
    cf1_styler = cf1.style.set_table_attributes("style='display:inline'").set_caption('Model 1')
    cf2_styler = cf2.style.set_table_attributes("style='display:inline'").set_caption('Model2')
    space = "\xa0" * 50
    display_html(cf1_styler._repr_html_()+ space  + cf2_styler._repr_html_(), raw=True)
    # print(display(cf1),"           ", display(cf2))
    
    print('''
    ________________________________________________________________________________
    
    Classification Report:
    ''')
     
    clas_rep1_styler = clas_rep1.style.set_table_attributes("style='display:inline'").set_caption('Model 1 Classification Report')
    clas_rep2_styler = clas_rep2.style.set_table_attributes("style='display:inline'").set_caption('Model 2 Classification Report')
    space = "\xa0" * 45
    display_html(clas_rep1_styler._repr_html_()+ space  + clas_rep2_styler._repr_html_(), raw=True)
   
