import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def bar_plot (features, df):
    '''
    Take in a features (max 4) to plot  and churn_rate.
    '''
    churn_rate = df['loan_status'].mean()
    lc = len(features)
    _, ax = plt.subplots(nrows=1, ncols=lc, figsize=(16, 6), sharey=True)
    for i, feature in enumerate(features):
        sns.barplot(feature, 'loan_status', data=df, ax=ax[i], alpha=0.5, saturation=1)
        ax[i].set_xlabel('Loan_Status')
        ax[i].set_ylabel(f'Loan Approval Rate ={churn_rate:.2%}')
        ax[i].set_title(feature)
        ax[i].axhline(churn_rate, ls='--', color='red')
