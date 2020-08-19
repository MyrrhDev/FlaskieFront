import pandas as pd
import numpy as np
from datetime import datetime
import time
import sklearn.preprocessing
from create_datasets.customer_dataset_functions import *


def buildCortevaDataset(transactional_df,currentyear=2019):

    a = time.time()
    df = transactional_df.copy()
    df = df[df["Order Qty (Chg + FOC)"] > 0]

    quantity_df = quantity_dataset(df, group_by = "Farmer", quantities = True)

    df_transformed_all = transform(dfaux = quantity_df,
                                   data = "Corteva",
                                   last_year = currentyear,
                                   predicting = False)

    status = df_transformed_all.copy()
    status  = set_status(status)

    cols = ['year', 'churn_vs_previous', 'new_vs_previous', 'retained_vs_previous', 'regained_vs_previous',
            'q_churn_vs_previous', 'q_new_vs_previous', 'q_retained_vs_previous', 'q_regained_vs_previous']
    corteva_df = pd.DataFrame(columns =  cols)

    # number of clients
    regained = status[status['stat'] == 'regained'].groupby('year').count()['t'].to_frame()
    regained.columns = ['regained']

    churn = status[status['stat'] == "churn" ].groupby('year').count()['t'].to_frame()
    churn.columns = ['churn']

    retained = status[status['stat'] == "retained"].groupby('year').count()['t'].to_frame()
    retained.columns = ['retained']

    new = status[status['stat'] == "new"].groupby('year').count()['t'].to_frame()
    new.columns = ['new']


    # quantity
    q_regained = status[status['stat'] == 'regained'].groupby('year').sum()['t'].to_frame()
    q_regained.columns = ['q_regained']

    q_churn = status[status['stat'] == "churn" ].groupby('year').sum()['t-01'].to_frame()
    q_churn.columns = ['q_churn']

    q_retained = status[status['stat'] == "retained"].groupby('year').sum()['t'].to_frame()
    q_retained.columns = ['q_retained']

    q_new = status[status['stat'] == "new"].groupby('year').sum()['t'].to_frame()
    q_new.columns = ['q_new']


    #mean
    m_regained = status[status['stat'] == 'regained'].groupby('year').mean()['t'].to_frame()
    m_regained.columns = ['mean_regained']

    m_churn = status[status['stat'] == "churn"].groupby('year').mean()['t-01'].to_frame()
    m_churn.columns = ['mean_churn']

    m_retained = status[status['stat'] == "retained"].groupby('year').mean()['t'].to_frame()
    m_retained.columns = ['mean_retained']

    m_new = status[status['stat'] == "new"].groupby('year').mean()['t'].to_frame()
    m_new.columns = ['mean_new']

    #max
    max_regained = status[status['stat'] == 'regained'].groupby('year').max()['t'].to_frame()
    max_regained.columns = ['max_regained']

    max_churn = status[status['stat'] == "churn" ].groupby('year').max()['t-01'].to_frame()
    max_churn.columns = ['max_churn']

    max_retained = status[status['stat'] == "retained"].groupby('year').max()['t'].to_frame()
    max_retained.columns = ['max_retained']

    max_new = status[status['stat'] == "new"].groupby('year').max()['t'].to_frame()
    max_new.columns = ['max_new']

    #min
    min_regained = status[status['stat'] == 'regained'].groupby('year').min()['t'].to_frame()
    min_regained.columns = ['min_regained']

    min_churn = status[status['stat'] == "churn" ].groupby('year').min()['t-01'].to_frame()
    min_churn.columns = ['min_churn']

    min_retained = status[status['stat'] == "retained"].groupby('year').min()['t'].to_frame()
    min_retained.columns = ['min_retained']

    min_new = status[status['stat'] == "new"].groupby('year').min()['t'].to_frame()
    min_new.columns = ['min_new']

    corteva_df['year'] = range(2013, currentyear + 1)
    corteva_df = corteva_df.set_index('year')


    result = pd.concat([corteva_df, retained, regained, new, churn,
                        q_retained, q_regained, q_new, q_churn,
                        m_retained, m_regained, m_new, m_churn,
                        max_regained, max_churn, max_retained, max_new,
                        min_regained, min_churn, min_retained, min_new],
                        axis = 1)


    cols_derived = ['churn_vs_previous', 'new_vs_previous', 'retained_vs_previous', 'regained_vs_previous',
                    'q_churn_vs_previous', 'q_new_vs_previous', 'q_retained_vs_previous', 'q_regained_vs_previous']
    cols = ['churn', 'new', 'retained', 'regained', 'q_churn', 'q_new', 'q_retained', 'q_regained']


    result[cols_derived] = (result.loc[:, cols] - result.loc[:, cols].shift(1)) / result.loc[:, cols].shift(1)

    result['churn_vs_retained'] = result['churn'] / result['retained']

    result["churn_vs_others"] = result.apply(churn_vs_others, axis = 1)

    result['year'] = result.index
    result.reset_index(drop=True, inplace=True)

    print("corteva done in " + str((time.time() - a) / 60) + " minutes")
    return result


