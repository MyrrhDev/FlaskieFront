import pandas as pd
import time

def findNoneColumns(df):
    none_cols = []
    for i in df.columns:
        if "_none" in i:
            none_cols.append(i)

    return none_cols


def findActivityColumns(df):
    act_cols = []
    for i in df.columns:
        if (("Visit_" in i) or ("Phone_" in i) or ("Services_" in i) or ("Mail_" in i) or ("Events_" in i) or (
                "Sales_" in i)):
            act_cols.append(i)

    return act_cols


def applyCorrelationAnalysis(custom, current_year):
    print()
    a = time.time()
    df = custom.copy()
    none_cols = findNoneColumns(df)
    df = df.drop(columns = none_cols)

    act_cols = findActivityColumns(df)
    corrDF =  df.drop(columns=act_cols)
    corrmat = corrDF.corr()

    columns_all = corrmat.columns.tolist()
    corrmat.dropna(axis=0, how='all', inplace=True)
    corrmat.dropna(axis=1, how='all', inplace=True)
    columns = corrmat.columns.tolist()
    nan_columns = list(set(columns_all) - set(columns))
    df = df.drop(columns=nan_columns)

    categorical = list(set(df.columns) - set(corrmat.columns) - set(act_cols))
    most_corr = corrmat["is_active_t"].sort_values(ascending = False)[0:50].keys()
    cols = list(set(list(most_corr) + act_cols + ["Season_year"] + categorical))
    finalDF = df[cols]

    #finalDF.to_csv("../data/output/exploration_output/correlated_variables.csv")
    print("applyCorrelationAnalysis  in  done in " + str((time.time() - a) / 60) + " minutes")

    return finalDF

