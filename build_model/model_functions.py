import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

def import_data(path):
    print('importing data....')
    mylist = []
    for chunk in pd.read_csv(path, encoding='utf-8', chunksize=20000, low_memory=False):
        mylist.append(chunk)
    df = pd.concat(mylist, axis=0)
    del mylist
    return df


def preprocess_data(df_aux, categorical_cols):
    df = df_aux.copy()
    df.drop(columns=['ID'], axis=1, inplace=True)
    numerical_cols = list(set(df.columns) - set(categorical_cols))
    numerical_cols.remove('Season_year')

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = df[col].astype('str')
        df[col] = le.fit_transform(df[col])

    min_max_scaler = preprocessing.MinMaxScaler()
    df[numerical_cols] = min_max_scaler.fit_transform(df[numerical_cols])

    return (df, numerical_cols, categorical_cols)


def split_data(df, current_year):
    train = df[(df.Season_year != current_year)]
    test = df[(df.Season_year == current_year)]

    train_y = train['is_active_t']
    test_y = test['is_active_t']
    train.drop(columns=['is_active_t', 'Season_year'], axis=1, inplace=True)
    test.drop(columns=['is_active_t', 'Season_year'], axis=1, inplace=True)

    return (train, train_y, test, test_y)


#returns model obj
def build_model(inputDF, current_year):
    df = inputDF.copy()
    categorical_columns = ['gl_Region','RET_gl_Region','gl_category','RET_gl_SalesArea_p','gl_category',
                           'gl_SalesArea_p','gl_popular_product_name','gl_popular_month','gl_popular_retailer_name']

    df, numerical_cols, categorical_cols = preprocess_data(df, categorical_columns)
    train, train_y, test, test_y = split_data(df, current_year)
    #model_lxgb_cat = sklearn.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,importance_type='split', learning_rate=0.05, max_depth=25, min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0, n_estimators=200, n_jobs=-1, num_leaves=900, objective=None, random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
    #model_lxgb_cat.fit(train,train_y, categorical_feature = categorical_cols)
    modelGradientBoosting =  GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=5,
              max_features=None, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=50,
              n_iter_no_change=None, presort='auto', random_state=None,
              subsample=1.0, tol=0.0001, validation_fraction=0.1,
              verbose=0, warm_start=False)
    modelGradientBoosting.fit(train,train_y)

    return modelGradientBoosting


def predict(inputDF, model):
    predictionDF = inputDF.copy()
    categorical_columns = ['gl_Region', 'RET_gl_Region', 'gl_category', 'RET_gl_SalesArea_p', 'gl_category', 'gl_SalesArea_p', 'gl_popular_product_name', 'gl_popular_month', 'gl_popular_retailer_name']
    predictionDF, numerical_cols, categorical_cols = preprocess_data(predictionDF, categorical_columns)
    predictionDF.drop(columns=['is_active_t', 'Season_year'], axis=1, inplace=True)

    y_predicted = model.predict(predictionDF)
    y_predicted_probability = model.predict_proba(predictionDF)
    y_predicted_probability = np.amax(y_predicted_probability, axis=1)

    res = inputDF.copy()
    res["prediction"] = y_predicted
    res["probability"]= y_predicted_probability
    res = res[['ID','prediction','probability']]

    res.reset_index(drop=True, inplace=True)

    return res
