from create_datasets.customer_dataset_functions import *
from create_datasets.corteva import buildCortevaDataset
from create_datasets.product import buildProductDataset
from create_datasets.retailer import buildRetailerDataset
from create_datasets.farmer import buildFarmerDataset
from create_datasets.activity import buildActivityDataset
from create_datasets.correlation_analysis import applyCorrelationAnalysis
import time


def buildGlobalCustomerDataset(transactionalDF, activityRawDF, current_year):
    a = time.time()
    corteva = buildCortevaDataset(transactionalDF, current_year)
    products = buildProductDataset(transactionalDF, current_year)
    retailers = buildRetailerDataset(transactionalDF, current_year)
    farmers = buildFarmerDataset(transactionalDF, current_year)
    activity = buildActivityDataset(activityRawDF, current_year)

    farmers.gl_popular_retailer_name = farmers["gl_popular_retailer_name"].astype(str)
    farmers = farmers.fillna(0)

    products["year"] = products["year"].str.extract("(\d+)", expand=False)
    products = products.fillna(0)
    products['year'] = products['year'].apply(lambda x: int(x) + 1)
    products.drop(columns=['quantity', 'top_quantity'], axis=1, inplace=True)

    df_merged = pd.merge(farmers, products, how='left', left_on=['year', 'gl_popular_product_name'],
                         right_on=['year', 'product'])
    df_merged.drop(columns=['product'], axis=1, inplace=True)
    df_merged = df_merged.fillna(0)

    retailers.Soldto = retailers["Soldto"].astype(str)
    retailers = retailers.fillna(0)

    df_merged = pd.merge(df_merged, retailers, how='left', left_on=['year', 'gl_popular_retailer_name'],
                         right_on=['year', 'Soldto'])
    df_merged.drop(columns=['Soldto'], axis=1, inplace=True)
    df_merged = df_merged.fillna(0)

    corteva['year'] = corteva['year'] + 1
    corteva = corteva.fillna(0)

    df_merged = pd.merge(df_merged, corteva, how='left', on=['year'])
    df_merged = df_merged.fillna(0)

    df_merged = df_merged.rename(columns={"Farmer": "ID", "year": "Season_year"})


    df_merged = pd.merge(df_merged, activity, how="left", on=["ID", "Season_year"])
    df_merged = df_merged.fillna(0)

    customer_dataset = df_merged.copy()



    customer_dataset.reset_index(drop=True, inplace=True)

    print("global customer dataset is built in " + str((time.time() - a) / 60) + " minutes")

    return customer_dataset
