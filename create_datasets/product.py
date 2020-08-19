import pandas as pd 
import numpy as np 
import time

from create_datasets.customer_dataset_functions import *


def buildProductDataset(transactional_df, currentyear=2019):
    a = time.time()
    df = transactional_df.copy()
    df = df[df["Order Qty (Chg + FOC)"] > 0]

    quantity = quantity_dataset(df, group_by="Product", quantities=True)
    quantity_group = quantity.sum(axis=0).to_frame()
    res_quantity = quantity_group.loc[:, quantity_group.columns[0]]

    top_regions_season = top_region_per_product(df, filter_by="Region", period="season")
    geo_season = pd.DataFrame(columns=top_regions_season.columns)

    for i in top_regions_season.columns:
        geo_season[i] = top_regions_season[i].value_counts().index[0]

    corn_products = df[df['Product Group'] == 'Corn']
    corn_products_quantity = quantity_dataset(corn_products, group_by="Product", quantities=True)

    corn_products_quantity = corn_products_quantity.loc[(corn_products_quantity != 0).any(1)]

    iterables = [corn_products_quantity.index.get_level_values(0), corn_products_quantity.columns]
    index = pd.MultiIndex.from_product(iterables, names=["product", "year"])
    cols = ['quantity', 'phase', 'age', 'vs_top', 'top_quantity', 'top_season', 'vs_others', 'vs_top_phase']
    phase_df = pd.DataFrame(np.nan, index, cols)

    # Index is (Product, Product Group) and row is Season_
    cols = list(corn_products_quantity.columns)

    for index, row in corn_products_quantity.iterrows():
        start_year = get_start(row, cols)

        for col in cols:
            ind = cols.index(col)
            row1 = row[0:ind + 1].copy()
            top = row1.max()
            top_year = int(row1.idxmax(axis=0).replace("season_", ""))
            current_year = int(col.replace("season_", ""))

            if current_year >= start_year:
                age = current_year - start_year + 1

                vs_top = 0

                if top > 0:
                    vs_top = row1[col] / top

                phase_df.loc[index, col]["vs_top"] = vs_top
                phase_df.loc[index, col]["phase"] = get_phase(row1[col], ind, row1, 2)

                phase_df.loc[index, col]["top_season"] = top_year
                phase_df.loc[index, col]["top_quantity"] = top

                if vs_top > 0.65:
                    phase_df.loc[index, col]["vs_top_phase"] = 1
                else:
                    phase_df.loc[index, col]["vs_top_phase"] = -1

            else:
                age = 0

            phase_df.loc[index, col]["quantity"] = row1[col]
            phase_df.loc[index, col]["vs_others"] = ((row1[col] * 100) / res_quantity[col])
            phase_df.loc[index, col]["age"] = age


    phase_df.reset_index(inplace=True)
    print("product dataset is built in " + str((time.time() - a) / 60) + " minutes")
    return phase_df
