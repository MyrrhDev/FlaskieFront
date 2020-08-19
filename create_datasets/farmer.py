import pandas as pd
from datetime import datetime
from create_datasets.customer_dataset_functions import *
import time


def buildFarmerDataset(transactional_df, currentyear = 2019, predict = False):

    a = time.time()
    #Read transactional dataset
    df = transactional_df.copy()

    # Variable to change study for each year
    df_non_corn = df[(df["Product Group"] != "Corn") & (df["Order Qty (Chg + FOC)"] > 0)]

    df = df[df["Order Qty (Chg + FOC)"]>0]
    df = df[df["Product Group"] == "Corn"]

    df['mnth_yr'] = df['Order Date'].apply(lambda x: datetime.strptime(x,'%d.%m.%Y').month)
    df['Order_Date1'] = df['Order Date'].apply(lambda x: datetime.strptime(x,'%d.%m.%Y'))
    df['timesince']= pd.to_datetime({'year': (df['Season_year']-1), 'month': 9,'day': 1})
    df['days'] = (df['Order_Date1']-df['timesince']).dt.days

    churn = get_custom(df, group_by = "Farmer", quantities = "False", data = "Farmer", predict = predict,year = currentyear)

    churn_aux = churn.copy()
    churn_aux = churn_aux.drop(columns = "t")

    count = obtain_history(df, group_by = "Farmer", quantities = False, data = "Farmer", predict = predict, year = currentyear)
    count.drop(columns = "t", axis = 1, inplace = True)

    quantity = obtain_history(df, group_by = "Farmer", quantities = True, data = "Farmer", predict = predict,year = currentyear)
    quantity.drop(columns="t", axis = 1, inplace = True)

    quantity_non_corn = obtain_history(df_non_corn, group_by = "Farmer", quantities = True, data = "Farmer", predict = predict,year = currentyear)
    quantity_non_corn.drop(columns = "t", axis = 1, inplace = True)

    # Calculate the difference in terms of purchases per cycles
    difference_p = column_difference(count, maxim = False, data = "Farmer")

    # Calculate the difference in terms of quantity per cycles
    difference_q = column_difference(quantity, maxim = False, data = "Farmer")

    # Calculate the ratio between (max_quantity - quantity_per_cycle) and max_quantity, %
    difference_max = column_difference(quantity, maxim = True, data = "Farmer")

    # Get the number of different products
    products = apply_transform(df, column = "Product", data = "Farmer", predict = predict, year= currentyear)
    products.drop(columns = "t", axis = 1, inplace = True)

    # Number of diferents retailers
    retailers = apply_transform(df, column = "Farmer", data = "Farmer", predict = predict, year = currentyear)
    retailers.drop(columns = "t", axis = 1, inplace = True)

    # Set index for aux datasets
    churn = churn.set_index("year", append = True)
    count = count.set_index("year", append = True)
    churn_aux = churn_aux.set_index("year", append = True)
    quantity = quantity.set_index("year", append = True)
    quantity_non_corn = quantity_non_corn.set_index("year", append = True)

    retailers = retailers.set_index("year", append = True)
    products = products.set_index("year", append = True)

    difference_p = difference_p.set_index("year", append = True)
    difference_q = difference_q.set_index("year", append = True)

    glob = pd.DataFrame()
    # Number of inactive and active cycles
    glob["gl_inactive_cycles"] = (churn_aux == 0).astype(int).sum(axis = 1)
    glob["gl_active_cycles"] = (churn_aux == 1).astype(int).sum(axis = 1)

    # Number of purchases
    aux_df = pd.DataFrame(count.sum(axis = 1))
    aux_df.columns = ["gl_purchases"]
    glob = glob.merge(aux_df, how = "left", left_index = True, right_index = True)

    # Avg number of purchases
    aux_df = pd.DataFrame(count.mean(axis = 1))
    aux_df.columns = ["gl_avg_purchases"]
    glob = glob.merge(aux_df, how = 'left', left_index = True, right_index = True)

    # Total quantity
    aux_df = pd.DataFrame(quantity.sum(axis = 1))
    aux_df.columns = ["gl_quantity"]
    glob = glob.merge(aux_df, how = "left", left_index = True, right_index = True)

    # Total quantity of non-corn
    aux_df = pd.DataFrame(quantity_non_corn.sum(axis = 1))
    aux_df.columns = ["gl_quantity_non_corn"]
    aux_df = aux_df.fillna(0)
    glob = glob.merge(aux_df, how = 'left', left_index = True, right_index = True)

    # Avg quantity
    aux_df = pd.DataFrame(quantity.mean(axis = 1))
    aux_df.columns = ["gl_avg_quantity"]
    glob = glob.merge(aux_df, how = 'left', left_index = True, right_index = True)

    # Max quantity
    aux_df = pd.DataFrame(quantity.max(axis = 1))
    aux_df.columns = ["gl_max_quantity"]
    glob = glob.merge(aux_df, how = 'left', left_index = True, right_index = True)

    # Get the most popular Region in terms of number of purchases
    aux_df = most_popular(df, namecolumn = "Region", data = "Farmer")
    aux_df.columns = ["gl_Region"]
    glob = glob.merge(aux_df, how = 'left', left_index = True, right_index = True)

    #  Get the most popular Sales Area in terms of number of purchases
    aux_df = most_popular(df, namecolumn = "Activity Sales Area", data = "Farmer")
    aux_df.columns = ["gl_SalesArea_p"]
    glob = glob.merge(aux_df, how = 'left', left_index = True, right_index = True)


    # Max number of products
    aux_df = pd.DataFrame(products.max(axis = 1))
    aux_df.columns = ["gl_max_products"]
    glob = glob.merge(aux_df, how = 'left', left_index = True, right_index = True)

    # Max number of retailers
    aux_df = pd.DataFrame(retailers.max(axis = 1))
    aux_df.columns = ["gl_max_retailers"]
    glob = glob.merge(aux_df, how = 'left', left_index = True, right_index = True)

    #  Avg number of products
    aux_df = pd.DataFrame(products.mean(axis = 1))
    aux_df.columns = ["gl_avg_products"]
    glob = glob.merge(aux_df, how = 'left', left_index = True, right_index = True)

    #  AVG number of retailers
    aux_df = pd.DataFrame(retailers.mean(axis = 1))
    aux_df.columns = ["gl_avg_retailers"]
    glob = glob.merge(aux_df, how = 'left', left_index = True, right_index = True)

    #The most popular product in terms of quantity
    aux_df = gl_diff(df, "Product", data = "Farmer")
    aux_df.columns = ["gl_popular_product_q", "gl_popular_product_name"]
    glob = glob.merge(aux_df, how = 'left', left_index = True, right_index = True)

    #The most popular retailer in terms of quantity
    aux_df = gl_diff(df, "Soldto", data = "Farmer")
    aux_df.columns = ["gl_popular_retailer_q", "gl_popular_retailer_name"]
    glob = glob.merge(aux_df,how = 'left', left_index = True, right_index = True)

    #Get farmer's category based on quantity(small, medium, big)
    aux_df = getFarmerCategory(quantity)
    glob = glob.merge(aux_df, how = 'left', left_index = True, right_index = True)

    #The most popular month
    aux_df = gl_diff(df, "mnth_yr", data = "Farmer")
    aux_df.columns = ["gl_popular_month_q", "gl_popular_month"]
    glob = glob.merge(aux_df, how = 'left', left_index = True, right_index = True)

    #  The mean of the number of days that passed from  the 1st of September  to the latest purchase date (among all cycles)
    aux_df = gl_days(df, "mean", data = "Farmer")
    aux_df.columns = ["gl_mean_purchase_month"]
    glob = glob.merge(aux_df, how = 'left', left_index = True, right_index = True)

    # The max number of days that passed from  the 1st of September  to the latest purchase date (among all cycles)
    aux_df = gl_days(df, "max", data = "Farmer")
    aux_df.columns = ["gl_max_purchase_month"]
    glob = glob.merge(aux_df, how = 'left', left_index = True, right_index = True)

    # How many times customer has left a company
    aux_df = getHowManyTimesleft(churn)
    glob = glob.merge(aux_df, how = 'left', left_index = True, right_index = True)

    # The maximum consecutive years
    glob = glob.merge(getMaxDuration(churn), how = 'left', left_index = True, right_index = True)

    # The maximum duration (from the last year)
    glob = glob.merge(getMaxPresence(churn), how = 'left', left_index = True, right_index = True)

    # Difference between max and mean
    aux_df = (quantity.max(axis = 1) - quantity.mean(axis = 1)).to_frame()
    aux_df.columns = ["gl_diff_max_mean"]
    glob = glob.merge(aux_df, how = 'left', left_index = True, right_index = True)

    #Get the ratio of two previous years  in terms of quantity
    aux_df = ratio_column_two_prev_years(quantity, data = "Farmer")
    aux_df.columns = ["gl_ratio_two_previous_years_quantity"]
    glob = glob.merge(aux_df, how = 'left', left_index = True, right_index = True)

    ##Get the ratio of two previous years  in terms of number of purchases
    aux_df = ratio_column_two_prev_years(count, data = "Farmer")
    aux_df.columns = ["gl_ratio_two_previous_years_purchase"]
    glob = glob.merge(aux_df, how = 'left', left_index = True, right_index = True)

    churn.columns = "is_active_" + churn.columns
    count.columns = "number_purchases" + count.columns

    # Total quantity of sold corn per cycle
    quantity.columns = "quantity_" + quantity.columns

    # Change of number of purchases per year
    difference_p.columns = "diff_number_purchases_" + difference_p.columns

    # Change of quantity per year
    difference_q.columns = "diff_quantity_" + difference_q.columns

    difference_max.columns = "diffMax_" + difference_max.columns
    products.columns = "number_products_" + products.columns
    retailers.columns = "number_retailers_" + retailers.columns

    merged = pd.merge(churn, count, left_index = True, right_index = True, how = 'left')
    merged = pd.merge(merged, quantity, left_index = True, right_index = True, how = 'left')
    merged = pd.merge(merged, difference_p, left_index = True, right_index = True, how = 'left')
    merged = pd.merge(merged, difference_q, left_index = True, right_index = True, how = 'left')
    merged = pd.merge(merged, difference_max, left_index = True, right_index = True, how = 'left')
    merged = pd.merge(merged, products, left_index = True, right_index = True, how = 'left')
    merged = pd.merge(merged, retailers, left_index = True, right_index = True, how = 'left')
    merged = pd.merge(merged, glob, left_index = True, right_index = True, how = 'left')

    merged.reset_index(inplace=True)

    print("farmer dataset is built in " + str((time.time() - a) / 60) + " minutes")

    return merged

