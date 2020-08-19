import pandas as pd
import numpy as np
import operator




def get_year_list(df):
    year_list = []
    if "Season_year" in df.columns:
        for i in range(len(df["Season_year"])):
            if df["Season_year"][i] not in year_list:
                year_list.append(df["Season_year"][i])

    elif "Date" in df.columns:
        for i in range(len(df["Date"])):
            if int(df["Date"][i][-4:]) not in year_list:
                year_list.append(int(df["Date"][i][-4:]))
    return year_list


def create_cols(df):
    cols = []
    if "t-04" in df.columns:

        cols = df.columns[df.columns != "year"]
        return list(cols)
    elif "Season_year" in df.columns:
        years = []
        for year in list(df["Season_year"].unique()):
            years.append(year)
    elif "season_2013" in df.columns:
        years = []
        for year in df.columns:
            years.append(int(year[-4:]))
    else:
        years = get_year_list(df)

    ordered_years = sorted(years)

    for year in range(len(ordered_years)):
        if "t-0" + str(ordered_years[-1] - ordered_years[year]) == "t-00":
            cols.append("t")
        else:
            cols.append("t-0" + str(ordered_years[-1] - ordered_years[year]))
    return cols



def quantity_dataset(dfaux, group_by, quantities, prod_group="Corn", column=["Season_year"], pref="season"):

    df = dfaux.copy()
    transformed = pd.get_dummies(data=df, columns=column, prefix=(pref))

    if quantities == True:
        for col in transformed.columns[(len(df.columns) - 1):]:
            transformed[col] = transformed[col] * transformed["Order Qty (Chg + FOC)"]

    product_info = transformed[transformed["Product Group"] == prod_group].groupby(group_by)[
        transformed.columns[(len(df.columns) - 1):]].sum()

    return product_info


def quantity_cycle(df, group_by, quantities, column=["Season_year"], pref="Season"):
    """
    For retailers use group_by = "Soldto"
    For farmers use group_by = "Farmer"
    For purchases, set quantities = False
    """
    cols = create_cols(df)
    new = pd.get_dummies(data=df, columns=column, prefix=(pref))

    if quantities == True:
        for col in new.columns[(len(df.columns) - 1):]:
            new[col] = new[col] * new["Order Qty (Chg + FOC)"]

    df1 = new.groupby(group_by)[new.columns[(len(df.columns) - 1):]].sum()
    df1.columns = cols
    return df1

def transform(dfaux, data, last_year=2019, predicting = False):

    df = dfaux.copy()
    cols = create_cols(df)
    df.columns = cols

    df["year"] = last_year
    df_aux = df.copy()

    for i in range(0, len(cols) - 2):
        df_aux = df_aux.loc[:, cols].shift(periods=1, axis=1)
        df_aux['year'] = last_year - 1 - i
        df = df.append(df_aux)

    if data == "Corteva":
        df = df[(df["t"] != 0) | (df["t-01"] != 0)]
        df.loc[:, cols] = df.loc[:, cols].applymap(lambda x: x if x > 0 else 0)
    else:
        if predicting==False:
            if data == "Farmer":
                df = df[(df["t-01"] != 0)]
        else:
            data_for_prediction = df.loc[(df.year == last_year) & (df.t > 0)]

            prediction_features = data_for_prediction.loc[:, cols].shift(periods=-1, axis=1)
            prediction_features.t = None
            prediction_features['year'] = last_year + 1
            prediction_features = prediction_features[(prediction_features["t-01"] != 0)]

            return  prediction_features
    return df

def set_status(df):
    status = df.copy()
    status['stat'] = status.apply(set_status_value_aux, axis=1)
    return status

def set_status_value_aux(row):
    cols_aux = list(row[:-2])
    number_of_positive = np.count_nonzero(cols_aux)

    if (row['t'] == 0 and row['t-01'] > 0):
        return 'churn'
    elif (row['t'] > 0 and row['t-01'] > 0):
        return 'retained'
    elif (number_of_positive > 0 and row['t'] > 0):
        return 'regained'
    else:
        return 'new'


def churn_vs_others(row):
    cols = ["retained", "new", "regained"]
    total = []
    for i in cols:
        if (~ np.isnan(row[i])):
            total.append(row[i])

    return row["churn"] / sum(total)


################# FROM create_data_product ############

def aggregate_by(df, aggregate_by, col="Season_year", pref="season", prod_group="Corn"):
    """
    Aggregates by specified group. 
    aggregate_by = "farmers", "retailers", "POS"
    """
    transformed = pd.get_dummies(data=df, columns=[col], prefix=(pref))

    if aggregate_by == "farmers":

        transformed = transformed.drop(columns=['Soldto', 'Shipto', 'Material', 'Order Date', 'Order Qty (Chg + FOC)',
                                                'Activity Postal Code', 'Activity City', 'Activity Political District',
                                                'Activity State', 'Activity Sales Area', 'Activity Promoter Area',
                                                'Region'])
        transformed = transformed.drop_duplicates()
        product_info = transformed[transformed["Product Group"] == prod_group].groupby(["Product", "Product Group"])[
            transformed.columns[3:]].sum()

        return product_info

    elif aggregate_by == "retailers":

        transformed = transformed.drop(columns=['Shipto', 'Material', 'Order Date', 'Order Qty (Chg + FOC)', 'Farmer',
                                                'Activity Postal Code', 'Activity City', 'Activity Political District',
                                                'Activity State', 'Activity Sales Area', 'Activity Promoter Area',
                                                'Region'])
        transformed = transformed.drop_duplicates()
        product_info = transformed[transformed["Product Group"] == prod_group].groupby(["Product", "Product Group"])[
            transformed.columns[3:]].sum()

        return product_info

    elif aggregate_by == "POS":

        transformed = transformed.drop(columns=['Soldto', 'Material', 'Order Date', 'Order Qty (Chg + FOC)', 'Farmer',
                                                'Activity Postal Code', 'Activity City', 'Activity Political District',
                                                'Activity State', 'Activity Sales Area', 'Activity Promoter Area',
                                                'Region'])

        transformed = transformed.drop_duplicates()
        product_info = transformed[transformed["Product Group"] == prod_group].groupby(['Product', 'Product Group'])[
            transformed.columns[3:]].sum()

        return product_info


def get_top(x, cols):
    d = {}

    for i in cols:

        if (x[i].value_counts().index[0] == '' and (x[i].value_counts().index.size > 1)):
            d[i] = x[i].value_counts().index[1]

        else:
            d[i] = x[i].value_counts().index[0]

    return pd.Series(d, index=cols)


def top_region_per_product(df, filter_by, period, product="Corn"):
    """
    period = "global" | period = "season"
    product = "Corn" (as default)
    """
    if period == "global":

        transformed = pd.get_dummies(data=df, columns=[filter_by])
        top = transformed[transformed['Product Group'] == product].groupby(['Product', 'Product Group'])[
            transformed.columns[(len(df.columns) - 1):]].sum().idxmax(axis=1).to_frame()

        return top

    elif period == "season":

        transformed = pd.get_dummies(data=df, columns=["Season_year"], prefix=('season'))

        for col in transformed.columns[(len(df.columns) - 1):]:
            transformed[col] = transformed[col] * transformed[filter_by]

        cols = transformed.columns[(len(df.columns) - 1):]
        product_info = transformed[transformed['Product Group'] == product].groupby(['Product', 'Product Group']).apply(
            get_top, cols)

        return product_info


def get_phase(current, ind, row1, n):
    if len(row1) > n:
        arr = row1[ind - n:ind]
    else:
        arr = row1[0:ind]

    arr = list(filter((0).__ne__, arr))

    if ((len(arr) == 0 and (current > 0)) or len(arr) > 0 and (current >= np.average(arr)) and current > 0):
        return 1
    else:
        return -1



def get_start(row, cols):
    start_index = np.min(np.nonzero(row.to_numpy()))
    return int(cols[start_index].replace('season_', ''))

def get_unique(df, column, data):
    cols = create_cols(df)
    new = pd.get_dummies(data=df, columns=["Season_year"], prefix=('season'))
    for col in new.columns[(len(df.columns) - 1):]:
        new[col] = new[col] * new[column]

    corn = new[new['Product Group'] == 'Corn'].groupby(data)[new.columns[(len(df.columns) - 1):]].nunique()
    corn.columns = cols
    return corn


##################################################################################################################
############################################## RETAILER ##########################################################
##################################################################################################################

def get_custom(df, group_by, quantities, data, predict = False,year=2019):
    cols = create_cols(df)
    to_transform = quantity_cycle(df, group_by=group_by, quantities=quantities)
    df1 = transform(to_transform, data=data, last_year=year, predicting = predict)

    if predict==True:
        cols.remove("t")

    df1.loc[:, cols] = df1.loc[:, cols].applymap(lambda x: 1 if x > 0 else 0)

    del (to_transform)
    return df1


#### redefinir per a totes les "get_the_most_popular" (most_popular)
# Obtain the most popular geo per cycle in terms of number of purchases

def most_popular(df, namecolumn, data):
    """
    data can either be "Soldto" or "Farmer"
    """
    df = df.groupby([data, 'Season_year'])[namecolumn].max()
    df = df.reset_index()
    df.head()

    df.Season_year = df.Season_year + 1
    df.rename(columns={'Season_year': 'year'}, inplace=True)
    df = df.set_index([data, 'year'])
    return df


# Get the latest and the mean date of purchase per cycle starting from the 1st of September
def gl_days(df, maxmean, data):
    """
    data can be either "Soldto" or "Farmer"
    """
    if maxmean == "max":
        df3 = df.groupby([data, "Season_year"])['days'].max().to_frame()
    else:
        df3 = df.groupby([data, "Season_year"])['days'].mean().to_frame()

    df3 = df3.reset_index()
    df3.Season_year = df3.Season_year + 1
    df3.rename(columns={'Season_year': 'year'}, inplace=True)
    df3 = df3.set_index([data, 'year'])

    return df3


def gl_diff(df, namecolumn, data):
    """
    data can be either "Soldto" or "Farmer"
    """
    df3 = df.groupby([data, "Season_year", namecolumn]).agg({"Order Qty (Chg + FOC)": sum})
    df3 = df3.reset_index()
    '''
    df3 = df3.groupby([data, "Season_year"])[namecolumn, "Order Qty (Chg + FOC)"].agg({"Order Qty (Chg + FOC)": max,
                                                                                       namecolumn: max})
    '''
    df3 = df3.groupby([data, "Season_year"])[[namecolumn, "Order Qty (Chg + FOC)"]].agg({"Order Qty (Chg + FOC)": max,namecolumn: max})

    df3 = df3.reset_index()
    df3.Season_year = df3.Season_year + 1
    df3.rename(columns={"Season_year": "year"}, inplace=True)
    df3 = df3.set_index([data, "year"])
    return df3



def apply_transform(df, data, column, predict = False, year=2019):
    cols = create_cols(df)
    unic = get_unique(df, column=column, data=data)
    df = transform(unic, data=data, last_year=year, predicting = predict)
    return df


#### Ajuntar les dues d'aquí sota en una de sola #####
def obtain_history(df, group_by, quantities, data,  predict = False, year = 2019):
    cycle = quantity_cycle(df, group_by=group_by, quantities=quantities)
    df = transform(cycle, last_year=year, data=data, predicting = predict)
    del (cycle)

    return df


# Get the column difference

def column_difference(df, data, maxim=False):
    """
    Takes two arguments, a df and then max. If max = True, it will return the maximum
    differnece among columns, otherwise it'll just return the difference.
    """
    c = df.copy()
    if maxim == False:
        cols = df.columns[0]
        for col in c.columns[1:len(c.columns) - 1]:
            c[cols] = df[cols] - df[col]
            cols = col
        return c

    elif maxim == True:
        c = c.reset_index()
        c.set_index([data, "year"], inplace=True)

        maxi = c[c.columns[0:(len(c.columns))]].max(axis=1).to_frame()

        for col in c.columns[0:len(c.columns)]:
            c[col] = (c[col] / maxi[0])
        return c


##### the following two call each other #######

def column_ratio(x):
    if ((x["t-02"] != 0) and (~np.isnan(x["t-02"]))):
        return x["t-01"] / x["t-02"]
    elif (x["t-01"] != 0):
        return 1
    else:
        return 0


# Get the ratio of two previous years
def ratio_column_two_prev_years(df, data):
    """
    data can either be "Farmer" or "Soldto"
    """
    df = df.reset_index()
    df["ratio"] = df.apply(lambda x: column_ratio(x), axis=1)
    df = df[["ratio", data, "year"]]
    df = df.set_index([data, "year"])

    return df


#### merge the following two into get_max

# Get the max presence in a company in terms of cycles  (in total)
def getMaxDuration(df):
    c = df.copy()
    c["conseq"] = 0
    c["gl_maxduration"] = 0
    for i in range(1, (len(df.columns))):
        c.loc[c[c.columns[i]] == 1, "conseq"] = 1 + c.loc[c[c.columns[i]] == 1, "conseq"]
        c.loc[c[c.columns[i]] == 0, "conseq"] = 0

        c.gl_maxduration = (c[["conseq", "gl_maxduration"]]).max(axis=1)

    c = c[["gl_maxduration"]]
    return c


# Get the max presence in a company in terms of cycles  starting from the last year
def getMaxPresence(df):
    c = df.copy()
    c["conseq"] = 0
    c["gl_max_presence"] = 0
    for i in range(1, (len(df.columns) - 2)):
        c.loc[c[c.columns[i]] == 1, "conseq"] = 1 + c.loc[c[c.columns[i]] == 1, "conseq"]
        c.loc[c[c.columns[i]] == 0, "conseq"] = 0

        c.gl_max_presence = (c[["conseq"]]).max(axis=1)

    c = c[["gl_max_presence"]]
    return c


# Obtain the list of number of purchases
def count_list(df, column_name):
    aux = df.groupby(['Farmer', column_name, 'Season_year'], sort=False)["Order Qty (Chg + FOC)"].count().to_frame()
    aux = aux.reset_index(column_name)

    aux1 = aux.groupby(['Farmer', 'Season_year'])["Order Qty (Chg + FOC)"].apply(list).to_frame()
    aux2 = aux.groupby(['Farmer', 'Season_year'])[column_name].apply(list).to_frame()

    prod_quant = pd.merge(aux1, aux2, left_index=True, right_index=True, how='left')

    corn1 = prod_quant.reset_index()
    corn1.Season_year = corn1.Season_year + 1
    corn1.rename(columns={'Season_year': 'year'}, inplace=True)
    corn1 = corn1.set_index(['Farmer', 'year'])
    # aux=aux.set_index(['Farmer', 'Season_year'])
    return corn1


# Get the list of quantity
def quantity_list(df, column_name):
    aux = df.groupby(['Farmer', column_name, 'Season_year'], sort=False)["Order Qty (Chg + FOC)"].sum().to_frame()
    aux = aux.reset_index(column_name)

    aux1 = aux.groupby(['Farmer', 'Season_year'])["Order Qty (Chg + FOC)"].apply(list).to_frame()
    aux2 = aux.groupby(['Farmer', 'Season_year'])[column_name].apply(list).to_frame()

    prod_quant = pd.merge(aux1, aux2, left_index=True, right_index=True, how='left')

    corn1 = prod_quant.reset_index()
    corn1.Season_year = corn1.Season_year + 1
    corn1.rename(columns={'Season_year': 'year'}, inplace=True)
    corn1 = corn1.set_index(['Farmer', 'year'])

    return corn1


# Get the farmer category according to corn quantity
def getFarmerCategory(quantity):
    z = quantity["t-01"].to_frame()
    q1 = z.quantile(0.25)
    q2 = z.quantile(0.75)

    z.loc[(z["t-01"] < q1["t-01"]), "gl_category"] = 0
    z.loc[((z["t-01"] < q2["t-01"]) & (z["t-01"] > q1["t-01"])), "gl_category"] = 1
    z.loc[(z["t-01"] > q2["t-01"]), "gl_category"] = 3
    z.drop(columns=['t-01'], axis=1, inplace=True)

    return z


# Get the number of times a customer has left a company
def getHowManyTimesleft(df):
    c = df.copy()
    c["gl_number_left"] = 0
    for i in range(1, (len(df.columns) - 2)):
        c.loc[operator.and_(c[c.columns[i]] == 1, c[c.columns[i - 1]] == 0), "gl_number_left"] = 1 + c.loc[
            operator.and_(c[c.columns[i]] == 1, c[c.columns[i - 1]] == 0), "gl_number_left"]

    c = pd.DataFrame(c["gl_number_left"])
    return c


def compute_after_before(row):
    if (pd.isnull(row["Activity_Date"]) == False):
        if row["Activity_Date"] >= row["Order_Date"]:
            return 1
        elif row["Activity_Date"] < row["Order_Date"]:
            return 0


def compute_season(row):
    if int(row["Activity_Date"].split(".")[1]) >= 9:
        return int(row["Activity_Date"].split(".")[2]) + 1
    elif int(row["Activity_Date"].split(".")[1]) < 9:
        return int(row["Activity_Date"].split(".")[2])


def create_list(row, column):
    return str(row[column]).split(";")


def treat_nan(df, column):
    col = []
    for i in range(len(df[column])):
        if df[column][i] == 0:
            col.append("none")
        else:
            col.append(df[column][i])

    df[column] = col
    return df


def add_year(df, date):
    """
    date should be the column in the data frame that indicates the date for which the activity/transaction
    happened in a dd/mm/yyyy format
    """
    date_list = df[date]
    year_list = []
    years = []

    for i in range(len(date_list)):
        year = date_list[i][-4:]
        year_list.append(year)

    df["year"] = year_list

    return df


def delete_blanks(df):
    """
    Takes blank spaces in variable names and swaps them for underscores.
    """
    df_cols = list(df.columns)
    for column in range(len(df_cols)):
        df_cols[column] = df_cols[column].replace(" ", "_")

    df.columns = df_cols
    return df


def type_id(df, ID):
    """
    Returns if the unique ID corresponds to a Farmer or to a Retailer
    """
    type_ID = []
    for i in range(len(df[ID])):
        if df[ID][i][:2] == "NE":
            type_ID.append("Farmer")
        else:
            type_ID.append("Retailer")
    df["type_ID"] = type_ID

    return df


def create_season(df, column):
    """
    Computes yearly season depending on month in which transaction/activity has occurred
    """
    Season_year = []
    for i in range(len(df[column])):
        if int(df[column][i][3:5]) < 9:
            Season_year.append(int(df[column][i][-4:]))
        else:
            Season_year.append(int(df[column][i][-4:]) + 1)

    df["Season_year"] = Season_year
    return df



def clean_names_onehot(row):
    row = row.replace(",", "; ").replace("lll", "ll").replace("null", "").replace(" ", "_")
    return row.split(";_")