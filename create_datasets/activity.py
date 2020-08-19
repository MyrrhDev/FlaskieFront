import pandas as pd
from create_datasets.customer_dataset_functions import *
from inspect import currentframe, getframeinfo
from sklearn.preprocessing import MultiLabelBinarizer
import time




def buildActivityDataset(activityRawDF, currentyear=2019):

    a = time.time()
    # Prepare data for transformation
    activity = activityRawDF.copy()
    activity.columns = activity.columns.map(lambda x: x.replace(" ", "_"))

    activity = add_year(activity, "Date")

    activity = activity.rename(columns={"SAP_ID": "ID",
                                        "Date": "Activity_Date"})
    activity = type_id(activity, "ID")
    activity = create_season(activity, "Activity_Date")
    dtype = dict(year=int)
    activity["year"] = activity["year"].astype(dtype)

    activity = activity.fillna(0)
    activity = activity[activity["type_ID"] == "Farmer"]
    activity = activity.reset_index()

    cols_nan = ['Visit', 'Phone', 'Services', 'Mail', 'Events', 'Sales_Activities', 'Others']
    activity[cols_nan] = activity[cols_nan].replace(to_replace=0, value="none")
    activity[cols_nan] = activity[cols_nan].applymap(clean_names_onehot)

    one_hot_encoder_list = [activity.columns[3], activity.columns[4], activity.columns[5],
                            activity.columns[6], activity.columns[7], activity.columns[8]]

    activity_test = activity.copy()

    for column in one_hot_encoder_list:
        # activity_test[column] = test.apply(lambda row: create_list(row, column), axis = 1)
        mlb = MultiLabelBinarizer()
        activity_test1 = pd.DataFrame(mlb.fit_transform(activity_test[column]), columns=mlb.classes_).add_prefix(column + '_')
        activity_test = activity_test.merge(activity_test1, left_index=True, right_index=True)



    final_cols = []
    for i in activity_test.columns:
        if (("?_Keine_?" not in i) and ("none_" not in i) and ("--none--" not in i)):
             final_cols.append(i)

    final_activity = activity_test[final_cols]

    cols_to_drop = ["index", "year", "Activity_Postal_Code", "Activity_City", "Activity_Political_District",
                    "Activity_State", "Activity_Sales_Area", "Activity_Promoter_Area", "Region"]
    final_activity = final_activity.drop(columns=cols_to_drop)
    final_activity = final_activity.groupby(["ID", "Season_year"], as_index=False).sum()

    final_activity.iloc[:,2:]= final_activity.iloc[:,2:].applymap(lambda x: 1 if (x > 0) else x)

    #final_activity.to_csv("../data/output/exploration_output/activity_" + str(currentyear) + ".csv")

    final_activity.reset_index(drop=True, inplace=True)

    print("activity dataset is built in " + str((time.time() - a) / 60) + " minutes")

    return final_activity

