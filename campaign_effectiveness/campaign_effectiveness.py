import pandas as pd
import matplotlib.pyplot as plt

from create_datasets import customer_dataset_functions

from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import time
import datetime
import plotly.offline as py#visualization
import plotly.express as px


#py.init_notebook_mode(connected=True)#visualization
import plotly.graph_objs as go#visualization


#Global variables
b = time.time()
#current_year = 2019

state_dict=dict({'Baden Wuerttemberg': 'Baden-Wuerttemberg', 'Baden-Württemberg': 'Baden-Wuerttemberg',
                'Bayern': 'Bayern',
                'Brandenburg': 'Brandenburg',
                'Bremen': 'Bremen',
                'Hamburg': 'Hamburg',
                'Hessen': 'Hessen',
                'Mecklemburg - Vorpommern': 'Mecklemburg-Vorpommern', 'Mecklenburg-Vorpommern': 'Mecklemburg-Vorpommern',
                'Niedersachsen': 'Niedersachsen',
                'Nordrhein Westfalen': 'Nordrhein-Westfalen', 'Nordrhein-Westfalen': 'Nordrhein-Westfalen',
                'Rheinland Pfalz': 'Rheinland-Pfalz', 'Rheinland-Pfalz': 'Rheinland-Pfalz',
                'Saarland': 'Saarland',
                'Sachsen': 'Sachsen',
                'Sachsen - Anhalt': 'Sachsen-Anhalt', 'Sachsen-Anhalt': 'Sachsen-Anhalt',
                'Schleswig Holstein': 'Schleswig-Holstein', 'Schleswig-Holstein': 'Schleswig-Holstein',
                'Thueringen': 'Thueringen', 'Thüringen': 'Thueringen'})


title_pct_dict={
'pct_customers':'total territory that is planted with Cortevas corn', 
'pct_new': 'percentage of the clients that are new', 
'pct_retained': 'percentage of the clients that are retained', 
'pct_regained': 'percentage of the clients that are regained',
'pct_churn': 'percentage of the previous year clients that churn' 
} 

increase_dict={
'Order_Qty_(Chg_+_FOC)_increase': 'Quantity dynamics',
'client_increase': 'Number of clients dynamics',
'churn_increase': 'Number of churn dynamics',
'new_increase': 'Number of new clients dynamics',
'not_client_increase': 'Number of not clients dynamics',
'regained_increase': 'Regained clients dynamics',
'retained_increase': 'Retianed clients dynamics'
}


overall_list=['Region',
'Season_year',
 'client',
 'churn',
 'new',
 'not_client',
 'regained',
 'retained']

client_list=['new','not_client','regained','retained']

increase_list=[]
pct_list=[]
feature_list = ['Farm_Visit', 'Guided_Planting','Free_Seed_Allocation','Corn_Field_Day','Corn_silage_sampling']

html_path = 'campaign_analysis_report.html'

list_of_graphs = []

def get_bar_df (df, current_year):
    df=df[df['Season_year']==current_year]
    df['Activity_Date']=df['Activity_Date'].apply(lambda x:datetime.datetime(int(x.split('.')[2]),int(x.split('.')[1]),int(x.split('.')[0])))
    df['Month']=df['Activity_Date'].apply(lambda x:int(x.month))
    df=df.groupby(['Month'], as_index=False).sum()
    df.reset_index()
    return (df)

def processTransactions(state_dict, transactional_path):
    #transaction = pd.read_csv(transactional_path, sep = ",", low_memory = False)
    transaction = pd.read_csv(transactional_path, low_memory=False, sep=";")
    transaction = transaction[transaction['Order Qty (Chg + FOC)'] >= 0]
    transaction = transaction[transaction['Product Group'] == 'Corn']
    transaction = transaction.rename(columns = {'Farmer': 'ID'})
    transaction['Activity State'].replace(state_dict, inplace = True)
    transaction = transaction.drop(columns = ['Distribution channel','Activity Postal Code'])
    return transaction

def applyGroupByTransactions(transaction):
    transactions = transaction.copy()
    transactions = customer_dataset_functions.delete_blanks(transactions)
    transactions = transactions[transactions['Season_year'] >= 2014]
    transaction2 = transactions.groupby(['ID', 'Season_year', 'Region', 'Activity_Sales_Area', 'Activity_State'], as_index = False).sum()
    
    return transaction2

def createStatusDF(transaction, transactionDF, current_year):
    
    farmer_to_state = dict(zip(transaction['ID'], transaction['Activity State']))
    farmer_to_area = dict(zip(transaction['ID'], transaction['Activity Sales Area']))
    farmer_to_region = dict(zip(transaction['ID'], transaction['Region']))
    
    quantity_df = customer_dataset_functions.quantity_dataset(transactionDF, group_by = "ID", quantities = False)
    df_transformed_all = customer_dataset_functions.transform(dfaux = quantity_df,
                                   data = "Corteva",
                                   last_year = current_year)   
    status = df_transformed_all.copy()
    #cols_aux = create_cols(status)[:-2]
    status['stat'] = status.apply(customer_dataset_functions.set_status_value_aux, axis=1)
    status = status.reset_index()
    status = status.rename(columns={'year': 'Season_year'})
    status['Activity Sales Area'] = status['ID'].map(farmer_to_area)
    status['Activity State'] = status['ID'].map(farmer_to_state)
    status['Region'] = status['ID'].map(farmer_to_region)
    status = customer_dataset_functions.delete_blanks(status)
    
    return status

def buildFinalDF(activity_transaction):
    final_df = activity_transaction.copy()
    cols_to_drop = []
    cols_to_keep = []
    for i in range(len(final_df.columns)):
        if final_df.columns[i][-2:] == '_x':
            cols_to_drop.append(final_df.columns[i])
        elif final_df.columns[i][-2:] == '_y':
            cols_to_drop.append(final_df.columns[i])
        else:
            cols_to_keep.append(final_df.columns[i])
            
    l=['year', 't-06', 't-05', 't-04', 't-03', 't-02', 't-01', 't']
    for i in l:
        cols_to_drop.append(i)
    final_df = final_df.drop(columns = cols_to_drop)
    
    final_df['stat'] = final_df['stat'].fillna('not_client')
    
    final_df = final_df.fillna(0)
    
    final_df['client'] = final_df['Order_Qty_(Chg_+_FOC)'].apply(lambda x: 1 if x > 0 else 0)
    
    dummies = pd.get_dummies(final_df['stat'])
    
    final_id = pd.concat([final_df, dummies], axis = 1)
    
    return final_id


def createFinalActivity(current_year, activity_path):
    a = time.time()
    df_activity = pd.read_csv(activity_path, encoding="ISO-8859-1", sep=';')
    df_activity['Activity State'].replace(state_dict, inplace=True)

    activity = customer_dataset_functions.delete_blanks(df_activity)
    activity = customer_dataset_functions.add_year(activity, "Date")
    activity = activity.rename(columns={"SAP_ID": "ID",
                                        "Date": "Activity_Date"})

    activity = customer_dataset_functions.type_id(activity, "ID")
    activity = customer_dataset_functions.create_season(activity, "Activity_Date")
    dtype = dict(year=int)
    activity["year"] = activity["year"].astype(dtype)

    activity = activity.fillna(0)
    activity = activity[activity["type_ID"] == "Farmer"]
    activity = activity.reset_index()

    cols_nan = ['Visit', 'Phone', 'Services', 'Mail', 'Events', 'Sales_Activities', 'Others']
    for i in cols_nan:
        activity = customer_dataset_functions.treat_nan(activity, i)

    for i in cols_nan:
        activity[i] = activity[i].apply(lambda x: customer_dataset_functions.clean_names_onehot(x))

    one_hot_encoder_list = [activity.columns[3], activity.columns[4], activity.columns[5],
                            activity.columns[6], activity.columns[7], activity.columns[8]]

    activity_test = activity.copy()

    for column in one_hot_encoder_list:
        mlb = MultiLabelBinarizer()
        activity_test1 = pd.DataFrame(mlb.fit_transform(activity_test[column]), columns=mlb.classes_)
        activity_test = activity_test.merge(activity_test1, left_index=True, right_index=True)

    final_cols = []
    for i in activity_test.columns:
        if (("?_Keine_?" not in i) and ("none_" not in i) and ("--none--" not in i)):
            final_cols.append(i)

    final_activity = activity_test[final_cols]

    cols_to_drop = ["index"]
    final_activity = final_activity.drop(columns=cols_to_drop)

    final_activity = final_activity[final_activity['Season_year'] >= 2014]
    final_activity = final_activity[final_activity['Season_year'] <= current_year]
    final_activity = final_activity.reset_index()
    bar_activity = get_bar_df(final_activity, current_year)
    final_activity = final_activity.groupby(['Region', 'Activity_Sales_Area', 'Activity_State', 'ID', 'Season_year'],
                                            as_index=False).sum()

    print("activity dataset is built in " + str((time.time() - a) / 60) + " minutes")
    return final_activity, bar_activity


def historical_chart (df, variable_x, variable_y, variable_color, title):
    fig = px.line(df, x=variable_x, y=variable_y, color=variable_color,  title=title)
    #fig.show()
    list_of_graphs.append(fig)

def bar_chart (df, variable_x, variable_y,  title):
    fig = px.bar(df, x=variable_x, y=variable_y,  title=title)
    #fig.show()
    list_of_graphs.append(fig)

def effect_chart(var_list, df, bar_df, df_market_region, region_list):
    df_aux = df.copy()
    type_client_list = ['new', 'not_client', 'regained', 'retained']
    for var in var_list:
        df_aux[var + '_eff'] = df_aux[var].apply(lambda x: 1 if x > 0 else 0)
        df_aux[var + '_eff'] = df_aux.apply(lambda x: x['client'] * x[var + '_eff'], axis=1)
        for i in type_client_list:
            df_aux[i + '_' + var + '_eff'] = df_aux[var].apply(lambda x: 1 if x > 0 else 0)
            df_aux[i + '_' + var + '_eff'] = df_aux.apply(lambda x: x[i] * x[i + '_' + var + '_eff'], axis=1)

        df_aux[var + '_eff'] = df_aux['new_' + var + '_eff'] + df_aux['not_client_' + var + '_eff'] + df_aux[
            'regained_' + var + '_eff'] + df_aux['retained_' + var + '_eff']

    df_aux_region = df_aux.groupby(['Region', 'Season_year'], as_index=False).sum()
    final_region = df_aux_region.merge(df_market_region, on=['Region'], how='left')
    # df_aux_area = df_aux.groupby(['Activity_Sales_Area', 'Season_year'], as_index = False).sum()


    efective_df = df_aux[
        ['Season_year', 'stat', var, 'new', 'not_client', 'regained', 'retained', 'new_' + var + '_eff',
         'not_client_' + var + '_eff', 'regained_' + var + '_eff', 'retained_' + var + '_eff']]
    efective_df = efective_df.groupby(['Season_year', 'stat'], as_index=False).sum()
    efective_df = efective_df[
        ['Season_year', 'new_' + var + '_eff', 'not_client_' + var + '_eff', 'regained_' + var + '_eff',
         'retained_' + var + '_eff']]
    efective_df = efective_df.groupby(['Season_year']).sum()
    efective_df.reset_index()
    efective_df = efective_df.unstack()
    efective_df = efective_df.reset_index()

    df_aux = df_aux.groupby(['Season_year', 'Activity_Sales_Area', 'Region'], as_index=False).sum()

    for var in var_list:
        bar_chart(bar_df, 'Month', var, 'Last year ' + var + ' distribution')
        historical_chart(final_region, 'Season_year', var + '_eff', 'Region',
                         'Number of clients that has recieved ' + str(var) + ' by region')
        historical_chart(efective_df, 'Season_year', 0, 'level_0',
                     'Number of farmers that has recieved ' + str(var) + ' by type of clients')


        for j in region_list:
            historical_chart(df_aux[df_aux['Region'] == j], 'Season_year', var + '_eff', 'Activity_Sales_Area',
                             'Number of clients that has recieved ' + str(var) + ' in ' + str(j))
        
def compute_features(df):
    df_aux=df.copy()
    df_aux['pct_customers'] = df_aux['client'] / df_aux['total_farms']
    df_aux['pct_new'] = df_aux['new'] / df_aux['client']
    df_aux['pct_retained'] = df_aux['retained'] / df_aux['client']
    df_aux['pct_regained'] = df_aux['regained'] / df_aux['client']

    for i in ['Order_Qty_(Chg_+_FOC)', 'client', 'churn', 'new', 'not_client', 'regained', 'retained']:
        df_aux['{}_increase'.format(i)] = df_aux[i].pct_change()
    df_aux['last_year_client'] = df_aux['client'].shift(periods=1)
    df_aux['pct_churn'] = df_aux['churn'] / df_aux['last_year_client']
    df_aux = df_aux[df_aux['Season_year'] >= 2015]
    df_aux.reset_index()
    #df_aux.drop(columns='index')
    return (df_aux)


def run_campaign(current_year,trans_uri,act_uri,farmers_uri):
    transaction = processTransactions(state_dict, trans_uri).reset_index().drop(columns = 'index')
    #transaction = processTransactions(state_dict, transactional_path).reset_index().drop(columns = 'index')
    transaction2 = applyGroupByTransactions(transaction)


    area_to_region = dict(zip(transaction2['Activity_Sales_Area'], transaction2['Region']))
    df_market = pd.read_excel(farmers_uri, header= None) ###
    df_market.columns = ['area', 'total_farms']
    df_market['Activity_Sales_Area'] = df_market['area'].apply(lambda x: x.strip())
    df_market['Region'] = df_market['Activity_Sales_Area'].map(area_to_region)
    df_market = df_market[df_market['Region'].isnull() == False]
    df_market = df_market.drop(columns = 'area')
    #market_size = market_size.reset_index().drop(columns = 'index')
    df_market = df_market.groupby(['Region', 'Activity_Sales_Area'], as_index = False).sum()

    status = createStatusDF(transaction,transaction, current_year)

    transaction_status = status.merge(transaction2, on = ['ID', 'Season_year', 'Region', 'Activity_Sales_Area', 'Activity_State'], how = 'left')
    transaction_status = transaction_status.fillna(0)

    final_activity, bar_df = createFinalActivity(current_year, act_uri)

    activity_transaction = final_activity.merge(transaction_status, on = ['Region', 'Activity_Sales_Area', 'Activity_State', 'ID','Season_year'], how = 'outer')

    finalDF = buildFinalDF(activity_transaction)
    finalDF = finalDF.drop(columns = ['index', 'Activity_Postal_Code'])

    final_quantity=finalDF.groupby(['Season_year', 'stat'], as_index=False).sum()
    #final_quantity=final_quantity[['Season_year', 'stat', 'Order_Qty_(Chg_+_FOC)']]

    finalDF_region = finalDF.groupby(['Region', 'Season_year'], as_index = False).sum()
    finalDF_area = finalDF.groupby(['Activity_Sales_Area', 'Season_year'], as_index = False).sum()

    df_market_region = df_market.groupby(['Region'], as_index = False).sum()
    df_market_area = df_market.groupby(['Activity_Sales_Area'], as_index = False).sum()

    final_region = finalDF_region.merge(df_market_region, on = ['Region'], how = 'left')
    final_area = finalDF_area.merge(df_market_area, on = ['Activity_Sales_Area'], how = 'left')

    final_area = compute_features(final_area)
    final_region = compute_features(final_region)
    a = time.time()

    for i in final_region.columns:
        if '_increase' in i:
            increase_list.append(i)
        if 'pct_' in i:
            pct_list.append(i)

    overall_df=final_region[overall_list]
    overall_df=overall_df.groupby(['Season_year']).sum()
    overall_df.reset_index()
    overall_df=overall_df.unstack()
    overall_df=overall_df.to_frame()
    overall_df=overall_df.reset_index()

    historical_chart(overall_df, 'Season_year', 0, 'level_0', 'Number of clients: Total number of clients grouped by type of clients')
    historical_chart(final_quantity, 'Season_year','Order_Qty_(Chg_+_FOC)' , 'stat', 'Quantity bought per client:Amount of quantity bougth by each type of client')


    overall_list.remove('Season_year')
    overall_list.remove('Region')

    for i in overall_list:
        historical_chart(final_region, 'Season_year', i, 'Region', 'Total number of '+str(i) +' per region')
        
    for i in increase_list:
        historical_chart(final_region, 'Season_year', i, 'Region', increase_dict[i] +" (comparing to the previous year)")
        
    for i in pct_list:
        historical_chart(final_region, 'Season_year', i, 'Region', str(i) +': '+title_pct_dict[i])
        
    region_list=finalDF['Region'].unique()
    for j in region_list:
        df=finalDF[finalDF['Region']==j]
        df=df.groupby(['Season_year','Activity_Sales_Area'], as_index=False ).sum()
        for i in overall_list:
            historical_chart(df, 'Season_year', i, 'Activity_Sales_Area', 'Number of '+str(i) +' in '+str(j))
            

    effect_chart(feature_list, finalDF, bar_df,df_market_region, region_list)
    print("all charts are built in " + str((time.time() - a) / 60) + " minutes")
    a = time.time()
    with open(html_path, 'w') as f:
        for graph in list_of_graphs:
            f.write(graph.to_html(full_html=False, include_plotlyjs='cdn'))

    print("html file built in " + str((time.time() - a) / 60) + " minutes")
    print("the whole process is built in " + str((time.time() - b) / 60) + " minutes")
    
    return (html_path)
