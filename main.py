import time
from flask import Flask, render_template, send_file, request, redirect, url_for
from azure.storage.blob import BlobServiceClient

import pandas as pd
from build_model import model_functions
from create_datasets import build_customer_dataset_for_training
from create_datasets import correlation_analysis
from create_datasets import build_customer_dataset_for_prediction

from campaign_effectiveness import campaign_effectiveness

import joblib
import pickle

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'csv'}
app.config.from_pyfile('config.py')
account = app.config['AZURE_STORAGE_ACCOUNT_NAME']
key = app.config['AZURE_STORAGE_ACCOUNT_KEY']
activities_filename=app.config['ACTIVITIES_FILENAME']
transactions_filename=app.config['TRANSACTIONS_FILENAME']
url = app.config['AZURE_STORAGE_USE_HTTPS']
container = app.config['AZURE_STORAGE_CONTAINER_NAME']
url_return = app.config['URL_RETURN']

if __name__ == '__main__':
    app.run(debug=True)

def upload_blob(file,filename,input_f, year):
    blob_service = BlobServiceClient(account_url=url,credential=key)
    loc_insert_filename = container + "/" + "output"
    if(input_f):        
        loc_insert_filename = container + "/" + "input"    
    insert_filenm = filename + "-" + year + ".csv"
    if(filename[0:5] == "model"):
        insert_filenm = filename + "-" + year + ".sav"
    blob_container = blob_service.get_container_client(loc_insert_filename)
    blob_container.upload_blob(insert_filenm,file, overwrite=True)
    return (None)

#constructs the uri
def get_blob(blob_name, input_f, year):
    dl_blob = url + container + "/output/" + blob_name + "-" + year + ".csv"
    if(input_f):        
        dl_blob = url + container + "/input/" + blob_name + "-" + year + ".csv"
    if(blob_name[0:5] == "model"):
        dl_blob = url + container + "/output/" + blob_name + "-" + year+ ".sav"
    elif(blob_name[0:7] == "farmers"):
        dl_blob = url + container + "/input/sfdc_farmers_region_area.xlsx"
    print(dl_blob)
    return (dl_blob)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route("/", methods=['GET', 'POST'])
def actions():
    year_selected = str(request.form.get("input_year"))
    if request.method == "POST":
        if request.form.get('action') == 'upload_activities':
            file = request.files['activities']
            if file.filename == '':
                return render_template('index.html', state_activities ="Please select a file", state_year=year_selected)
            if file and allowed_file(file.filename):
                upload_blob(file,activities_filename,True,year_selected)
                return render_template('index.html', state_activities ="Uploaded successfully!", state_year=year_selected)
            else:
                return render_template('index.html', state_activities ="You can only upload a .csv file", state_year=year_selected)
        elif request.form.get('action') == "upload_transactions":
            file = request.files['transactions']
            if file.filename == '':
                return render_template('index.html', state_transactions ="Please select a file", state_year=year_selected)
            if file and allowed_file(file.filename):
                upload_blob(file,transactions_filename,True,year_selected)
                return render_template('index.html', state_transactions ="Uploaded successfully!", state_year=year_selected)
            else:
                return render_template('index.html', state_transactions ="You can only upload a .csv file", state_year=year_selected)

        elif request.form.get('action')  == 'train':
            a = time.time()
            trans_uri = get_blob(transactions_filename, True,year_selected)
            transactionalDF = pd.read_csv(trans_uri, low_memory=False, sep=";",  encoding="UTF-8")
            transactionalDF = transactionalDF.drop(columns=["Distribution channel", "Auftragszuordnung"])

            act_uri = get_blob(activities_filename,True,year_selected)
            activityRawDF = pd.read_csv(act_uri, sep=";", encoding="ISO-8859-1")

            customerDF = build_customer_dataset_for_training.buildGlobalCustomerDataset(transactionalDF, activityRawDF, int(year_selected))
            reducedCustomerDF = correlation_analysis.applyCorrelationAnalysis(customerDF, int(year_selected))

            cust_name = "global_customer_dataset_reduced_" + year_selected
            csv_customer = reducedCustomerDF.to_csv(encoding="utf-8", index=False)

            upload_blob(csv_customer, cust_name, False, year_selected)

            time_to_build_dataset = time.time()
            print("dataset for training was built  was in " + str((time_to_build_dataset - a) / 60) + " minutes")

            built_model = model_functions.build_model(reducedCustomerDF, int(year_selected))

            model_name = "model"
            with open('model.sav', 'wb') as model_var:
                pickle.dump(built_model, model_var)
                with open('model.sav', 'rb') as model_var:
                    upload_blob(model_var, model_name,False,year_selected)

            print("upload model")
            print("model was built in " + str((time.time() - time_to_build_dataset) / 60) + " minutes")
            print("training total time: " + str((time.time() - a) / 60) + " minutes")

            return render_template('index.html', state_train ="Training is finished", state_year=year_selected)

        elif request.form.get('action') == 'predict':

            trans_uri = get_blob("transactions", True, year_selected)
            transactionalDF = pd.read_csv(trans_uri, low_memory=False, sep=";", encoding="utf-8")
            transactionalDF = transactionalDF.drop(columns=["Distribution channel", "Auftragszuordnung"])

            act_uri = get_blob("activities", True, year_selected)
            activityRawDF = pd.read_csv(act_uri, sep=";", encoding="ISO-8859-1")

            cust_name = "global_customer_dataset_reduced_" + year_selected
            customer_uri = get_blob(cust_name, False, year_selected)
            customerDF = pd.read_csv(customer_uri, encoding="utf-8")

            predictionDF = build_customer_dataset_for_prediction.buildPredictionDataset(transactionalDF, activityRawDF,
                                                                                        list(customerDF.columns),
                                                                                        int(year_selected))
            model_obj = None
            with open('model.sav', 'rb') as model_var:
                model_obj = joblib.load(model_var)

            resDF = model_functions.predict(predictionDF, model_obj)

            res = resDF.to_csv(encoding='utf-8', index=False)

            upload_blob(res, "prediction", False, str(int(year_selected) + 1))

            return render_template('index.html', state_predict="Prediction is done", state_year=year_selected)

        elif request.form.get('action') == 'get_prediction':
            url_return = url + container + '/output/' + "prediction-" + str(int(year_selected) + 1) + ".csv"
            return redirect(url_return)

        elif request.form.get('action') == 'campaigns':
            trans_uri = get_blob("transactions", True, year_selected)
            act_uri = get_blob("activities", True, year_selected)
            farmers_uri = get_blob("farmers", True, year_selected)
            url_return = campaign_effectiveness.run_campaign(int(year_selected), trans_uri, act_uri, farmers_uri)
            return render_template('index.html', state_year=year_selected)

        elif request.form.get('action') == 'get_analysis':
            return send_file("campaign_analysis_report.html", as_attachment=True)


    return render_template('index.html', state_year="Select year")



