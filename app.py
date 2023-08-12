from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np
import os
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from exception import CustomException
from utils import load_object
from sklearn.decomposition import PCA


application = Flask(__name__)

app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for predicting data
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')    

    else:
        data=CustomData(
            Customer_Age=int(request.form.get('Customer_Age')),
            Dependent_count=int(request.form.get('children')),
            Months_on_book=int(request.form.get('monthOnBooks')),
            Total_Relationship_Count=int(request.form.get('totalRelationshipCount')),
            Months_Inactive_12_mon=int(request.form.get('monthsInactive12mon')),
            Contacts_Count_12_mon=int(request.form.get('contactsCount12mon')),
            Credit_Limit=float(request.form.get('creditLimit')),
            Total_Revolving_Bal=float(request.form.get('totalRevolvingBal')),
            Avg_Open_To_Buy=float(request.form.get('avgOpenToBuy')),
            Total_Amt_Chng_Q4_Q1=float(request.form.get('totalAmtChngQ4Q1')),
            Total_Trans_Amt=float(request.form.get('totalTransAmt')),
            Total_Trans_Ct=float(request.form.get('totalTransCt')),
            Total_Ct_Chng_Q4_Q1=float(request.form.get('totalCtChngQ4Q1')),
            Avg_Utilization_Ratio=float(request.form.get('avgUtilizationRatio')),
            Education_Level=request.form.get('Education_Level'),
            Marital_Status=request.form.get('Marital_Status'),
            Income_Category=request.form.get('Income_Category'),
            Gender=request.form.get('Gender'),
            Attrition_Flag=request.form.get('Attrition_Flag')
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("Results:", results)
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0")
