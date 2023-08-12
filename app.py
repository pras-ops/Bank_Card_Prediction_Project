from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np
import os
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from exception import CustomException
from utils import load_object
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder


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
            name = request.form.get('name'),
            age = float(request.form.get('age')),
            gender = int(request.form.get('gender')),
            children = float(request.form.get('children')),
            edu = int(request.form.get('edu')),
            marital = int(request.form.get('marital')),
            income = int(request.form.get('income')),
            monthOnBooks = int(request.form.get('monthOnBooks')),
            totalRelationshipCount = int(request.form.get('totalRelationshipCount')),
            monthsInactive12mon = int(request.form.get('monthsInactive12mon')),
            contactsCount12mon = int(request.form.get('contactsCount12mon')),
            creditLimit = float(request.form.get('creditLimit')),
            totalRevolvingBal = float(request.form.get('totalRevolvingBal')),
            avgOpenToBuy = float(request.form.get('avgOpenToBuy')),
            totalAmtChngQ4Q1 = float(request.form.get('totalAmtChngQ4Q1')),
            totalTransAmt = float(request.form.get('totalTransAmt')),
            totalTransCt = float(request.form.get('totalTransCt')),
            totalCtChngQ4Q1 = float(request.form.get('totalCtChngQ4Q1')),
            avgUtilizationRatio = float(request.form.get('avgUtilizationRatio')),
            Education_Level = request.form.get('Education_Level'),
            Marital_Status = request.form.get('Marital_Status'),
            Income_Category = request.form.get('Income_Category'),
            Gender = request.form.get('Gender'),
            Attrition_Flag = request.form.get('Attrition_Flag'),
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0")
