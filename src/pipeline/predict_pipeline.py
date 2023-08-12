import sys
import pandas as pd
import os
from exception import CustomException
from utils import load_object
from sklearn.decomposition import PCA


from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
                 name: str,
                 age: int,
                 gender: str,
                 children: int,
                 edu: int,
                 marital: int,
                 income: int,
                 monthOnBooks: int,
                 totalRelationshipCount: int,
                 monthsInactive12mon: int,
                 contactsCount12mon: int,
                 creditLimit: float,
                 totalRevolvingBal: float,
                 avgOpenToBuy: float,
                 totalAmtChngQ4Q1: float,
                 totalTransAmt: float,
                 totalTransCt: float,
                 totalCtChngQ4Q1: float,
                 avgUtilizationRatio: float,
                 Education_Level:object,
                 Marital_Status:object,
                 Income_Category:object,
                 Gender:object,
                 Attrition_Flag:object,):
         
        self.name = name
        self.age = age
        self.gender = gender
        self.children = children
        self.edu = edu
        self.marital = marital
        self.income = income
        self.monthOnBooks = monthOnBooks
        self.totalRelationshipCount = totalRelationshipCount
        self.monthsInactive12mon = monthsInactive12mon
        self.contactsCount12mon = contactsCount12mon
        self.creditLimit = creditLimit
        self.totalRevolvingBal = totalRevolvingBal
        self.avgOpenToBuy = avgOpenToBuy
        self.totalAmtChngQ4Q1 = totalAmtChngQ4Q1
        self.totalTransAmt = totalTransAmt
        self.totalTransCt = totalTransCt
        self.totalCtChngQ4Q1 = totalCtChngQ4Q1
        self.avgUtilizationRatio = avgUtilizationRatio
        self.Education_Level=Education_Level
        self.Marital_Status=Marital_Status
        self.Income_Category=Income_Category
        self.Gender=Gender
        self.Attrition_Flag=Attrition_Flag

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Customer_Age": [self.age],
                "Dependent_count": [self.children],
                "Months_on_book": [self.monthOnBooks],
                "Total_Relationship_Count": [self.totalRelationshipCount],
                "Months_Inactive_12_mon": [self.monthsInactive12mon],
                "Contacts_Count_12_mon": [self.contactsCount12mon],
                "Credit_Limit": [self.creditLimit],
                "Total_Revolving_Bal": [self.totalRevolvingBal],
                "Avg_Open_To_Buy": [self.avgOpenToBuy],
                "Total_Amt_Chng_Q4_Q1": [self.totalAmtChngQ4Q1],
                "Total_Trans_Amt": [self.totalTransAmt],
                "Total_Trans_Ct": [self.totalTransCt],
                "Total_Ct_Chng_Q4_Q1": [self.totalCtChngQ4Q1],
                "Avg_Utilization_Ratio": [self.avgUtilizationRatio],
                "Education_Level": [self.Education_Level],
                "Marital_Status": [self.Marital_Status],
                "Income_Category": [self.Income_Category],
                "Gender": [self.Gender],
                "Attrition_Flag": [self.Attrition_Flag]
            }

            return pd.DataFrame(custom_data_input_dict)


        except Exception as e:
            raise CustomException(e, sys)
    