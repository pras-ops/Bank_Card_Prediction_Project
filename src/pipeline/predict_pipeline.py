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
                 Customer_Age: int,
                 Dependent_count: int,
                 Months_on_book: int,
                 Total_Relationship_Count: int,
                 Months_Inactive_12_mon: int,
                 Contacts_Count_12_mon: int,
                 Credit_Limit: float,
                 Total_Revolving_Bal: float,
                 Avg_Open_To_Buy: float,
                 Total_Amt_Chng_Q4_Q1: float,
                 Total_Trans_Amt: float,
                 Total_Trans_Ct: float,
                 Total_Ct_Chng_Q4_Q1: float,
                 Avg_Utilization_Ratio: float,
                 Education_Level: object,
                 Marital_Status: object,
                 Income_Category: object,
                 Gender: object,
                 Attrition_Flag: object):
         
        self.Customer_Age = Customer_Age
        self.Dependent_count = Dependent_count
        self.Months_on_book = Months_on_book
        self.Total_Relationship_Count = Total_Relationship_Count
        self.Months_Inactive_12_mon = Months_Inactive_12_mon
        self.Contacts_Count_12_mon = Contacts_Count_12_mon
        self.Credit_Limit = Credit_Limit
        self.Total_Revolving_Bal = Total_Revolving_Bal
        self.Avg_Open_To_Buy = Avg_Open_To_Buy
        self.Total_Amt_Chng_Q4_Q1 = Total_Amt_Chng_Q4_Q1
        self.Total_Trans_Amt = Total_Trans_Amt
        self.Total_Trans_Ct = Total_Trans_Ct
        self.Total_Ct_Chng_Q4_Q1 = Total_Ct_Chng_Q4_Q1
        self.Avg_Utilization_Ratio = Avg_Utilization_Ratio
        self.Education_Level = Education_Level
        self.Marital_Status = Marital_Status
        self.Income_Category = Income_Category
        self.Gender = Gender
        self.Attrition_Flag = Attrition_Flag

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Customer_Age": [self.Customer_Age],
                "Dependent_count": [self.Dependent_count],
                "Months_on_book": [self.Months_on_book],
                "Total_Relationship_Count": [self.Total_Relationship_Count],
                "Months_Inactive_12_mon": [self.Months_Inactive_12_mon],
                "Contacts_Count_12_mon": [self.Contacts_Count_12_mon],
                "Credit_Limit": [self.Credit_Limit],
                "Total_Revolving_Bal": [self.Total_Revolving_Bal],
                "Avg_Open_To_Buy": [self.Avg_Open_To_Buy],
                "Total_Amt_Chng_Q4_Q1": [self.Total_Amt_Chng_Q4_Q1],
                "Total_Trans_Amt": [self.Total_Trans_Amt],
                "Total_Trans_Ct": [self.Total_Trans_Ct],
                "Total_Ct_Chng_Q4_Q1": [self.Total_Ct_Chng_Q4_Q1],
                "Avg_Utilization_Ratio": [self.Avg_Utilization_Ratio],
                "Education_Level": [self.Education_Level],
                "Marital_Status": [self.Marital_Status],
                "Income_Category": [self.Income_Category],
                "Gender": [self.Gender],
                "Attrition_Flag": [self.Attrition_Flag]
            }

            return pd.DataFrame(custom_data_input_dict)


        except Exception as e:
            raise CustomException(e, sys)
    