import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.decomposition import PCA

from exception import CustomException
from logger import logging
import os

from utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = [
                "Customer_Age",
                "Dependent_count",
                "Months_on_book",
                "Total_Relationship_Count",
                "Months_Inactive_12_mon",
                "Contacts_Count_12_mon",
                "Credit_Limit",
                "Total_Revolving_Bal",
                "Avg_Open_To_Buy",
                "Total_Amt_Chng_Q4_Q1",
                "Total_Trans_Amt",
                "Total_Trans_Ct",
                "Total_Ct_Chng_Q4_Q1",
                "Avg_Utilization_Ratio",
            ]

            categorical_columns = [
                "Education_Level",
                "Marital_Status",
                "Income_Category",
                "Gender",
                "Attrition_Flag"
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),

                ]
            )

            cat_pipeline=Pipeline(

                steps=[

                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(drop="first", handle_unknown="ignore"))
                
                ]

            )
            #pca = PCA(n_components=7)  # Define PCA with the desired number of components
            pca2 = PCA(n_components = 10)

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns),
                #("pca", pca, numerical_columns),  # Add PCA as a step in the pipeline
                ("pca2", pca2, numerical_columns)

                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="Card_Category"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)