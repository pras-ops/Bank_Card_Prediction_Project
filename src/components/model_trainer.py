import os
import sys
from dataclasses import dataclass


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from exception import CustomException
from logger import logging

from utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
            }
            params={
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256,300],
                    'n_jobs': [-1]

                },                
            }


            model_name = "Random Forest"
            selected_models = models[model_name]
            selected_params = params[model_name]

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models={model_name: selected_models}, param={model_name: selected_params})
            
            best_params = model_report[model_name]['best_params']
            best_model = selected_models.set_params(**best_params)  # Instantiate a new model with best parameters
            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            score = accuracy_score(y_test, predicted) 

            return score

        except Exception as e:
            raise CustomException(e, sys)