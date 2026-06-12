import os
import sys

from sklearn.ensemble import(
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
    
)
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from src.components.utils import save_object
from src.logger import logging

log  = logging.getLogger('file1')

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:

    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        

    def initiate_model_trainer(self, train_array, test_array):
        try:
            print('Splitting training and test input data')
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]

            models = {
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor(),
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(),
                'KNN Regressor': KNeighborsRegressor(),
                'CatBoost Regressor': CatBoostRegressor(verbose=False),
                'XGBRegressor': XGBRegressor()
            }

            model_report = {}

            for model_name, model in models.items():
               
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)

                model_report[model_name] = test_model_score

            best_model_score = max(model_report.values())

            best_model_name = max(model_report , key=model_report.get)
            best_model = model_report[best_model_name]

            log.debug('Best model found on both training and testing dataset')
            log.debug('Model Report ')
            log.debug(model_report)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            log.error('Error in model training')
            raise e

    