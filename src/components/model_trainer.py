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
from sklearn.model_selection import GridSearchCV

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

            params = {
                'Random Forest' : {
                    'n_estimators':[10,20,30],
                    'max_depth' : [3,4,5]

                },
                'Gradient Boosting' :{
                    'n_estimators': [10,20,30],
                    'max_depth' : [3,4,5],
                    'learning_rate':[0.01,0.1,1.0]
                },
                'AdaBoost Regressor' : {
                    'n_estimators':[10,20,30],
                    'learning_rate':[0.01,0.1,1.0]
                },
                'Linear Regression' :{},
                'Decision Tree': {
                    'max_depth':[3,4,5],
                    'min_samples_split':[2,3,4]
                },
                'KNN Regressor':{
                    'n_neighbors':[3,5,7,9]
                },
                'CatBoost Regressor' :{
                    'iterations': [3,5,10],
                    'learning_rate':[0.01,0.1,1.0],
                    'depth':[3,5,10]
                },
                'XGBRegressor':{
                    'n_estimators':[10,20,30],
                    'learning_rate':[0.01,0.1,1.0],
                    'max_depth':[3,4,5]
                }

            }
            model_report = {}
            best_estimators = {}
            for model_name, model in models.items():
               
               gs = GridSearchCV(
                   estimator=model,
                   param_grid=params[model_name],
                   cv=3,
                   n_jobs=1,
                   scoring='r2'
               )
               gs.fit(X_train, y_train)
               best_estimators[model_name] = gs.best_estimator_
               model_report[model_name] = gs.best_score_
               log.debug(f'ESTIMATOR: {model_name} , and its CV R2 SCRORE: {gs.best_score_}')
               

            
            best_model_name= max(model_report, key=model_report.get)
            best_model = best_estimators[best_model_name]
            
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_r2_score = r2_score(y_train,y_train_pred)
            test_r2_score = r2_score(y_test,y_test_pred)

            log.debug(f'FINAL BEST MODEL: {best_model_name}')
            log.debug(f'FINAL TEST R2 SCORE : {test_r2_score}')
            log.debug(f'FINAL TRAIN R2 SCORE : {train_r2_score}')



            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name , test_r2_score, 
        
        except Exception as e:
            log.error('Error in model training')
            raise e

    