import os

from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.components.utils import save_object


@dataclass
class DataTransformationConfig:
    pre_process_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, train_df):
        try:
            numerical_features = train_df.select_dtypes(include='number').columns.tolist()
            categorical_features = train_df.select_dtypes(exclude='number').columns.tolist()

            num_pipline = Pipeline(
                [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(
                                        handle_unknown='ignore',
                                        sparse_output=False)
                    )
                ]
            )
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipline, numerical_features),
                    ('cat_pipeline', cat_pipeline, categorical_features)
                ]
            )

            print('Pipeline created successfully')
            return preprocessor
        except Exception as e:
            print('Error in data transformation')
            raise e

    def initiate_data_transformation(self, train_path, test_path,target_column_name):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            print('Read train and test data completed')

            
            #target_column_name = 'writing score'
            #numerical_features = ['math score', 'reading score']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns =[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            preprocessor_obj = self.get_data_transformer_object(input_feature_train_df)

                     
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path = self.data_transformation_config.pre_process_obj_file_path,
                obj = preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.pre_process_obj_file_path
            )
        
        except Exception as e:
            print('Error in data transformation')
            raise e

