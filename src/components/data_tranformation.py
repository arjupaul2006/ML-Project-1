# to transform the data
import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer    # handles the missing values
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTranfromationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transfromation_config = DataTranfromationConfig()

    # to change the catergorical features to numerical features and standardization 
    def get_data_transformer_object(self):
        try:
            num_features = ['reading_score', 'writing_score']
            cat_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            # step by step pipeline for numerical features -
            # 1. handle the missing values
            # 2. standardize the numerical values
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # step by step pipeline for categorical features -
            # 1. handle the missing values
            # 2. one hot encode the categorical values
            # 3. standardize the numerical values
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one-hot-encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info('Numerical Columns encoding completed')
            logging.info('Categorical Columns encoding completed')
            
            # combine the pipelines using column transformer
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_features),
                    ('cat_pipeline', cat_pipeline, cat_features)
                ]
            )
            
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # to apply the preprocessor to the train and test data and save the preprocessor in preprocessor.pkl file
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            
            logging.info('Obtaining preprocessing object')
            # fetch the preprocessing obj from the previous function
            preprocessing_obj = self.get_data_transformer_object()

            target_col_name = 'math_score'
            num_features = ['reading_score', 'writing_score']

            # separate the input and target features
            input_features_train_df = train_df.drop(target_col_name, axis=1)    # X_train
            target_feature_train_df = train_df[target_col_name]     # y_train

            input_features_test_df = test_df.drop(target_col_name, axis=1)    # X_test
            target_feature_test_df = test_df[target_col_name]     # y_test

            logging.info('Applying preprocessing object on the train and test datafarme')

            # fit_transform - learn from the training data and transform it
            # transform - transform the test data using the learned parameters from the training data
            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)

            # np.c_ means column-wise concatenation
            train_arr = np.c_[
                input_features_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_features_test_arr, np.array(target_feature_test_df)
            ]

            logging.info('Saved Preprocessor Object')
            # save the preprocessor into preprocessor.pkl file
            save_object(
                file_path=self.data_transfromation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return(
                train_arr, 
                test_arr, 
                self.data_transfromation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
