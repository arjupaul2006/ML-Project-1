# to train the model
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import evalute_model, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            # split the train and test dataset
            logging.info('Split Train and Test dataset')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'LineearRegression': LinearRegression(),
                'KNearestNeighbour': KNeighborsRegressor(),
                'DecisionTree': DecisionTreeRegressor(),
                'RandomForest': RandomForestRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'GradientBoost': GradientBoostingRegressor(),
                'XgBoost': XGBRegressor(),
                'CatBoost': CatBoostRegressor(verbose=False)
            }

            # train the model
            model_report:dict = evalute_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # fetch the best model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('Not Best Model Exist')

            logging.info('Best Model found')

            # Save the model in the pickle file
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            y_pred = best_model.predict(X_test)
            score = r2_score(y_test, y_pred)
            return score

        except Exception as e:
            raise CustomException(e, sys)