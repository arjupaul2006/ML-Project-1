import sys
import os
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e, sys)

        

class CustomData:
    def __init__(self, gender, ethnicity, parent_education, lunch, test_course, writing_score, reading_score):
        self.gender = gender
        self.ethnicity = ethnicity
        self.parent_education = parent_education
        self.lunch = lunch
        self.test_course = test_course
        self.writing_score = writing_score
        self.reading_score = reading_score

    def get_data_ass_dataframe(self):
        try:
            data = {
                'gender': [self.gender],
                'race_ethnicity': [self.ethnicity],
                'parental_level_of_education': [self.parent_education],
                'lunch': [self.lunch],
                'test_preparation_course': [self.test_course],
                'reading_score': [self.reading_score],
                'writing_score': [self.writing_score],
            }

            return pd.DataFrame(data)
            
        except Exception as e:
            raise CustomException(e, sys)