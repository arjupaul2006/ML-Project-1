# to fetch the data
from numpy import true_divide
import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# path where the train, test and raw data are stored
@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'data.csv')

# main class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self): 
        logging.info('Entered the data ingestion methods or components')
        try:
            # read the dataset
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info('Read the data in the dataframe')

            # create the artifacts directory or folder
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # store the dataset into data.csv file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train Test Split Initiate')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data Ingestion is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__=='__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()