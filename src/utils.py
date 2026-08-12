import os
import sys
# pyrefly: ignore [missing-import]
import dill

from src.exception import CustomException

def save_preprocessing(file_path, preprocessor):
    try:
        dir_path = os.path.dirname(file_path)   # extract the folder from the file_path

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(preprocessor, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    