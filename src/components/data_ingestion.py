import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('data', 'train.csv')
    test_data_path: str = os.path.join('data', 'test.csv')
    raw_data_path: str = os.path.join('data', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion...")
        try:
            df = pd.read_csv(r'notebook\data\stud.csv')
            logging.info("Read dataset as dataframe")

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully.")
            return (
                self.config.train_data_path,
                self.config.test_data_path,
                self.config.raw_data_path
            )
        except Exception as e:
            #logging.error(f"Error occurred during data ingestion: {e}")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data, _ = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer= ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
