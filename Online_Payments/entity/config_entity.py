import os
import sys
from Online_Payments.exception.exception import CustomException
from  Online_Payments.logger.logger import logging
from datetime import datetime
from Online_Payments.constants.training_pipeline import ARTIFACT_DIR_PATH,DATA_INGESTION_FILE_PATH,DATA_INGESTION_INGESTED_FILE_PATH,DATASET_FILE_NAME,TRAIN_SET_FILE_NAME,TEST_SET_FILE_NAME,FEATURE_STORE_FILE_PATH,TRAIN_TEST_SPLIT_RATIO





# set up the training pipeline config
class TrainingPipelineConfig:
    try:
        def __init__(self,timestamp=datetime.now()):
            timestamp=timestamp.strftime("%d_%m_%y_%H_%M_%S")
            self.timestamp=timestamp
            self.artifact_dir_name=ARTIFACT_DIR_PATH
            self.artifact_dir_path=os.path.join(self.artifact_dir_name,self.timestamp)
    except Exception as e:
        raise CustomException(e,sys)
    

# set up the data ingestion config
class DataIngestionConfig:
    try:
        def __init__(self,training_pipeline_config: TrainingPipelineConfig):
            self.data_ingestion_dir=os.path.join(training_pipeline_config.artifact_dir_path,DATA_INGESTION_FILE_PATH)
            self.feature_store_file_path=os.path.join(self.data_ingestion_dir,FEATURE_STORE_FILE_PATH,DATASET_FILE_NAME)
            self.train_set_file_path=os.path.join(self.data_ingestion_dir,DATA_INGESTION_INGESTED_FILE_PATH,TRAIN_SET_FILE_NAME)
            self.test_set_file_path=os.path.join(self.data_ingestion_dir,DATA_INGESTION_INGESTED_FILE_PATH,TEST_SET_FILE_NAME)
            self.train_test_split_ratio=TRAIN_TEST_SPLIT_RATIO
            
    except Exception as e:
        raise CustomException(e,sys)