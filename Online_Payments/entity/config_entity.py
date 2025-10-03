import os
import sys
from Online_Payments.exception.exception import CustomException
from  Online_Payments.logger.logger import logging
from datetime import datetime
from Online_Payments.constants.training_pipeline import ARTIFACT_DIR_PATH,DATA_INGESTION_FILE_PATH,DATA_INGESTION_INGESTED_FILE_PATH,DATASET_FILE_NAME,TRAIN_SET_FILE_NAME,TEST_SET_FILE_NAME,FEATURE_STORE_FILE_PATH,TRAIN_TEST_SPLIT_RATIO
from Online_Payments.constants import training_pipeline




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
    

# set up data validation config
class DataValidationConfig:
    try:
        def __init__(self,training_pipeline_config: TrainingPipelineConfig):
            self.data_validation_dir=os.path.join(training_pipeline_config.artifact_dir_path,training_pipeline.DATA_VALIDAION_DIR)
            self.valid_data_dir=os.path.join(self.data_validation_dir,training_pipeline.DATA_VALIDAION_VALID_DATA_DIR)
            self.invalid_data_dir=os.path.join(self.data_validation_dir,training_pipeline.DATA_VALIDAION_INVALID_DATA_DIR)
            self.valid_train_file_path=os.path.join(self.valid_data_dir,training_pipeline.TRAIN_SET_FILE_NAME)
            self.invalid_train_file_path=os.path.join(self.invalid_data_dir,training_pipeline.TRAIN_SET_FILE_NAME)
            self.valid_test_file_path=os.path.join(self.valid_data_dir,training_pipeline.TEST_SET_FILE_NAME)
            self.invalid_test_file_path=os.path.join(self.invalid_data_dir,training_pipeline.TEST_SET_FILE_NAME)
            self.drift_report_file_path=os.path.join(self.data_validation_dir,training_pipeline.DATA_VALIDAION_DRIFT_REPORT_PATH,
                                                     training_pipeline.DATA_VALIDAION_DRIFT_REPORT_NAME)
            
    except Exception as e:
        raise CustomException(e,sys)



# set up data transformation config
class DataTransformationConfig:
    try:
        def __init__(self,training_pipeline_config: TrainingPipelineConfig):
            self.data_transformation_dir=os.path.join(training_pipeline_config.artifact_dir_path,training_pipeline.DATA_TRANSFORMATION_DIR)
            self.transformed_train_file_path=os.path.join(self.data_transformation_dir,training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DIR,
                                                          training_pipeline.DATA_TRANSFORMATION_TRAIN_FILE_NAME)
            self.transformed_test_file_path=os.path.join(self.data_transformation_dir,training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DIR,
                                                         training_pipeline.DATA_TRANSFORMATION_TEST_FILE_NAME)
            self.preprocessor_object_path=os.path.join(self.data_transformation_dir,training_pipeline.DATA_TRANSFORMATION_PREPROCESSOR_OBJECT_PATH,
                                                       training_pipeline.DATA_TRANSFORMATION_PREPROCESSOR_OBJECT_NAME)
    except Exception as e:
        raise CustomException(e,sys)
    

# set up model trainer config
class ModelTrainerConfig:
    try:
        def __init__(self,training_pipeline_config: TrainingPipelineConfig):
            self.model_trainer_dir=os.path.join(training_pipeline_config.artifact_dir_path,training_pipeline.MODEL_TRAINER_DIR)
            self.trained_model_dir=os.path.join(self.model_trainer_dir,training_pipeline.TRAINED_MODEL_FILE_PATH,training_pipeline.TRAINED_MODEL_OBJECT_NAME)
            self.expected_model_recall=training_pipeline.MODEL_EXPECTED_RECALL
            self.overfitting_underfitting_threshold=training_pipeline.MODEL_OVERFITTING_UNDERFITTING_THRESHOLD
    except Exception as e:
        raise CustomException(e,sys)