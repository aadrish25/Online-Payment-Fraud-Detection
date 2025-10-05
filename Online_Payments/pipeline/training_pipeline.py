import os
import sys
from Online_Payments.components.data_ingestion import DataIngestionComponent
from Online_Payments.components.data_validation import DataValidationComponent
from Online_Payments.components.data_transformation import DataTransformationComponent
from Online_Payments.components.model_trainer import ModelTrainerComponent
from Online_Payments.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig
from Online_Payments.entity.artifact_enity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact,ModelTrainerArtifact
from Online_Payments.utils.main_utils.utils import save_validation_set
from Online_Payments.exception.exception import CustomException
from Online_Payments.logger.logger import logging


class TrainingPipeline:
    def __init__(self):
        try:
            self.training_pipeline_config=TrainingPipelineConfig()
        except Exception as e:
            raise CustomException(e,sys)
        
    # start data ingestion
    def start_data_ingestion(self):
        try:
            logging.info("Starting data ingestion.")
            data_ingestion_config=DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            data_ingestion_component=DataIngestionComponent(training_pipeline_config=self.training_pipeline_config,data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact=data_ingestion_component.initiate_data_ingestion()
            logging.info("Completed data ingestion.")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e,sys)
        

    # start data validation
    def start_data_validation(self,data_ingestion_artifact):
        try:
            logging.info("Starting data validation.")
            dataset_path=data_ingestion_artifact.dataset_path
            schema_path=save_validation_set(csv_path=dataset_path)
            data_validation_config=DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation_component=DataValidationComponent(data_validation_config=data_validation_config,data_ingestion_artifact=data_ingestion_artifact,schema_path=schema_path)
            data_validation_artifact=data_validation_component.initiate_data_validation()
            logging.info("Completed data validation.")
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
    
    # start data transformation
    def start_data_transformation(self,data_validation_artifact):
        try:
            logging.info("Starting data transformation.")
            data_transformation_config=DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation_component=DataTransformationComponent(data_validation_artifact=data_validation_artifact,data_transformation_config=data_transformation_config)
            data_transformation_artifact=data_transformation_component.initiate_data_transformation()
            logging.info("Completed data transformation.")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
    # start model training
    def start_model_trainer(self,data_transformation_artifact):
        try:
            logging.info("Starting model training.")
            model_trainer_config=ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer_component=ModelTrainerComponent(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact=model_trainer_component.initiate_model_training()
            logging.info("Completed model training.")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
    # start the process
    def run_pipeline(self):
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact=self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact=self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)