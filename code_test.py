from Online_Payments.components.data_ingestion import DataIngestionComponent,DataIngestionArtifact
from Online_Payments.components.data_validation import DataValidationComponent,DataValidationArtifact
from Online_Payments.components.data_transformation import DataTransformationComponent
from Online_Payments.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig,DataTransformationConfig
from Online_Payments.entity.artifact_enity import DataIngestionArtifact,DataTransformationArtifact
from Online_Payments.exception.exception import CustomException
from Online_Payments.logger.logger import logging
from Online_Payments.utils.main_utils.utils import save_validation_set
import sys


if __name__=="__main__":
    try:
        # test the data ingestion part
        training_pipeline_config=TrainingPipelineConfig()
        data_ingestion_config=DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion_component=DataIngestionComponent(training_pipeline_config=training_pipeline_config,data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact=data_ingestion_component.initiate_data_ingestion()

        # create the schema for data validation
        dataset_path=data_ingestion_artifact.dataset_path
        schema_path=save_validation_set(dataset_path)

        # test the data validation part
        data_validation_config=DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation_component=DataValidationComponent(data_validation_config=data_validation_config,data_ingestion_artifact=data_ingestion_artifact,schema_path=schema_path)
        data_validation_artifact=data_validation_component.initiate_data_validation()

        # test the data transformation part
        data_transformation_config=DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation_component=DataTransformationComponent(data_transformation_config=data_transformation_config,data_validation_artifact=data_validation_artifact)
        data_transformation_artifact=data_transformation_component.initiate_data_transformation()
        print(data_transformation_artifact)
    except Exception as e:
        raise CustomException(e,sys)