from Online_Payments.components.data_ingestion import DataIngestionComponent,DataIngestionArtifact
from Online_Payments.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
from Online_Payments.entity.artifact_enity import DataIngestionArtifact
from Online_Payments.exception.exception import CustomException
from Online_Payments.logger.logger import logging
from schema_maker import save_validation_set
import sys


if __name__=="__main__":
    try:
        # test the data ingestion part
        training_pipeline_config=TrainingPipelineConfig()
        data_ingestion_config=DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion_component=DataIngestionComponent(training_pipeline_config=training_pipeline_config,data_ingestion_config=data_ingestion_config)

        data_ingestion_artifact=data_ingestion_component.initiate_data_ingestion()
        dataset_path=data_ingestion_artifact.dataset_path
        save_validation_set(dataset_path)
    except Exception as e:
        raise CustomException(e,sys)