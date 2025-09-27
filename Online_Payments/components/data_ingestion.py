import os
import sys
from Online_Payments.exception.exception import CustomException
from Online_Payments.logger.logger import logging
from Online_Payments.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from Online_Payments.entity.artifact_enity import DataIngestionArtifact


class DataIngestionComponent:
    def __init__(self,training_pipeline_config: TrainingPipelineConfig,data_ingestion_config:DataIngestionConfig):
        try:
            self.training_pipeline_config=training_pipeline_config
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise CustomException(e,sys)
        

    def prepare_raw_data(self)->pd.DataFrame:
        try:
            df=pd.read_csv("E:\MLprojects\Online-Payment-Fraud-Detection\online_payments_data\payments.csv")
            # remove the isFlaggedFraud column
            df=df.drop(labels="isFlaggedFraud",axis=1)
            return df
        except Exception as e:
            raise CustomException(e,sys)
        

    def save_data_to_feature_store(self,dataframe: pd.DataFrame):
        try:
            feature_store_dir=self.data_ingestion_config.feature_store_file_path
            feature_store_dir_name=os.path.dirname(feature_store_dir)
            os.makedirs(feature_store_dir_name,exist_ok=True)

            dataframe.to_csv(feature_store_dir,index=False,header=True)
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)
    
    def save_train_test_data(self,train_set:pd.DataFrame,test_set:pd.DataFrame):
        try:
            train_set_dir=self.data_ingestion_config.train_set_file_path
            train_set_dir_name=os.path.dirname(train_set_dir)
            os.makedirs(train_set_dir_name,exist_ok=True)

            # save the train and test spits
            train_set.to_csv(train_set_dir)
            test_set.to_csv(self.data_ingestion_config.test_set_file_path)
        except Exception as e:
            raise CustomException(e,sys)

    # initiate data ingestion class
    def initiate_data_ingestion(self):
        try:
            logging.info("Entered initiate data ingestion class")
            # step 1-> take the raw data and store it to feature store
            dataframe=self.prepare_raw_data()
            dataframe=self.save_data_to_feature_store(dataframe=dataframe)
            logging.info("Step 1-> save data to feature store completed.")

            # step 2-> get train set and test set
            train_set,test_set=train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_split_ratio,random_state=42)

            logging.info("Step 2-> Obtain the train set and test set completed.")

            # step 3-> save the train set and test set to supposed location
            self.save_train_test_data(train_set=train_set,test_set=test_set)

            logging.info("Step 3-> save the train and test sets to respective folders completed.")

            # finally return data ingestion artifacts
            data_ingestion_artifacts=DataIngestionArtifact(train_path=self.data_ingestion_config.train_set_file_path,
                                                           test_path=self.data_ingestion_config.test_set_file_path,
                                                           dataset_path=self.data_ingestion_config.feature_store_file_path)
            
            logging.info(f"Data ingestion completed, data ingestion artifacts: {data_ingestion_artifacts}")

            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e,sys)