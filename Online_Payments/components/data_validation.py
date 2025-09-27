import os
import sys
from Online_Payments.exception.exception import CustomException
from Online_Payments.logger.logger import logging
from Online_Payments.constants import training_pipeline
from Online_Payments.entity.config_entity import TrainingPipelineConfig,DataValidationConfig
from Online_Payments.entity.artifact_enity import DataIngestionArtifact,DataValidationArtifact
from Online_Payments.utils.main_utils.utils import read_yaml_file,write_yaml_file
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np





class DataValidationComponent:
    def __init__(self,data_validation_config:DataValidationConfig,data_ingestion_artifact:DataIngestionArtifact,schema_path):
        try:
            self.data_validation_config=data_validation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.schema_config=read_yaml_file(schema_path)
        except Exception as e:
            raise CustomException(e,sys)
        

    # function to check the number of columns
    def validate_no_of_columns(self,dataframe:pd.DataFrame):
        try:
            no_of_columns=len(self.schema_config["columns"])
            logging.info(f"Required no of columns: {no_of_columns}")

            columns_in_df=len(dataframe.columns)
            logging.info(f"Columns in dataframe: {columns_in_df}")

            if no_of_columns==columns_in_df:
                return True
            return False
        except Exception as e:
            raise CustomException(e,sys)
        

    # function to validate the numerical columns
    def validate_numerical_columns(self,dataframe: pd.DataFrame):
        try:
            numerical_columns=self.schema_config["numerical columns"]
            logging.info(f"Required numerical columns: {numerical_columns}")

            dataframe_cols=dataframe.columns
            logging.info(f"Dataframe has numerical columns: {dataframe_cols}")

            for col in numerical_columns:
                if col not in dataframe_cols:
                    return False
                
            return True
        except Exception as e:
            raise CustomException(e,sys)
        

    # detect drift in dataset
    def detect_drift(self,base_df: pd.DataFrame,current_df:pd.DataFrame,threshold=0.05):
        try:
            status =True
            report={}


            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]

                is_same_dist=ks_2samp(data1=d1,data2=d2)

                if(is_same_dist.pvalue>=threshold):
                    is_found=False
                else:
                    is_found=True
                    status=False

                report.update({
                    column:{
                        "p-value":float(is_same_dist.pvalue),
                        "is_found":bool(is_found)
                    }
                })

            # finally save the report to a directory
            drift_report_dir=os.path.dirname(self.data_validation_config.drift_report_file_path)
            os.makedirs(drift_report_dir,exist_ok=True)

            # write the report to the yaml file
            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path,content=report)

            return status

        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_validation(self):
        try:
            logging.info("Enter the initiate_data_validation class")
            # Step 1-> read the train and test data from data ingestion artifact
            train_set=pd.read_csv(self.data_ingestion_artifact.train_path)
            test_set=pd.read_csv(self.data_ingestion_artifact.test_path)
            logging.info("Step 1-> Read train and test data from data ingestion artifact completed.")

            # Step 2-> validate number of columns
            train_status=self.validate_no_of_columns(dataframe=train_set)
            if not train_status:
                raise CustomException("Missing columns in training set",sys)
            
            test_status=self.validate_no_of_columns(dataframe=test_set)
            if not test_status:
                raise CustomException("Missing columns in test set",sys)
            
            logging.info("Step 2-> Validate no of columns completed.")

            # Step 3-> validate numerical columns

            train_status_numerical=self.validate_numerical_columns(dataframe=train_set)
            if not train_status_numerical:
                raise CustomException("Missing numerical columns in training set",sys)
            
            test_status_numerical=self.validate_numerical_columns(dataframe=test_set)
            if not test_status_numerical:
                raise CustomException("Missing numerical columns in test set",sys)
            
            logging.info("Step 3-> Validate numerical columns completed.")

            # Step 4-> detect drift in dataset
            drift_status=self.detect_drift(base_df=train_set,current_df=test_set)

            valid_train_dir=os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(valid_train_dir,exist_ok=True)

            # if drift status is ok, then save the valid train and test files
            if drift_status:
                train_set.to_csv(
                    self.data_validation_config.valid_train_file_path,index=False,header=True
                )

                test_set.to_csv(
                    self.data_validation_config.valid_test_file_path,index=False,header=True
                )

            logging.info("Step 4-> drift status checking done, saved the validated train and test sets.")

            # finally return the data validation artifacts
            data_validation_artifacts=DataValidationArtifact(
                validation_status=drift_status,
                valid_train_file_path=self.data_ingestion_artifact.train_path,
                valid_test_file_path=self.data_ingestion_artifact.test_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(f"Data validation completed, data validation artifacts: {data_validation_artifacts}")

            return data_validation_artifacts
        except Exception as e:
            raise CustomException(e,sys)