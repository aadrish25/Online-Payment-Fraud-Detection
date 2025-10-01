import os
import sys
import pandas as pd
import numpy as np
from Online_Payments.exception.exception import CustomException
from Online_Payments.logger.logger import logging
from Online_Payments.entity.config_entity import DataTransformationConfig
from Online_Payments.entity.artifact_enity import DataValidationArtifact,DataTransformationArtifact
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from Online_Payments.utils.main_utils.utils import save_numpy_arr,save_pickle_object

class DataTransformationComponent:
    def __init__(self,data_transformation_config:DataTransformationConfig,data_validation_artifact:DataValidationArtifact):
        try:
            self.data_transformation_component=data_transformation_config
            self.data_validation_artifact=data_validation_artifact
        except Exception as e:
            raise CustomException(e,sys)

    # static method to read the train and test set from data validation artifact    
    @staticmethod
    def read_train_test_data(data_path):
        try:
            return pd.read_csv(data_path)
        except Exception as e:
            raise CustomException(e,sys)

    # function to prepare the preprocessor object
    def prepare_preprocessor_object(self,df:pd.DataFrame):
        try:
            # first set up the column transformer
            # OHE on categorical columns
            categorical_columns=df.select_dtypes(include='object').columns
            # scaling on numerical columns
            numerical_columns=df.select_dtypes(exclude='object').columns
            # we'll also take knn imputer
            imputer=SimpleImputer(strategy="most_frequent")
            # prepare categorical pipeline
            categorical_pipeline=Pipeline([
                ("imputer",imputer),
                ("oh_encoder",OneHotEncoder(drop='first'))
            ])

            # prepare numerical pipeline
            numerical_pipeline=Pipeline([
                ("imputer",imputer),
                ("scaler",StandardScaler())
            ])

            # finally prepare the preprocessor
            preprocessor=ColumnTransformer([
                ("categorical_transformer",categorical_pipeline,categorical_columns),
                ("numerical_transformer",numerical_pipeline,numerical_columns)
            ],remainder='passthrough')

            logging.info("Preprocessor object is ready.")
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self):
        try:
            logging.info("Entered the initiate data transformation class.")
            # Step 1-> read the valid train and test data from data validation artifact
            train_set=DataTransformationComponent.read_train_test_data(data_path=self.data_validation_artifact.valid_train_file_path)
            test_set=DataTransformationComponent.read_train_test_data(data_path=self.data_validation_artifact.valid_test_file_path)
            logging.info("Step 1-> Read the train and test data from data validation artifact.")

            # Step 2-> Drop unimportant columns and divide the train_set and test_set based on target and input
            train_set=train_set.drop(labels=['nameOrig','nameDest'],axis=1)
            test_set=test_set.drop(labels=['nameOrig','nameDest'],axis=1)

            input_feature_train_df=train_set.drop(labels=["isFraud"],axis=1)
            target_feature_train_df=train_set['isFraud']

            input_feature_test_df=test_set.drop(labels=["isFraud"],axis=1)
            target_feature_test_df=test_set['isFraud']

            logging.info("Step 2-> Drop unimportant columns and divide the train_set and test_set based on target and input completed.")

            # Step 3-> prepare the preprocessor object
            preprocessor_object=self.prepare_preprocessor_object(df=input_feature_train_df)
            logging.info("Step 3-> Obtained the preprocessor object.")

            # Step 4-> apply the preprocessor object on input feature train and test
            input_feature_train_transformed_df=preprocessor_object.fit_transform(input_feature_train_df)
            input_feature_test_transformed_df=preprocessor_object.transform(input_feature_test_df)

            logging.info("Step 4-> Applied the preprocessor on train and test set.")

            # Step 5-> Combine the transformed arrays
            train_arr=np.c_[input_feature_train_transformed_df,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_transformed_df,np.array(target_feature_test_df)]

            logging.info("Step 5-> Combined the arrays to obtain train_arr and test_arr")

            # Step 6-> Save the train array and test array, and the preprocessor object as well  
            save_numpy_arr(path=self.data_transformation_component.transformed_train_file_path,array=train_arr)
            save_numpy_arr(path=self.data_transformation_component.transformed_test_file_path,array=test_arr)

            save_pickle_object(path=self.data_transformation_component.preprocessor_object_path,object=preprocessor_object)

            logging.info("Step 6-> Saved the numpy arrays and preprocessor objects.")

            # finally return the artifacts
            data_transformation_artifact=DataTransformationArtifact(
                transformed_train_array_path=self.data_transformation_component.transformed_train_file_path,
                transformed_test_array_path=self.data_transformation_component.transformed_test_file_path,
                preprocessor_object_path=self.data_transformation_component.preprocessor_object_path
            )

            logging.info("Data transformation completed.")

            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e,sys)