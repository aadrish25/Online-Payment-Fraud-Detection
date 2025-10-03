import os



# common variables 
TARGET_COLUMN="isFraud"
ARTIFACT_DIR_PATH="Artifacts"
DATASET_FILE_NAME="online_payments.csv"
TRAIN_SET_FILE_NAME="train.csv"
TEST_SET_FILE_NAME="test.csv"

DATA_SCHEMA_FILE_PATH="online_data"

# variables related to data ingestion
DATA_INGESTION_FILE_PATH="data_ingestion"
FEATURE_STORE_FILE_PATH="feature_store"
DATA_INGESTION_INGESTED_FILE_PATH="ingested"
TRAIN_TEST_SPLIT_RATIO=0.2


# variables related to data validaion
DATA_VALIDAION_DIR="data_validation"
DATA_VALIDAION_VALID_DATA_DIR="validated"
DATA_VALIDAION_INVALID_DATA_DIR="invalidated"
DATA_VALIDAION_DRIFT_REPORT_PATH="drift_report"
DATA_VALIDAION_DRIFT_REPORT_NAME="report.yaml"

# variables related to data transformation
DATA_TRANSFORMATION_DIR="data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR="transformed"
DATA_TRANSFORMATION_TRAIN_FILE_NAME="train.npy"
DATA_TRANSFORMATION_TEST_FILE_NAME="test.npy"
DATA_TRANSFORMATION_PREPROCESSOR_OBJECT_PATH="preprocessor_object"
DATA_TRANSFORMATION_PREPROCESSOR_OBJECT_NAME="preprocessor.pkl"

# variables related to model trainer
MODEL_TRAINER_DIR="model_trainer"
TRAINED_MODEL_FILE_PATH="model"
TRAINED_MODEL_OBJECT_NAME="model.pkl"
MODEL_EXPECTED_RECALL:float=0.7
MODEL_OVERFITTING_UNDERFITTING_THRESHOLD:float=0.05