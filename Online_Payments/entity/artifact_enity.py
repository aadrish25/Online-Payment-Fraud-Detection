import os,sys
from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_path:str
    test_path:str
    dataset_path:str

@dataclass
class DataValidationArtifact:
    validation_status:str
    valid_train_file_path:str
    valid_test_file_path:str
    invalid_train_file_path:str
    invalid_test_file_path:str
    drift_report_file_path:str


@dataclass
class DataTransformationArtifact:
    transformed_train_array_path:str
    transformed_test_array_path:str
    preprocessor_object_path:str

@dataclass
class ModelTrainerArtifact:
    final_model_path:str
    train_metric_artifact:str
    test_metric_artifact:str


@dataclass
class ClassificationMetricArtifact:
    accuracy_score:float
    precision:float
    recall:float
    f1_score:float
    roc_score:float
