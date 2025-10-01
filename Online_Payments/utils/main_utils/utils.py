import os
import sys
import yaml
from Online_Payments.exception.exception import CustomException
from Online_Payments.constants.training_pipeline import DATA_SCHEMA_FILE_PATH
import pandas as pd
import numpy as np
import pickle as pkl



# read clean data and prepare the numerical columns and column list
def save_validation_set(csv_path):
    try:
        # create the data path
        data_schema_path=os.path.join(DATA_SCHEMA_FILE_PATH,"schema.yaml")
        os.makedirs("online_data",exist_ok=True)

        df=pd.read_csv(csv_path)

        schema_columns=[]
        numerical_columns=[]

        for column in df.columns:
            dtype=str(df[column].dtype)

            if "int" in dtype:
                mapped_dtype="int64"
                numerical_columns.append(column)
            elif "float" in dtype:
                mapped_dtype="float64"
                numerical_columns.append(column)
            else:
                mapped_dtype="string"

            schema_columns.append({column:mapped_dtype}) 

        schema={
            "columns":schema_columns,
            "numerical columns":numerical_columns
        }

        with open(data_schema_path,"w") as file:
            yaml.dump(schema,file)

        return data_schema_path
    except Exception as e:
        raise CustomException(e,sys)


def read_yaml_file(file_path)-> dict:
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e: 
        raise CustomException(e,sys)
    
def write_yaml_file(file_path,content,status=False)-> dict:
    try:
        if status:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as yaml_file:
            yaml.dump(data=content,stream=yaml_file)
    except Exception as e:
        raise CustomException(e,sys)
    
def save_numpy_arr(path:str,array:np.array):
    try:
        dir_path=os.path.dirname(path)
        os.makedirs(dir_path,exist_ok=True)

        with open(path,"wb") as file_object:
            np.save(file_object,array)
    except Exception as e:
        raise CustomException(e,sys)
    
def save_pickle_object(path: str,object):
    try:
        dir_path=os.path.dirname(path)
        os.makedirs(dir_path,exist_ok=True)

        with open(path,"wb") as file_object:
            pkl.dump(object,file_object)
    except Exception as e:
        raise CustomException(e,sys)