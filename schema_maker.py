import os
import sys
import yaml
from Online_Payments.constants.training_pipeline import DATA_SCHEMA_FILE_PATH
from Online_Payments.exception.exception import CustomException
import pandas as pd



# read clean data and prepare the numerical columns and column list
def save_validation_set(csv_path):
    try:
        # create the data path
        data_schema_path=os.path.join(DATA_SCHEMA_FILE_PATH,"report.yaml")
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

    except Exception as e:
        raise CustomException(e,sys)
    
