import os
import sys
import yaml
from Online_Payments.exception.exception import CustomException
from Online_Payments.constants.training_pipeline import DATA_SCHEMA_FILE_PATH
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import recall_score



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
    

def load_numpy_arr(path: str)->np.array:
    try:
        with open(path,"rb") as file_object:
            arr_data=np.load(file_object)
            return arr_data
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
    

def load_pickle_object(path: str):
    try:
        with open(path,"rb") as file_object:
            object=pkl.load(file_object)
            return object 
    except Exception as e:
        raise CustomException(e,sys)
    

# evaluate models and return a model report
def evaluate_models(models: dict,params: dict,X_train,X_test,y_train,y_test):
    try:
        model_report={}

        for i in range(len(list(models))):
            # get the model and model_name
            model=list(models.values())[i]
            model_name=list(models.keys())[i]

            # now load the parameters for the model
            model_params=params[model_name]

            # and now train randomized search cv
            rs_classifer=RandomizedSearchCV(estimator=model,param_distributions=model_params,cv=3,n_jobs=-1,verbose=1)
            rs_classifer.fit(X_train,y_train)

            # get the best params for model
            best_params=rs_classifer.best_params_
            # now train with the best params
            model.set_params(**best_params)
            model.fit(X_train,y_train)

            # get predictions for train set
            y_pred_train=rs_classifer.predict(X_train)
            # get predictions for test set
            y_pred_test=rs_classifer.predict(X_test)

            # calculate recall score for train and test set
            recall_train=recall_score(y_true=y_train,y_pred=y_pred_train)
            recall_test=recall_score(y_true=y_test,y_pred=y_pred_test)

            model_report[model_name]=recall_test
            
        return model_report

    except Exception as e:
        raise CustomException(e,sys)