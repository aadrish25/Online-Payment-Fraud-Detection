import os
import sys
from Online_Payments.exception.exception import CustomException
from Online_Payments.logger.logger import logging
from Online_Payments.entity.config_entity import ModelTrainerConfig
from Online_Payments.entity.artifact_enity import DataTransformationArtifact,ModelTrainerArtifact
from Online_Payments.utils.main_utils.utils import load_numpy_arr,evaluate_models,save_pickle_object,load_pickle_object
from Online_Payments.utils.ml_utils.metric.classification_metric import get_classification_metrics
from Online_Payments.utils.ml_utils.model.fraud_detection_model import FraudDetectionModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class ModelTrainerComponent:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)
        

    def train_model(self,X_train,y_train,X_test,y_test):
        try:
            self.models={
                "Logistic Regression":LogisticRegression(n_jobs=-1,verbose=1),
                "Decision Tree":DecisionTreeClassifier()
                # "Random Forest":RandomForestClassifier(n_jobs=-1,verbose=1)
            }

            params={
                "Logistic Regression":{
                    "class_weight":[{0:1,1:10},{0:5,1:10},{0:5,1:50},{0:10,1:50},{0:10,1:100}]
                },
                "Decision Tree":{
                    "splitter":["best","random"],
                    "min_samples_split":[2,5,7,10,15,20,25,30,35,40,45,50],
                    "class_weight":[{0:1,1:10},{0:5,1:10},{0:5,1:50},{0:10,1:50},{0:10,1:100}]
                }
                # "Random Forest":{
                #     "class_weight":[{0:1,1:10},{0:5,1:10},{0:5,1:50},{0:10,1:50},{0:10,1:100}]
                # }
            }

            model_report=evaluate_models(models=self.models,params=params,X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)

            return model_report
        except Exception as e:
            raise CustomException(e,sys)
        

    def get_classification_metric(self,X_train,y_train,X_test,y_test,model_report: dict):
        try:
            # get best model score
            best_model_score=max(list(model_report.values()))

            # get best model name
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # now load the best model 
            best_model=self.models[best_model_name]

            # # train wrt best model
            # best_model.fit(X_train,y_train)

            # get train prediction
            y_train_pred=best_model.predict(X_train)
            y_train_prob=best_model.predict_proba(X_train)

            # get test prediction
            y_test_pred=best_model.predict(X_test)
            y_test_prob=best_model.predict_proba(X_test)

            # get train classification metrics
            train_classification_metrics=get_classification_metrics(y_true=y_train,y_pred=y_train_pred,y_score=y_train_prob[:,1])

            # get test classification metrics
            test_classification_metrics=get_classification_metrics(y_true=y_test,y_pred=y_test_pred,y_score=y_test_prob[:,1])

            # now return the metrics
            return (
                train_classification_metrics,
                test_classification_metrics,
                best_model
            )

        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_model_training(self):
        try:
            logging.info("Entered initiate_model_training class.")

            # Step 1-> Read the train array and test array
            train_array=load_numpy_arr(path=self.data_transformation_artifact.transformed_train_array_path)
            test_array=load_numpy_arr(path=self.data_transformation_artifact.transformed_test_array_path)

            logging.info("Step 1-> Read the train and test arrays, completed.")

            # Step 2-> Split the arrays to get X_train,y_train,X_test and y_test
            X_train,X_test,y_train,y_test=(
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1],
            )
        
            logging.info("Step 2-> Obtain X_train,X_test,y_train and y_test, completed.")

            # Step 3-> Train various models, and perform hyperparameter tuning and get the model report
            model_report=self.train_model(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)
            logging.info("Step 3-> Model training, along with hyperparameter tuning completed.")

            # Step 4-> Obtain the best model,according to recall score
            train_classification_metric,test_classification_metric,best_model=self.get_classification_metric(X_train=X_train,X_test=X_test,y_test=y_test,y_train=y_train,model_report=model_report)
            logging.info("Step 4-> Obtained the best model, according to recall.")

            # Step 5-> Now load the preprocessor object, along with final model
            

            preprocessor_object=load_pickle_object(self.data_transformation_artifact.preprocessor_object_path)

            fraud_detection_model=FraudDetectionModel(preprocessor=preprocessor_object,model=best_model)

            logging.info("Step 5-> Obtained the preprocessor object, and prepared the final model.")

            # Step 6-> Now save the final model along with preprocessor object
            model_dir_name=os.path.dirname(self.model_trainer_config.trained_model_dir)
            os.makedirs(model_dir_name)
            save_pickle_object(path=self.model_trainer_config.trained_model_dir,object=fraud_detection_model)
            logging.info("Step 6-> Saved the final model.")
            
            # finally return the model trainer artifacts
            model_trainer_artifacts=ModelTrainerArtifact(
                final_model_path=self.model_trainer_config.trained_model_dir,
                train_metric_artifact=train_classification_metric,
                test_metric_artifact=test_classification_metric
            )

            logging.info(f"Model trainer completed. Model trainer artifacts: {model_trainer_artifacts}")

            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e,sys)