import os
import sys
import pandas as pd
from Online_Payments.exception.exception import CustomException
from Online_Payments.utils.main_utils.utils import load_pickle_object
from Online_Payments.utils.ml_utils.model.fraud_detection_model import FraudDetectionModel
from Online_Payments.logger.logger import logging


class LoadData:
    def __init__(self,step,payment_type,transaction_amount,initial_balance_original,final_balance_original,initial_balance_destination,final_balance_destination):
        try:
            self.step=step
            self.payment_type=payment_type
            self.transaction_amount=transaction_amount
            self.initial_balance_original=initial_balance_original
            self.final_balance_original=final_balance_original
            self.initial_balance_destination=initial_balance_destination
            self.final_balance_destination=final_balance_destination
        except Exception as e:
            raise CustomException(e,sys)
        
    def get_data_as_frame(self):
        try:
            data_dict={
                "step":[self.step],
                "type":[self.payment_type],
                "amount":[self.transaction_amount],
                "oldbalanceOrg":[self.initial_balance_original],
                "newbalanceOrig":[self.final_balance_original],
                "oldbalanceDest":[self.initial_balance_destination],
                "newbalanceDest":[self.final_balance_destination]
            }
            
            logging.info("Obtained the features as dataframe.")
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e,sys)
        

class PredictionPipeline:
    def __init__(self):
        pass

    def predict_data(self,features):
        try:
            model_path="final_model/model.pkl"
            preprocessor_path="final_model/preprocessor.pkl"

            model=load_pickle_object(path=model_path)
            preprocessor=load_pickle_object(path=preprocessor_path)

            # now load the fraud detection model
            logging.info("Loaded the fraud detection model.")
            fraud_detection_model=FraudDetectionModel(preprocessor=preprocessor,model=model)
            # and pass my set of features
            y_predicted=fraud_detection_model.predict(x=features)
            y_predicted_probability=fraud_detection_model.predict_fraud_probability(x=features)
            logging.info(f"Predicted value for the new input features: {y_predicted} and fraud risk score: {y_predicted_probability}")
            return y_predicted,y_predicted_probability[:,1]
        except Exception as e:
            raise CustomException(e,sys)