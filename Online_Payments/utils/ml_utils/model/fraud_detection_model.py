import os
import sys
from Online_Payments.exception.exception import CustomException



class FraudDetectionModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor=preprocessor
            self.model=model
        except Exception as e:
            raise CustomException(e,sys)
        
    def predict(self,x):
        try:
            x_transformed=self.preprocessor.transform(x)
            y_hat=self.model.predict(x_transformed)

            return y_hat
        except Exception as e:
            raise CustomException(e,sys)
        
    def predict_fraud_probability(self,x):
        try:
            x_transformed=self.preprocessor.transform(x)
            y_hat_probability=self.model.predict_proba(x_transformed)
            return y_hat_probability
        except Exception as e:
            raise CustomException(e,sys)