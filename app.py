import os
import sys
from flask import Flask,jsonify,render_template,redirect,request,url_for
from flask_cors import CORS
from Online_Payments.exception.exception import CustomException
from Online_Payments.pipeline.training_pipeline import TrainingPipeline
from Online_Payments.pipeline.prediction_pipeline import LoadData,PredictionPipeline


app=Flask(__name__)
CORS(app=app)

@app.route("/",methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/train.html",methods=["GET"])
def train_route():
    try:
        train_pipeline=TrainingPipeline()
        # start the training process
        train_pipeline.run_pipeline()
        return render_template("train.html")
    except Exception as e:
        raise CustomException(e,sys)
    
@app.route("/prediction.html")
def prediction_page():
    """Renders the transaction input form."""
    return render_template("prediction.html")

@app.route("/predict",methods=["GET","POST"])
def transaction_prediction():
    try:
        if request.method=="POST":
            input_data=LoadData(
                step=float(request.form.get("step")),
                payment_type=request.form.get("type"),
                transaction_amount=float(request.form.get("amount")),
                initial_balance_original=float(request.form.get("oldbalanceOrg")),
                final_balance_original=float(request.form.get("newbalanceOrig")),
                initial_balance_destination=float(request.form.get("oldbalanceDest")),
                final_balance_destination=float(request.form.get("newbalanceDest"))
            )

            input_df=input_data.get_data_as_frame()

            prediction_start=PredictionPipeline()
            result=prediction_start.predict_data(features=input_df)

            if bool(result):
                prediction="HIGH"
            else:
                prediction="LOW"

            return render_template("result.html",
                                   prediction=prediction,
                                   transaction_type=input_data.payment_type,
                                   amount=input_data.transaction_amount,
                                    time_step=input_data.step,
                                    oldbalanceOrg=input_data.initial_balance_original,
                                    newbalanceOrig=input_data.final_balance_original,
                                    oldbalanceDest=input_data.initial_balance_destination,
                                    newbalanceDest=input_data.final_balance_destination
                                   )
        
        elif request.method=="GET":
            return redirect(url_for('prediction_page'))
    except Exception as e:
        raise CustomException(e,sys)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080,debug=True)



    
