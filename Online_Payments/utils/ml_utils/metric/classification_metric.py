import os
import sys
from Online_Payments.exception.exception import CustomException
from Online_Payments.logger.logger import logging
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from Online_Payments.entity.artifact_enity import ClassificationMetricArtifact


def get_classification_metrics(y_true,y_pred,y_score):
    try:
        # get the metrics
        acc_score=accuracy_score(y_true=y_true,y_pred=y_pred)
        precision=precision_score(y_true=y_true,y_pred=y_pred)
        recall=recall_score(y_true=y_true,y_pred=y_pred)
        f1=f1_score(y_true=y_true,y_pred=y_pred)
        roc=roc_auc_score(y_score=y_score,y_true=y_true)


        classification_metric_artifact=ClassificationMetricArtifact(
            accuracy_score=acc_score,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_score=roc
        )

        return classification_metric_artifact

    except Exception as e:
        raise CustomException(e,sys)