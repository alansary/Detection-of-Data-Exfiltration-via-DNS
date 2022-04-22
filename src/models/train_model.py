import src.models.model_utils as models
import pandas as pd
import json

def execute_models(processed_data_with_features):
    """ Execute one or more models from model_utils
        
        Args: processed_data_with_features is the resulting data after discretization
        
        Returns: models metrics
    """

    y = models.load_labels_for_training(processed_data_with_features)
    x = models.load_data(processed_data_with_features)

    with open('config.json') as json_data_file:
        config = json.load(json_data_file)

    rf = models.random_forest(x, y)
    models.mlflow_logger(rf)

    dt = models.decision_tree(x, y)
    models.mlflow_logger(dt)

    xgboost = models.xgboost(x, y)
    models.mlflow_logger(xgboost)

    # mlp = models.mlp(x, y)
    # models.mlflow_logger(mlp)

    print("returning data")
    # data_to_be_logged = {'Random Forest': rf, 'Decision Tree': dt, 'XGBoost': xgboost, 'MLP': mlp}
    data_to_be_logged = {'Random Forest': rf, 'Decision Tree': dt, 'XGBoost': xgboost}

    return data_to_be_logged

