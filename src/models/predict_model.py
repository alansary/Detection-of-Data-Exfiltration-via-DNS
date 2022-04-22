from src.features.build_features import build_features
from src.models import model_utils
import mlflow
import json
import pickle as pickle
import pandas as pd

with open('config.json') as json_data_file:
    config = json.load(json_data_file)

# Our champion model is xgboost since it has the lowest False Negatives, therefore we will use it in prediction
with open(config['data']['xgboost']['filename'], 'rb') as pickle_file:
    model = pickle.load(pickle_file)

def predict_model(url):
    """
    Retuns a list containing the result of the prediction
    The list shows the url, model, precition score, and confidence score.

    Args:
        url: A string

    Retuns:
        A list with 4 keys: url, model, prediction_score, confidence_score
    """

    df = ['' + url]
    df = pd.DataFrame(df, columns=['url'])

    dataframe = build_features(df, False)
    domain = dataframe['url'].to_string().split('.')
    domain = domain[-2] if len(domain) >= 2 else domain
    features = dataframe.to_dict(orient="records")[0]

    # dataframe = model_utils.load_data(dataframe)
    dataframe = dataframe[list(config['data']['columns'].keys())]
    dataframe.drop(columns=['longest_word'], inplace=True)
    dataframe.drop(columns=['sld'], inplace=True)

    # Find the prediction score and condience score of the model
    pred_score = model.predict(dataframe).tolist()[0]
    confidence_score = model.predict_proba(dataframe)[0][pred_score]

    # result = {'url': url,
    #           'domain': domain,
    #           'model': 'XGBoost',
    #           'prediction_score': pred_score,
    #           'confidence_score': confidence_score,
    #           'features': features
    #           }
    result = features
    result['domain'] = result['url'].split('.')
    result['domain'] = result['domain'][-2] if len(result['domain']) >=2 else len(result['domain'])
    result['predicted_label'] = pred_score
    result['score'] = confidence_score
    # print(result)

    # metric_logger(url, result)

    return result


def metric_logger(url, data):
    # with mlflow.start_run():
    #     mlflow.log_param("url", url)
    #     mlflow.log_param("Model", data['model'])
    #     mlflow.log_metric("Prediction Score", data['prediction_score'])
    #     mlflow.log_metric("Confidence Score", data['confidence_score'])

    with mlflow.start_run():
        mlflow.log_param("url", url)
        mlflow.log_param("Model", "XGBoost")
        mlflow.log_metric("Prediction Score", data['predicted_label'])
        mlflow.log_metric("Confidence Score", data['score'])
