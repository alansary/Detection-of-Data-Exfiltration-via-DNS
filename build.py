from src.data.make_dataset import make_dataset
from src.features.build_features import build_features
from src.models.train_model import execute_models
import logging
import os
import json

with open('config.json') as json_data_file:
    config = json.load(json_data_file)

logging.basicConfig(
    filename=os.getcwd()+'/'+config['log_path'],
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)

def builder():
    """
    Will be called to create all the models, after processing the dataset and building the features, which will then be accessed during usage of api.py
    """

    print("making dataset...")
    logger.info("making dataset...")
    df = make_dataset()

    print("build features...")
    logger.info("build features...")
    # processed_data_with_features = build_features(df, True)
    processed_data_with_features = df

    print("train model...")
    logger.info("train model...")
    execute_models(processed_data_with_features)

    print("done")
    logger.info("done")

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    builder()
