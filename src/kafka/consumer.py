import sys
import time
import json
from kafka import KafkaConsumer
from kafka import KafkaProducer
from src.models.predict_model import predict_model
import logging
import os
import pandas as pd

def send_to_kafka(result):
    """
    sends the url to the given kafka topic.

    Arguments:
    result: list

    Returns: None
    """
    logger = logging.getLogger(__name__)
    producer = KafkaProducer(bootstrap_servers=config['kafka']['bootstrap_servers'],
                            api_version=tuple(config['kafka']['api_version']))
    logger.info("Sending url to Kafka...")
    producer.send(config['kafka']['consumer']['kafka_topic_to_send_to'], json.dumps(result).encode('utf-8'))


def url_bulk_consume():
    """
    Consumes urls from configured kafka topic , runs the predictions on them,
    sends the malicious urls to the second configured kafka topic

    Returns: None
    """
    logger = logging.getLogger(__name__)
    consumer = KafkaConsumer(config['kafka']['consumer']['kafka_topic_to_read_from'],
                             bootstrap_servers=config['kafka']['bootstrap_servers'],
                             api_version=tuple(config['kafka']['api_version'])
                             )
    start = time.time()
    logger.info("Reading data ...")

    output = pd.DataFrame(columns=list(config['data']['columns'].keys()) + ['domain', 'predicted_label', 'score'])
    # output.to_csv(config['data']['output'], index=False) # for inner-loop live append
    dataframes = []
    dataframes.append(output)

    counter = 1
    for message in consumer:
        url = message.value.decode("utf-8")
        logger.info(url)
        result = predict_model(url)
        result.pop('url')
        # prediction_score = result['prediction_score']
        # confidence_score = result['confidence_score']
        prediction_score = result['predicted_label']
        confidence_score = result['score']

        logger.info("prediction: " + str(prediction_score))

        # # Send malicious urls to another Kafka topic
        # if (prediction_score == config['kafka']['predicted_label'] and
        #         confidence_score >= config['kafka']['confidence_score']):
        #     logger.info("Found malicious url")
        #     send_to_kafka(result)

        # print(result)
        result = pd.DataFrame([result])
        # result.to_csv(config['data']['output'], mode='a', header=False, index=False)
        dataframes.append(result)

        if counter == 100000:
            break
        print(counter)
        counter += 1
    
    dataframes = pd.concat(dataframes)
    dataframes.to_csv(config['data']['output'], index=False)
    end = time.time()
    logger.info("Ellapsed time: " + str(end - start) + " for consuming and predicting urls from kafka topic")


if __name__ == "__main__":
    with open('config.json') as json_data_file:
        config = json.load(json_data_file)

    logging.basicConfig(
        filename=os.getcwd()+'/'+config['log_path'],
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    if (config['kafka']['consumer']['kafka_topic_to_read_from'] is None):
        logger.critical("Cannot process. Please make sure the kafka topic to read from is set in the configuration file, "
                        "config.py")
        raise ValueError("Cannot process. Please make sure the kafka topic to read from is set in the configuration file, "
                         "config.py")

    url_bulk_consume()
