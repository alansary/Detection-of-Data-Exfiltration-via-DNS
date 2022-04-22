import pandas as pd
from kafka import KafkaProducer
from datetime import datetime
import json
import logging
import time
import os

def instantiate_kafka_producer():
    producer = KafkaProducer(
        bootstrap_servers=config['kafka']['bootstrap_servers'],
        api_version=tuple(config['kafka']['api_version']),
    )
    
    if producer.bootstrap_connected():
        print(f"Successfully connected to bootstrap server")
    else:
        print("Couldn't connect to bootstrap server.")
        
    return producer

def produce_message(producer_instance, topic, message):
    producer_instance.send(topic, message)
    producer_instance.flush()
    return

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

    producer = instantiate_kafka_producer()
    
    logger.info("Sending urls to Kafka...")
    start = time.time()
    print("Ingesting the Data in Batches of 10000")
    
    for batch_id, data_batch in enumerate(pd.read_csv(config['kafka']['producer']['bulk_load_file_path'], chunksize=10000)):
        start_time = datetime.now()
        print(f"Ingesting Batch: {batch_id}")

        for idx, instance in data_batch.iterrows():
            domain = bytes(instance.domain, encoding="utf-8")
            produce_message(producer_instance=producer, topic=config['kafka']['producer']['kafka_topic'], message=domain)

        end_time = datetime.now()
        print(f"Batch {batch_id} took {end_time-start_time} time for ingesting data")

    logger.info("Finished sending urls to kafka.")
    end = time.time()
    logger.info("Ellapsed time: " + str(end - start) + " for sending urls to kafka topic")
    print("Ingestion Completed")