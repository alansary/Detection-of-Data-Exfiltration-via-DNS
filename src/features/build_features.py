import json
import logging
import time
import src.features.url_utils as url_utils
from dotenv import find_dotenv, load_dotenv

with open('config.json') as json_data_file:
    dataset = json.load(json_data_file)

load_dotenv(find_dotenv())

final_dataset_path = dataset['data']['processed-output'] + 'final_processed_with_features.csv'

def build_features(df, csv):
    """
    returns a dataframe with all the columns created and populated

    Args:
        df: A dataframe
        csv: A boolean. If True, the df data will be written to a csv file

    Returns:
        A dataframe.
    """
    logger = logging.getLogger(__name__)

    # Ellapsed time check for efficiency
    start = time.time()

    # URL Stateless Features
    logger.info("beginning Stateless Features")
        
    df["FQDN_count"] = df["url"].map(lambda x: url_utils.get_FQDN_count(x))
    df["subdomain_length"] = df["url"].map(lambda x: url_utils.get_subdomain_length(x))
    df["upper"] = df["url"].map(lambda x: url_utils.get_uppercase_characters_count(x))
    df["lower"] = df["url"].map(lambda x: url_utils.get_lowercase_characters_count(x))
    df["numeric"] = df["url"].map(lambda x: url_utils.get_digit_count(x))
    df["entropy"] = df["url"].map(lambda x: url_utils.entropy(x))
    df["special"] = df["url"].map(lambda x: url_utils.get_special_characters_count(x))
    df["labels"] = df["url"].map(lambda x: url_utils.get_url_labels_count(x))
    df["labels_max"] = df["url"].map(lambda x: url_utils.get_max_label_length(x))
    df["labels_average"] = df["url"].map(lambda x: url_utils.get_avg_label_length(x))
    df["longest_word"] = df["url"].map(lambda x: url_utils.get_longest_word(x))
    df["sld"] = df["url"].map(lambda x: url_utils.get_sld_length(x))
    df["len"] = df["url"].map(lambda x: url_utils.get_length_of_domain_and_subdomain_length(x))
    df["subdomain"] = df["url"].map(lambda x: url_utils.url_has_subdomain(x))

    end = time.time()
    logger.info("Ellapsed time: " + str(end - start) + " for features")

    if csv:
        logger.info("Sending to CSV...")
        df.to_csv(final_dataset_path, index=False)

    return df
