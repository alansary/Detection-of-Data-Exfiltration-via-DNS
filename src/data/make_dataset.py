import logging
import pandas as pd
import json
from dotenv import find_dotenv, load_dotenv
from pandas_profiling import ProfileReport
import numpy as np

load_dotenv(find_dotenv())

def make_dataset():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
        Returns: dataframe
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')
    with open('config.json') as json_data_file:
        config = json.load(json_data_file)

    datasets = config['data']['datasets']
    final_file = []  # Combine all the dataframes into an array

    for key in datasets:
        dataset_details = config['data']['datasets'][key]
        path = dataset_details['path']
        csv_data = pd.read_csv(path)
        final_file.append(csv_data)  # Add the processed dataframe to the array

    result = pd.concat(final_file, sort=False)  # combine the dataframes into one dataframes

    # Exclude timestamp column
    result.drop(columns=['timestamp'], inplace=True)

    # # Drop object columns; namely, longest_word and sld
    # result = result.select_dtypes(exclude=['object'])

    # Replace infinity values by the mean
    result.replace([np.inf, -np.inf], np.nan, inplace=True) # Replace by NaN
    result = result.fillna(result.mean()) # Replace NaN by mean

    result.to_csv(config['data']['data_loc'], index=False)  # Save the processed file into the 'data' directory

    logger.info('Finished making the final data set')
    logger.info('Preparing a report for the data set.')

    # Create a report from the processed data
    profile = ProfileReport(result, minimal=True, title='Preprocessed Data Report',
                            html={'style': {'full_width': True}})
    profile.to_file(output_file=config['data']['preprocessed_data_report_loc'])
    logger.info('The report is ready. It can be found at data/processed/preprocessed_data_report.html')

    return result
