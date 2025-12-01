import numpy as np
import time
from pprint import pprint
import logging
import os
import sys
from datetime import datetime
import pickle
import pandas as pd
import yaml

# OMEGA MAX WINDOW SIZE
OMEGA_MAX_WINDOW_SIZE = 1024

def save_embedding_data(embedding_list, output_file):
    """
    Save embedding data to a pickle file for later classification tasks.
    
    Args:
        embedding_list: List of dictionaries containing embedding data
        output_dir: Directory to save the data
    """
    print(f"Save {output_file} to folder {os.path.dirname(output_file)}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if output_file.endswith(".pickle"):
        with open(output_file, 'wb') as f:
            pickle.dump(embedding_list, f)
    elif output_file.endswith(".csv"):
        df = pd.DataFrame(embedding_list)
        df = df[["index","file","read_index","window_size","embedding","labels"]]
        # Convert 'embedding' column into 768 separate columns
        embeddings_df = pd.DataFrame(df['embedding'].apply(np.squeeze).to_list())

        # Optionally, rename the new columns
        embeddings_df.columns = [f'embed_{i}' for i in range(768)]

        # Combine with original DataFrame (drop 'embedding' column if not needed)
        df_expanded = pd.concat([df.drop(columns='embedding'), embeddings_df], axis=1)

        df_expanded.to_csv(output_file, index=False)

    print(f"Saved {len(embedding_list)} embedding samples to {output_file}")
    return True

def load_embedding_data(data_path):
    """
    Load embedding data for classification tasks.
    
    Args:
        data_path: Path to the pickle file containing embedding data
    
    Returns:
        Dictionary with organized data for classification
    """
    with open(data_path, 'rb') as f:
        embedding_list = pickle.load(f)
    
    # Organize data for easy access
    organized_data = {
        'embeddings': np.array([item['embedding'] for item in embedding_list]),
        'indices': np.array([item['index'] for item in embedding_list]),
        'read_indices': np.array([item['read_index'] for item in embedding_list]),
        'window_sizes': np.array([item['window_size'] for item in embedding_list]),
        'sensor_timestamps': np.array([item['sensor_timestamp'] for item in embedding_list]),
        'timestamps': np.array([item['timestamp'] for item in embedding_list]),
        'variates': [item['variates'] for item in embedding_list],
        'labels': [item['labels'] for item in embedding_list],
        'files': [item['file'] for item in embedding_list]
    }
    
    return organized_data, embedding_list

def get_csv_chunks(file_path: str, data_columns: list, time_variate: str = None, label_columns: list = None, window_size: int = 1024, step_size: int = 1024):

    # assert that the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    df = pd.read_csv(file_path)
    logging.debug(f"CSV columns: {df.columns}")
    logging.debug(f"CSV length: {len(df)}")
    if not all(col in df.columns for col in data_columns):
        raise ValueError(f"Data columns {data_columns} not found in the CSV file.")
    
    cols_to_extract = data_columns.copy()
    if time_variate is not None:
        cols_to_extract += [time_variate]
    if label_columns is not None:
        cols_to_extract += label_columns
    logging.debug(f"cols to extract: {cols_to_extract}")

    # each datacolumn is treated as different variate
    chunks = []
    for i in range(0, len(df), step_size):
        window = df.iloc[i:i + window_size][cols_to_extract]
        chunks.append(window)
    return chunks

# timepoint mask utility function
def get_timepoint_mask(timeseries: list):
    """
    Returns a mask for the timepoints in the timeseries.
    The mask is a list of booleans indicating whether each timepoint is valid (True) or not (False).
    """
    
    if len(timeseries) < OMEGA_MAX_WINDOW_SIZE:
        timepoint_mask = np.ones(len(timeseries), dtype=bool)
        padded_data = np.pad(timeseries, (OMEGA_MAX_WINDOW_SIZE - len(timeseries), 0), mode='constant', constant_values=0)
        timepoint_mask = np.pad(timepoint_mask, (OMEGA_MAX_WINDOW_SIZE - len(timeseries), 0), mode='constant', constant_values=0)
    
    elif len(timeseries) > OMEGA_MAX_WINDOW_SIZE:
        timepoint_mask = np.ones(OMEGA_MAX_WINDOW_SIZE, dtype=bool)
        padded_data = timeseries[-OMEGA_MAX_WINDOW_SIZE:]
        timepoint_mask = np.pad(timepoint_mask, (0, OMEGA_MAX_WINDOW_SIZE - len(timeseries)), mode='constant', constant_values=0)
    
    else:
        timepoint_mask = np.ones(OMEGA_MAX_WINDOW_SIZE, dtype=bool)
        padded_data = timeseries
    
    return padded_data, timepoint_mask


def load_config(config_file):
    """
    Loads a config YAML file and return its contents as a dictionary.
    
    Args:
        config_file (str): Path to the YAML file.
    """
    try:
        with open(config_file, 'r') as file:
            data = yaml.safe_load(file)
            logging.info(f"Config: {data}")
            return data
    except FileNotFoundError:
        print(f"File not found: {config_file}")
    except yaml.YAMLError as e:
        print(f"Error parsing config: {e}")