import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
from collections import defaultdict
import sys
sys.path.insert(0, "tsc_1/")
import warnings
import argparse
from eval_script import save_config
import math

UTC_CONSTANT = 1760103374

warnings.filterwarnings(
    "ignore",
    message="numpy.core.numeric is deprecated",
    category=DeprecationWarning,
)
mutivariate_list = []


def prepare_sample(dataset_output_dir, dataset_name, t, sample_sensor_data, file_idx, label_name, dataset_type, train_or_test):
    if dataset_type == "Univariate":
        df = pd.DataFrame(sample_sensor_data[:, :], columns=['a_1'])
    else:    
        df = pd.DataFrame(sample_sensor_data[:, :3], columns=['a_1', 'a_2', 'a_3'])
    df['timestamp'] = UTC_CONSTANT + t
    df.insert(0, 'timestamp', df.pop('timestamp'))
    
    path_to_dir = f"{dataset_output_dir}/{dataset_name}/prepared_{train_or_test.lower()}"
    os.makedirs(path_to_dir, exist_ok=True)
        
    output_path= f'{path_to_dir}/{label_name}_{train_or_test.lower()}_{file_idx}.csv'
        
    df.to_csv(output_path, index=False)
    return df


def choose_window_and_step(L, overlap=0.875):
    # window = largest power of 2 <= L
    window = 2 ** int(math.floor(math.log2(L)))

    # step = window * (1 - overlap)
    step = max(int(round(window * (1 - overlap))), 1)

    return window, step


def generate_config_file(dataset_name, train_file_idx_max, test_file_idx_max, labels, train_data_shape, test_data_shape, file_path):
    config = {}
    config["dataset_name"] = dataset_name
    config["labels"] = labels
    config["N_max"] = train_file_idx_max
    config["test_file_idx_max"] = test_file_idx_max
    config["timestamp_column"] = "timestamp"
    if train_data_shape[2] == 1:
        config["data_columns"] = ["a_1"]
    else:
        config["data_columns"] = ["a_1", "a_2", "a_3"]    
    config["train_data_shape"] = train_data_shape 
    config["test_data_shape"] = test_data_shape 
    window, step = choose_window_and_step(train_data_shape[1])
    config["window_size"] = window
    config["step_size"] = step
    config["normalize_input"] = False
    config["prenormalize"] = False
    
    save_config(config, file_path)
    
    return config
    

def preprocess_dataset(args, generate_config=True):
    for train_or_test in ["TRAIN", "TEST"]:
        per_label_counts = defaultdict(int)
        
        with open(f'./data/{args.dataset_type}_ts/{args.dataset_name}/{args.dataset_name}_{train_or_test}.pickle', 'rb') as f:
            data = pickle.load(f)
        print(f"{train_or_test} data shape: ", data["channel_data"][0].shape)
        print(f"{train_or_test} label shape:", data["channel_data"][1].shape)
        
        sensor_data = data["channel_data"][0]
        label_data = data["channel_data"][1]
        
        label_map = data['metadata']['label_map']
        label_map_inv = {value: key for key, value in label_map.items()}
        labels = [str(k) for k in label_map.keys()]
        print(f"{train_or_test} labels:", labels)
        
        T = sensor_data.shape[1]
        t = np.arange(T) / args.frequency
        
        for sample_id in range(sensor_data.shape[0]):
            # label_int = int(label_data[sample_id])
            label_str = str(label_map_inv[label_data[sample_id]])
                        
            sample_id_in_activity = per_label_counts[label_str]
            prepare_sample(args.dataset_output_dir, args.dataset_name, t, sensor_data[sample_id], sample_id_in_activity, label_str, args.dataset_type, train_or_test)
            per_label_counts[label_str] += 1

        if train_or_test == "TRAIN":
            train_file_idx_max = min(per_label_counts.values())
            train_data_shape = data["channel_data"][0].shape 
        else:
            test_file_idx_max = min(per_label_counts.values())
            test_data_shape = data["channel_data"][0].shape
            
    config = None
    
    if generate_config:
        config_file_path = f"tsc_1/{args.base_config_dir}/{args.dataset_name}_config.json"
        os.makedirs(f"tsc_1/{args.base_config_dir}", exist_ok=True)
        config = generate_config_file(args.dataset_name, train_file_idx_max, test_file_idx_max, labels, train_data_shape, test_data_shape, config_file_path)
        # print(config["data_columns"])
    return config
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="ACSF1", type=str)
    parser.add_argument("--dataset_type", default="Univariate", type=str)
    parser.add_argument("--dataset_output_dir", default="data_processed", type=str)
    parser.add_argument("--base_config_dir", default="data_configs", type=str)
    parser.add_argument("--frequency", default=0.1, type=float)
    args = parser.parse_args()
    
    # preprocess_dataset(args, generate_config=True)
    dataset_names = (os.listdir(f'data/{args.dataset_type}_ts'))
    for dataset_name in tqdm(dataset_names):
        args.dataset_name = dataset_name
        preprocess_dataset(args, generate_config=True)