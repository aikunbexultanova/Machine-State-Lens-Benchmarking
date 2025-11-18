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


def preprocess_dataset(args):
    for train_or_test in ["TRAIN", "TEST"]:
        per_label_counts = defaultdict(int)
        with open(f'../data/{args.dataset_type}_ts/{args.dataset_name}/{args.dataset_name}_{train_or_test}.pickle', 'rb') as f:
            data = pickle.load(f)
        print(f"{train_or_test} data shape: ", data["channel_data"][0].shape)
        print(f"{train_or_test} label shape", data["channel_data"][1].shape)
        
        sensor_data = data["channel_data"][0]
        label_data = data["channel_data"][1]
        
        label_map = data['metadata']['label_map']
        label_map_inv = {value: key for key, value in label_map.items()}
        
        T = sensor_data.shape[1]
        t = np.arange(T) / args.frequency
        
        for sample_id in range(sensor_data.shape[0]):
            # label_int = int(label_data[sample_id])
            label_str = str(label_map_inv[label_data[sample_id]])
            
            sample_id_in_activity = per_label_counts[label_str]
            prepare_sample(args.dataset_output_dir, args.dataset_name, t, sensor_data[sample_id], sample_id_in_activity, label_str, args.dataset_type, train_or_test)
            
            per_label_counts[label_str] += 1  
        print(train_or_test, min(per_label_counts.values()))
        print(per_label_counts)    
    return    
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="ACSF1", type=str)
    parser.add_argument("--dataset_type", default="Univariate", type=str)
    parser.add_argument("--dataset_output_dir", default="data_processed", type=str)
    parser.add_argument("--frequency", default=0.1, type=float)
    args = parser.parse_args()
    
    preprocess_dataset(args)