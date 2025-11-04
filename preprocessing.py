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

UTC_CONSTANT = 1760103374

warnings.filterwarnings(
    "ignore",
    message="numpy.core.numeric is deprecated",
    category=DeprecationWarning,
)


def prepare_sample(dataset_name, t, sample_sensor_data, file_idx, label_name):
    df = pd.DataFrame(sample_sensor_data[:, :3], columns=['a_1', 'a_2', 'a_3'])
    # df = pd.DataFrame(sample[:, :3], schema=['a_x', 'a_y', 'a_z']) for polars 
    df['timestamp'] = UTC_CONSTANT + t
    df.insert(0, 'timestamp', df.pop('timestamp'))
    
    path_to_dir = f"data_processed/{dataset_name}/prepared_{train_or_test.lower()}"
    os.makedirs(path_to_dir, exist_ok=True)
        
    output_path= f'{path_to_dir}/{label_name}_{train_or_test.lower()}_{file_idx}.csv'
        
    df.to_csv(output_path, index=False)
    return df
    

if __name__ == "__main__":
    dataset_name = "Heartbeat"
    fs = 1 / 0.0183
    
    for train_or_test in ["TRAIN", "TEST"]:
        per_label_counts = defaultdict(int)
        with open(f'../data/Multivariate_ts/{dataset_name}/{dataset_name}_{train_or_test}.pickle', 'rb') as f:
            data = pickle.load(f)
        print("Data shape: ", data["channel_data"][0].shape)
        print("Label shape", data["channel_data"][1].shape)
        
        sensor_data = data["channel_data"][0]
        label_data = data["channel_data"][1]
        
        label_map = data['metadata']['label_map']
        label_map_inv = {value: key for key, value in label_map.items()}
        
        T = sensor_data.shape[1]
        t = np.arange(T) / fs
        
        for sample_id in range(sensor_data.shape[0]):
            label_int = int(label_data[sample_id])
            label_str = str(label_map_inv[label_data[sample_id]])
            
            sample_id_in_activity = per_label_counts[label_str]
            prepare_sample(dataset_name, t, sensor_data[sample_id], sample_id_in_activity, label_str)
            
            per_label_counts[label_str] += 1
            
        print(train_or_test, per_label_counts)   