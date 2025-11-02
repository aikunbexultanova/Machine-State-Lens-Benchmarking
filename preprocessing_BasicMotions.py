import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
import warnings
import sys
sys.path.insert(0, "tsc/")
# from normalize_z_score import process

# path = "./metadata/metadata_0-1.parquet" 
# df_parquet = pd.read_parquet(path)

warnings.filterwarnings(
    "ignore",
    message="numpy.core.numeric is deprecated",
    category=DeprecationWarning,
)

ACC_SLICE = slice(0, 3)   # acc x,y,z
GYR_SLICE = slice(3, 6)   # gyro x,y,z

def plot_sample(t, imu_data, label_name, sample_id):
    acc = imu_data[:, ACC_SLICE]
    gyro = imu_data[:, GYR_SLICE]
    
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ax[0].plot(t, acc[:, 0], label="acc_x")
    ax[0].plot(t, acc[:, 1], label="acc_y")
    ax[0].plot(t, acc[:, 2], label="acc_z")
    ax[0].set_ylabel("Acceleration")
    ax[0].legend(loc="upper right")
    ax[0].grid(True)
    
    ax[1].plot(t, gyro[:, 0], label="gyr_x")
    ax[1].plot(t, gyro[:, 1], label="gyr_y")
    ax[1].plot(t, gyro[:, 2], label="gyr_z")
    ax[1].set_ylabel('Gyroscope')
    ax[1].legend(loc="upper right")
    ax[1].grid(True)
    
    plt.suptitle(f"IMU data for {label_name} activity")
    plt.tight_layout()
    plt.show()
    path_to_dir = f"data_proccessed/figs/{label_name}/"
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)
    plt.savefig(f"{path_to_dir}/fig_{sample_id}.jpg")
    plt.close('all')


def plot_sample_df(df, label_name, sample_id, train_or_test):
    plt.figure(figsize = (12, 6))
    plt.plot(df["a_x"], label="a_x")
    plt.plot(df['a_y'], label="a_y")
    plt.plot(df['a_z'], label="a_z")
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.title(f"IMU data for {label_name} activity")
    path_to_dir = f"data_proccessed/figs_{train_or_test.lower()}/{label_name}/"
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)
    plt.savefig(f"{path_to_dir}/fig_{sample_id}.jpg")
    plt.close('all')


def prepare_sample(t, sample, label_name, sample_id, train_or_test, normalize=False):
    utc_constant = 1760103374
    df = pd.DataFrame(sample[:, :3], columns=['a_x', 'a_y', 'a_z'])
    # df = pd.DataFrame(sample[:, :3], schema=['a_x', 'a_y', 'a_z']) for polars 
    df['timestamp'] = utc_constant + t
    df.insert(0, 'timestamp', df.pop('timestamp'))
    
    columns_to_normalize = ["a_x", "a_y", "a_z"]
    # df_normalized = process(df, columns_to_normalize)
    
    if normalize: 
        df_normalized = pd.DataFrame()
        df_normalized[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].mean()) / df[columns_to_normalize].std()
        df_normalized['timestamp'] = utc_constant + t
        df_normalized.insert(0, 'timestamp', df_normalized.pop('timestamp'))
        df = df_normalized
    
    path_to_dir = f"data_proccessed/prepared_{train_or_test.lower()}/"
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)
    if train_or_test == "TEST":
        output_path= f'{path_to_dir}/{label_name}_{train_or_test.lower()}_imu_{sample_id}.csv'
    else:
        output_path = f'{path_to_dir}/{label_name}_imu_{sample_id}.csv'
        
    df.to_csv(output_path, index=False)
    return df
    

if __name__ == "__main__":
    for train_or_test in ["TRAIN", "TEST"]:
        with open(f'../data/Multivariate_ts/BasicMotions/BasicMotions_{train_or_test}.pickle', 'rb') as f:
            data = pickle.load(f)
            
        number_of_samples_in_activity = 10
        number_of_activities = 4

        imu_data = data['channel_data'][0]
        label_data = np.ravel(data['channel_data'][1])
        
        label_map = data['metadata']['label_map']
        
        label_map_inv = {value: key for key, value in label_map.items()}

        fs = 10.0
        T = imu_data.shape[1]
        t = np.arange(T) / fs

        for sample_id_in_activity in tqdm(range(10)):
            for i in range(4):
                indexes_i = np.where(label_data == i)
                
                df = prepare_sample(t, imu_data[indexes_i][sample_id_in_activity], 
                            str(label_map_inv[label_data[indexes_i][sample_id_in_activity]]), sample_id_in_activity, train_or_test, normalize=False)
                
                plot_sample_df(df, str(label_map_inv[label_data[indexes_i][sample_id_in_activity]]), sample_id_in_activity, train_or_test)
            
    # for sample_id_in_activity in range(number_of_samples_in_activity):
    #     for activity in range(number_of_activities):
    #         indexes_i = np.where(label_data == activity)
    #         prepare_sample(t, imu_data[indexes_i][sample_id_in_activity], 
    #                       str(label_map_inv[label_data[indexes_i][sample_id_in_activity]]), sample_id_in_activity)     