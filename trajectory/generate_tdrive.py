import numpy as np
import pandas as pd
import os
import yaml
from datetime import datetime
from tqdm import tqdm
from utils.trajectory_utils import standard_trajectory_processing, save_npz, select_files, validate_config_exists

TDRIVE_DIR = "./traj_data/T-Drive"
OUTPUT_DIR = "./tdrive"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_tdrive_file(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 4:
                taxi_id = int(parts[0])
                timestamp = datetime.strptime(parts[1], '%Y-%m-%d %H:%M:%S')
                longitude = float(parts[2])
                latitude = float(parts[3])
                data.append([taxi_id, timestamp, longitude, latitude])
    return data

def generate_trajectory(cfg):
    validate_config_exists(cfg, 'client', 'selection_method', 'traj_number_min')

    files = [f for f in os.listdir(TDRIVE_DIR) if f.endswith('.txt')]
    file_data = []
    for f in files:
        data = read_tdrive_file(os.path.join(TDRIVE_DIR, f))
        if len(data) >= cfg['traj_number_min']:
            file_data.append((f, len(data)))

    selected = select_files(file_data, cfg['client'], cfg['selection_method'])

    for i, (filename, _) in enumerate(tqdm(selected, desc=f"T-Drive ({cfg['selection_method']})")):
        data = read_tdrive_file(os.path.join(TDRIVE_DIR, filename))
        df = pd.DataFrame(data, columns=['taxi_id', 'timestamp', 'longitude', 'latitude'])
        positions, timestamps = standard_trajectory_processing(df)
        save_npz(f'{OUTPUT_DIR}/{i}.npz', positions, timestamps, taxi_id=int(df['taxi_id'].iloc[0]))

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    generate_trajectory(config)