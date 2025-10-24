import numpy as np
import pandas as pd
import os
import yaml

from datetime import datetime
from tqdm import tqdm

TDRIVE_DIR = "trajectory/traj_data/T-Drive"
OUTPUT_DIR = "trajectory/traj_tdrive"
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
    client_num = cfg['client']
    selection_method = cfg['selection_method']
    traj_number_min = cfg['traj_number_min']
    
    txt_files = [f for f in os.listdir(TDRIVE_DIR) if f.endswith('.txt')]
    
    # Collect all files with their trajectory counts
    file_data = []
    for txt_file in txt_files:
        filepath = os.path.join(TDRIVE_DIR, txt_file)
        data = read_tdrive_file(filepath)
        if len(data) >= traj_number_min:  # Only include files that meet minimum requirement
            file_data.append((txt_file, len(data)))
    
    # Sort files based on selection method
    if selection_method == 'most_traj':
        # Sort by trajectory count (descending) and select top clients
        file_data.sort(key=lambda x: x[1], reverse=True)
        selected_files = [item[0] for item in file_data[:client_num]]
    else:  # random selection
        # Randomly shuffle and select clients
        import random
        random.shuffle(file_data)
        selected_files = [item[0] for item in file_data[:client_num]]
    
    for i, txt_file in enumerate(tqdm(selected_files, desc=f"Processing T-Drive files (method: {selection_method}, limit: {client_num})")):
        filepath = os.path.join(TDRIVE_DIR, txt_file)
        data = read_tdrive_file(filepath)
        
        if not data:
            continue

        df = pd.DataFrame(data, columns=['taxi_id', 'timestamp', 'longitude', 'latitude'])

        df = df.sort_values('timestamp')

        positions = df[['longitude', 'latitude']].values
        
        # Convert timestamps to relative minutes from the first timestamp in this file
        # This makes each trajectory start from time 0, similar to GeoLife logic
        first_timestamp = df['timestamp'].min()
        timestamps = [(t - first_timestamp).total_seconds() / 60.0 for t in df['timestamp']]  
        timestamps = np.array(timestamps)
        
        # Add random offset (0-10 minutes) to avoid all trajectories starting from 0
        import random
        random_offset = random.uniform(0, 10)
        timestamps = timestamps + random_offset

        np.savez(
            f'{OUTPUT_DIR}/{i}.npz',
            position=positions,
            timestamp=timestamps,
        )

if __name__ == "__main__":
    with open('trajectory/config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    generate_trajectory(config)