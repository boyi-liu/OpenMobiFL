import pandas as pd
import os
import yaml
import xml.etree.ElementTree as ET
from datetime import datetime
from tqdm import tqdm
from utils.trajectory_utils import standard_trajectory_processing, save_npz, select_files, validate_config_exists

CABSPOTTING_DIR = "./traj_data/cabspotting"
OUTPUT_DIR = "./cabspotting"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_cabspotting_file(filepath):
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    lat, lon, occupancy, timestamp = parts
                    data.append([float(lon), float(lat), int(occupancy), int(timestamp)])
    except:
        return []
    return data

def process_cabspotting_data(raw_data, taxi_id):
    if not raw_data or len(raw_data) < 10:
        return None
    
    processed_data = []
    for lon, lat, occupancy, timestamp in raw_data:
        dt = datetime.fromtimestamp(timestamp)
        processed_data.append([taxi_id, dt, lon, lat, occupancy])
    
    return processed_data

def get_taxi_list():
    taxi_list = []
    cabs_file = os.path.join(CABSPOTTING_DIR, "_cabs.txt")
    
    try:
        with open(cabs_file, 'r') as f:
            content = f.read()
        root = ET.fromstring(f"<root>{content}</root>")
        
        for cab in root.findall('cab'):
            taxi_id = cab.get('id')
            updates = int(cab.get('updates', 0))
            if taxi_id and updates > 10:
                taxi_list.append((taxi_id, updates))
    except:
        for filename in os.listdir(CABSPOTTING_DIR):
            if filename.startswith('new_') and filename.endswith('.txt'):
                taxi_id = filename[4:-4]
                taxi_list.append((taxi_id, 1000))
    
    return taxi_list

def read_cabspotting_files():
    all_trajectories = []
    taxi_list = get_taxi_list()
    taxi_list.sort(key=lambda x: x[1], reverse=True)
    
    for taxi_id, _ in taxi_list:
        filename = f"new_{taxi_id}.txt"
        filepath = os.path.join(CABSPOTTING_DIR, filename)
        
        if not os.path.exists(filepath):
            continue
            
        raw_data = read_cabspotting_file(filepath)
        if raw_data:
            processed_data = process_cabspotting_data(raw_data, taxi_id)
            if processed_data and len(processed_data) >= 10:
                all_trajectories.append(processed_data)
    
    return all_trajectories

def generate_trajectory(cfg):
    validate_config_exists(cfg, 'client', 'selection_method', 'traj_number_min')
    
    all_trajectories = read_cabspotting_files()
    
    if not all_trajectories:
        return
    
    trajectory_data = []
    for i, trajectory in enumerate(all_trajectories):
        if len(trajectory) >= cfg['traj_number_min']:
            trajectory_data.append((i, len(trajectory), trajectory))
    
    selected = select_files(trajectory_data, cfg['client'], cfg['selection_method'])
    
    for i, (traj_id, traj_len, trajectory) in enumerate(tqdm(selected, desc="Cabspotting")):
        df_data = [[row[0], row[1], row[2], row[3]] for row in trajectory]
        df = pd.DataFrame(df_data, columns=['taxi_id', 'timestamp', 'longitude', 'latitude'])
        
        positions, timestamps = standard_trajectory_processing(df)
        taxi_id = str(df['taxi_id'].iloc[0])
        save_npz(f'{OUTPUT_DIR}/{i}.npz', positions, timestamps, 
                taxi_id=taxi_id, trip_id=traj_id, original_length=traj_len)

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    generate_trajectory(config)