import pandas as pd
import os
import yaml
import ast
from datetime import datetime
from tqdm import tqdm
from utils.trajectory_utils import standard_trajectory_processing, save_npz, select_files, validate_config_exists

PORTO_FILE = "./traj_data/porto.csv"
OUTPUT_DIR = "./porto"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据限制配置 - 增加生成数量
MAX_TRAJECTORIES = 50000  # 增加到5万轨迹
MAX_CHUNKS = 1000  # 增加到1000个chunk  
CHUNK_SIZE = 5000  # 每个chunk处理5000行

def read_porto_polyline(polyline_str):

    coordinates = ast.literal_eval(polyline_str)
    return coordinates


def process_porto_row(row):

    taxi_id = int(row['TAXI_ID'])
    timestamp = int(row['TIMESTAMP'])  
    polyline = read_porto_polyline(row['POLYLINE'])
    
    if not polyline or len(polyline) < 2:
        return None
    
    data = []
    base_time = datetime.fromtimestamp(timestamp)
    
    for i, (lon, lat) in enumerate(polyline):
        current_time = base_time.timestamp() + i
        data.append([taxi_id, datetime.fromtimestamp(current_time), lon, lat])
    
    return data

def read_porto_file(filepath, max_trajectories=MAX_TRAJECTORIES, max_chunks=MAX_CHUNKS):

    all_trajectories = []
    chunk_count = 0
    
    for chunk in pd.read_csv(filepath, chunksize=CHUNK_SIZE):
        chunk_count += 1
        if chunk_count > max_chunks:
            break
            
        for _, row in chunk.iterrows():
            if len(all_trajectories) >= max_trajectories:
                return all_trajectories
                
            trajectory_data = process_porto_row(row)
            if trajectory_data and len(trajectory_data) >= 10:  # 至少10个点
                all_trajectories.append(trajectory_data)
                    
    return all_trajectories

def generate_trajectory(cfg):

    validate_config_exists(cfg, 'client', 'selection_method', 'traj_number_min')

    all_trajectories = read_porto_file(PORTO_FILE)

    trajectory_data = []
    for i, trajectory in enumerate(all_trajectories):
        if len(trajectory) >= cfg['traj_number_min']:
            trajectory_data.append((i, len(trajectory), trajectory))
    
    print(f"Collected {len(trajectory_data)} valid trajectories")

    selected = select_files(trajectory_data, cfg['client'], cfg['selection_method'])

    for i, (traj_id, traj_len, trajectory) in enumerate(tqdm(selected, desc="Porto")):
        df = pd.DataFrame(trajectory, columns=['taxi_id', 'timestamp', 'longitude', 'latitude'])
        positions, timestamps = standard_trajectory_processing(df)
        taxi_id = int(df['taxi_id'].iloc[0])
        save_npz(f'{OUTPUT_DIR}/{i}.npz', positions, timestamps, 
                taxi_id=taxi_id, trip_id=traj_id, original_length=traj_len)

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    generate_trajectory(config)