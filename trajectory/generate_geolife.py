import numpy as np
import pandas as pd
import os
import yaml
from datetime import datetime
from tqdm import tqdm
from utils.trajectory_utils import standard_trajectory_processing, save_npz, select_files, validate_config_exists

GEOLIFE_DIR = "./traj_data/Geolife/Data"
OUTPUT_DIR = "./geolife"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_geolife_plt(filepath):

    data = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        for line in lines[6:]:  # 跳过前6行头部
            parts = line.strip().split(',')
            if len(parts) >= 7:
                lat, lon, alt = float(parts[0]), float(parts[1]), float(parts[3])
                timestamp = datetime.strptime(f"{parts[5]} {parts[6]}", '%Y-%m-%d %H:%M:%S')
                data.append([lon, lat, alt, timestamp])
    except:
        return []
    return data

def generate_trajectory(cfg):

    validate_config_exists(cfg, 'client', 'selection_method', 'traj_number_min')

    file_data = []
    user_dirs = [d for d in os.listdir(GEOLIFE_DIR) if d.isdigit()]
    for user_dir in sorted(user_dirs):
        traj_dir = os.path.join(GEOLIFE_DIR, user_dir, 'Trajectory')
        if os.path.exists(traj_dir):
            for f in os.listdir(traj_dir):
                if f.endswith('.plt'):
                    data = read_geolife_plt(os.path.join(traj_dir, f))
                    if len(data) >= cfg['traj_number_min']:
                        file_data.append((os.path.join(traj_dir, f), len(data), user_dir, f))
    
    selected = select_files(file_data, cfg['client'], cfg['selection_method'])
    
    for i, (filepath, _, user_dir, plt_file) in enumerate(tqdm(selected, desc=f"GeoLife ({cfg['selection_method']})")):
        data = read_geolife_plt(filepath)
        df = pd.DataFrame(data, columns=['longitude', 'latitude', 'altitude', 'timestamp'])
        positions, timestamps = standard_trajectory_processing(df)
        save_npz(f'{OUTPUT_DIR}/{i}.npz', positions, timestamps, user_id=user_dir, original_file=plt_file)

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    generate_trajectory(config)