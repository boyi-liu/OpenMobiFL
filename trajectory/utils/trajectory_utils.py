import numpy as np
import pandas as pd
import random
from typing import List, Tuple

def process_timestamps_to_relative_minutes(timestamps: pd.Series) -> np.ndarray:

    first_time = timestamps.min()
    return np.array([(t - first_time).total_seconds() / 60.0 for t in timestamps])


def add_random_offset_minutes(timestamps: np.ndarray, min_offset: float = 0, max_offset: float = 10) -> np.ndarray:

    return timestamps + random.uniform(min_offset, max_offset)


def select_files(file_data: List[Tuple], count: int, method: str = "random") -> List[Tuple]:
    if method == "most_traj":
        file_data.sort(key=lambda x: x[1], reverse=True)
        return file_data[:count]
    else:  
        import random
        random.shuffle(file_data)
        return file_data[:count]


def standard_trajectory_processing(df: pd.DataFrame, 
                                 lon_col: str = 'longitude', 
                                 lat_col: str = 'latitude',
                                 time_col: str = 'timestamp') -> Tuple[np.ndarray, np.ndarray]:
 
    df = df.sort_values(time_col)

    positions = df[[lon_col, lat_col]].values

    timestamps = process_timestamps_to_relative_minutes(df[time_col])
    timestamps = add_random_offset_minutes(timestamps)
    
    return positions, timestamps


def save_npz(output_path: str, positions: np.ndarray, timestamps: np.ndarray, **kwargs) -> None:
    np.savez(output_path, position=positions, timestamp=timestamps, **kwargs)


def validate_config_exists(cfg: dict, *keys) -> None:
    for key in keys:
        if key not in cfg:
            raise KeyError(f"Missing required config key: {key}")