# Trajectory Dataset Information

This system supports four major trajectory datasets, covering taxi, pedestrian, and multi-modal transportation data.

## 🚕 Taxi Trajectory Datasets

### 1. T-Drive (Beijing Taxis)
- **Dataset**: Microsoft T-Drive
- **Download**: https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/
- **Scale**: 10,357 taxis, 1 week of data (February 2-8, 2008)
- **Coverage**: Beijing, China
- **Sampling Rate**: 1-5 seconds
- **Records**: ~15 million GPS points
- **File Format**: One text file per taxi
- **Format**: `taxi_id, timestamp(YYYY-MM-DD HH:MM:SS), longitude, latitude`
- **Processing Script**: `generate_tdrive.py`

### 2. Cabspotting (San Francisco Taxis)
- **Dataset**: Cabspotting Project (Exploratorium)
- **Download**: http://www.lac.inpe.br/~rafael.santos/Docs/CAP394/Proj_Cabspotting.html
- **Scale**: 536 taxis, 30 days of data (May-June 2008)
- **Coverage**: San Francisco Bay Area, USA
- **Sampling Rate**: Average <60 seconds
- **Records**: ~11 million GPS points
- **File Format**: One text file per taxi
- **Format**: `latitude longitude occupancy_status(0/1) timestamp(UNIX)`
- **Processing Script**: `generate_cabspotting.py`

### 3. Porto Taxi (Porto Taxis)
- **Dataset**: Taxi Service Trajectory - ECML/PKDD 2015
- **Download**: https://www.kaggle.com/datasets/crailtap/taxi-trajectory
- **Scale**: 442 taxis, 1.7 million trips, 1 year of data (2013-2014)
- **Coverage**: Porto, Portugal
- **Sampling Rate**: 15 seconds
- **Records**: ~170 million GPS points
- **File Format**: Single CSV file
- **Format**: `TRIP_ID, CALL_TYPE, TAXI_ID, TIMESTAMP, POLYLINE([[lon,lat],...])`
- **Processing Script**: `generate_porto.py`

## 🚶 Pedestrian and Multi-modal Trajectory Datasets

### 4. GeoLife GPS Trajectories
- **Dataset**: Microsoft GeoLife
- **Download**: https://www.microsoft.com/en-us/download/details.aspx?id=52367
- **Scale**: 182 users, 17,621 trajectories, 2007-2012
- **Coverage**: Beijing, Seattle, and other cities
- **Sampling Rate**: 1-5 seconds
- **Records**: ~24 million GPS points
- **File Format**: One directory per user, multiple PLT files
- **Format**: PLT format (latitude, longitude, altitude, date, time)
- **Processing Script**: `generate_geolife.py`

## 📊 Dataset Statistics Comparison

| Dataset | Vehicles/Users | Time Span | Trajectory Points | Sampling Interval | Coverage Area |
|---------|----------------|-----------|-------------------|-------------------|---------------|
| T-Drive | 10,357 taxis | 1 week | ~15 million | 1-5 seconds | Beijing |
| Cabspotting | 536 taxis | 30 days | ~11 million | <60 seconds | San Francisco |
| Porto | 442 taxis | 1 year | ~170 million | 15 seconds | Porto |
| GeoLife | 182 users | 5 years | ~24 million | 1-5 seconds | Beijing/Seattle |

## 🔄 Unified Processing Pipeline

### Standard Output Format
All datasets are processed and saved in NPZ format, containing:
- `position`: [longitude, latitude] coordinate array
- `timestamp`: relative timestamps (minutes)
- `taxi_id`/`user_id`: identifiers
- `trip_id`: internal trajectory ID
- `original_length`: original trajectory length

### Processing Script Parameters
All generation scripts support the same configuration parameters:
```yaml
client: 100                    # Number of clients to select
selection_method: "random"     # Selection method: random/most_traj
traj_number_min: 100           # Minimum trajectory points required
```

## 📚 References

1. **T-Drive**: Yuan, J., Zheng, Y., Xie, X., & Sun, G. (2010). "Driving with knowledge from the physical world." *Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining*.

2. **Cabspotting**: Piorkowski, M., Sarafijanovoc-Djukic, N., & Grossglauser, M. (2009). "A Parsimonious Model of Mobile Partitioned Networks with Clustering." *COMSNETS 2009*.

3. **Porto Taxi**: Moreira-Matias, L., Gama, J., Ferreira, M., Mendes-Moreira, J., & Damas, L. (2013). "Predicting taxi–passenger demand using streaming data." *IEEE Transactions on Knowledge and Data Engineering*, 26(3), 681-694.

4. **GeoLife**: Zheng, Y., Zhang, L., Xie, X., & Ma, W. Y. (2009). "Mining interesting locations and travel sequences from GPS trajectories." *Proceedings of the 18th international conference on World wide web*.

## 🔗 Quick Start

### Generate Local Datasets
```bash
# Generate T-Drive dataset
python generate_tdrive.py

# Generate Cabspotting dataset  
python generate_cabspotting.py

# Generate Porto dataset
python generate_porto.py

# Generate GeoLife dataset
python generate_geolife.py
```

### Check Generated Results
```python
import numpy as np

# Load generated trajectory data
data = np.load('tdrive/0.npz')
print(f"Trajectory points: {len(data['position'])}")
print(f"Time range: {data['timestamp'].min():.1f} - {data['timestamp'].max():.1f} minutes")
print(f"Geographic range: {data['position'][:,0].min():.3f}°E - {data['position'][:,0].max():.3f}°E")
print(f"Geographic range: {data['position'][:,1].min():.3f}°N - {data['position'][:,1].max():.3f}°N")
```