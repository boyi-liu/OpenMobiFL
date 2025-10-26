# 轨迹数据集信息

本系统支持以下四个主要的轨迹数据集，涵盖出租车、行人和多模式交通数据。

## 🚕 出租车轨迹数据集

### 1. T-Drive (北京出租车)
- **数据集**: Microsoft T-Drive
- **下载地址**: https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/
- **数据规模**: 10,357辆出租车，1周数据 (2008年2月2-8日)
- **覆盖区域**: 中国北京市
- **采样频率**: 1-5秒
- **记录数**: 约1500万GPS点
- **文件格式**: 每辆出租车一个文本文件
- **格式**: `出租车ID, 时间戳(YYYY-MM-DD HH:MM:SS), 经度, 纬度`
- **处理脚本**: `generate_tdrive.py`

### 2. Cabspotting (旧金山出租车)
- **数据集**: Cabspotting Project (Exploratorium)
- **下载地址**: http://www.lac.inpe.br/~rafael.santos/Docs/CAP394/Proj_Cabspotting.html
- **数据规模**: 536辆出租车，30天数据 (2008年5-6月)
- **覆盖区域**: 美国旧金山湾区
- **采样频率**: 平均<60秒
- **记录数**: 约1100万GPS点
- **文件格式**: 每辆出租车一个文本文件
- **格式**: `纬度 经度 载客状态(0/1) 时间戳(UNIX)`
- **处理脚本**: `generate_cabspotting.py`

### 3. Porto Taxi (波尔图出租车)
- **数据集**: Taxi Service Trajectory - ECML/PKDD 2015
- **下载地址**: https://www.kaggle.com/datasets/crailtap/taxi-trajectory
- **数据规模**: 442辆出租车，170万条行程，1年数据 (2013-2014)
- **覆盖区域**: 葡萄牙波尔图市
- **采样频率**: 15秒
- **记录数**: 约1.7亿GPS点
- **文件格式**: 单个CSV文件
- **格式**: `TRIP_ID, CALL_TYPE, TAXI_ID, TIMESTAMP, POLYLINE([[lon,lat],...])`
- **处理脚本**: `generate_porto.py`

## 🚶 行人及多模式轨迹数据集

### 4. GeoLife GPS Trajectories
- **数据集**: Microsoft GeoLife
- **下载地址**: https://www.microsoft.com/en-us/download/details.aspx?id=52367
- **数据规模**: 182位用户，17,621条轨迹，2007-2012年
- **覆盖区域**: 北京、西雅图等城市
- **采样频率**: 1-5秒
- **记录数**: 约2400万GPS点
- **文件格式**: 每个用户一个目录，多个PLT文件
- **格式**: PLT格式 (经纬度, 海拔, 日期, 时间)
- **处理脚本**: `generate_geolife.py`

## 📊 数据集统计对比

| 数据集 | 车辆/用户数 | 时间跨度 | 轨迹点数 | 采样间隔 | 覆盖区域 |
|--------|-------------|----------|----------|----------|----------|
| T-Drive | 10,357辆 | 1周 | ~1500万 | 1-5秒 | 北京 |
| Cabspotting | 536辆 | 30天 | ~1100万 | <60秒 | 旧金山 |
| Porto | 442辆 | 1年 | ~1.7亿 | 15秒 | 波尔图 |
| GeoLife | 182用户 | 5年 | ~2400万 | 1-5秒 | 北京/西雅图 |

## 🔄 统一处理流程

### 标准输出格式
所有数据集经过处理后统一保存为NPZ格式，包含：
- `position`: [经度, 纬度] 坐标数组
- `timestamp`: 相对时间戳（分钟）
- `taxi_id`/`user_id`: 标识符
- `trip_id`: 轨迹内部ID
- `original_length`: 原始轨迹长度

### 处理脚本参数
所有生成脚本支持相同的配置参数：
```yaml
client: 100                    # 选择的客户端数量
selection_method: "random"     # 选择方法: random/most_traj
traj_number_min: 100           # 最小轨迹点数要求
```

## 📚 参考文献

1. **T-Drive**: Yuan, J., Zheng, Y., Xie, X., & Sun, G. (2010). "Driving with knowledge from the physical world." *Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining*.

2. **Cabspotting**: Piorkowski, M., Sarafijanovoc-Djukic, N., & Grossglauser, M. (2009). "A Parsimonious Model of Mobile Partitioned Networks with Clustering." *COMSNETS 2009*.

3. **Porto Taxi**: Moreira-Matias, L., Gama, J., Ferreira, M., Mendes-Moreira, J., & Damas, L. (2013). "Predicting taxi–passenger demand using streaming data." *IEEE Transactions on Knowledge and Data Engineering*, 26(3), 681-694.

4. **GeoLife**: Zheng, Y., Zhang, L., Xie, X., & Ma, W. Y. (2009). "Mining interesting locations and travel sequences from GPS trajectories." *Proceedings of the 18th international conference on World wide web*.

## 🔗 快速开始

### 生成本地数据集
```bash
# 生成T-Drive数据集
python generate_tdrive.py

# 生成Cabspotting数据集  
python generate_cabspotting.py

# 生成Porto数据集
python generate_porto.py

# 生成GeoLife数据集
python generate_geolife.py
```

### 检查生成结果
```python
import numpy as np

# 加载生成的轨迹数据
data = np.load('tdrive/0.npz')
print(f"轨迹点数: {len(data['position'])}")
print(f"时间范围: {data['timestamp'].min():.1f} - {data['timestamp'].max():.1f} 分钟")
print(f"地理范围: {data['position'][:,0].min():.3f}°E - {data['position'][:,0].max():.3f}°E")
print(f"地理范围: {data['position'][:,1].min():.3f}°N - {data['position'][:,1].max():.3f}°N")
```