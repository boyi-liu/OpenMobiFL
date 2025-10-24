# 轨迹数据集推荐

除了已有的T-Drive数据集，以下是一些优质的开源轨迹数据集推荐：

## 🚗 车辆轨迹数据集

### 1. GeoLife GPS Trajectories (微软研究院)
- **下载地址**: https://www.microsoft.com/en-us/download/details.aspx?id=52367
- **数据规模**: 182位用户，17,621条轨迹，2007-2012年
- **覆盖区域**: 北京、西雅图等城市
- **采样频率**: 1-5秒
- **特点**: 多模式交通（步行、自行车、公交、汽车）
- **适用场景**: 轨迹挖掘、用户行为识别、位置推荐

### 2. Porto Taxi (波尔图出租车)
- **下载地址**: https://tianchi.aliyun.com/dataset/94216
- **数据规模**: 442辆出租车，170万条完整行程，1年时间
- **覆盖区域**: 葡萄牙波尔图市
- **采样频率**: 15秒
- **特点**: 包含行程元数据（费用、时间、天气）
- **适用场景**: 交通预测、路径规划、轨迹相似性计算

### 3. Roma Taxi (罗马出租车)
- **下载地址**: https://ieee-dataport.org/open-access/crawdad-romataxi
- **数据规模**: 320辆出租车，4个月数据
- **覆盖区域**: 意大利罗马市
- **采样频率**: 7秒
- **特点**: 适合城市网络研究
- **适用场景**: 城市交通模式分析

## 🚶 行人轨迹数据集

### 4. YJMob100K (行人移动性)
- **下载地址**: https://doi.org/10.6084/m9.figshare.21424018.v1
- **数据规模**: 100,000+匿名行人GPS轨迹，1年时间
- **覆盖区域**: 日本横滨、印尼雅加达
- **采样频率**: 1Hz
- **特点**: 大规模行人移动数据
- **适用场景**: 行人行为分析、城市计算

## 🚗 自动驾驶相关

### 5. highD Dataset
- **下载地址**: https://www.highd-dataset.com
- **数据规模**: 11,000+车辆轨迹，25Hz采样
- **覆盖区域**: 德国高速公路
- **特点**: 无人机拍摄，高精度
- **适用场景**: 自动驾驶行为研究

### 6. INTERACTION Dataset
- **下载地址**: https://interaction-dataset.com
- **特点**: 交叉路口车辆交互数据
- **适用场景**: 自动驾驶决策算法

## 🏙️ 城市规模数据

### 7. DiDi GAIA Initiative
- **申请地址**: https://gaia.didichuxing.com
- **数据规模**: 10,000+网约车，多城市
- **采样频率**: 2-3秒
- **特点**: 高分辨率，需要学术申请
- **适用场景**: 大规模城市移动性研究

### 8. NYC Taxi Data
- **下载地址**: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- **数据规模**: 每年4亿+行程
- **特点**: 仅包含起止点，无中间轨迹
- **适用场景**: 大规模出行模式分析

## 🔄 合成数据集

### 9. SynMob
- **下载地址**: https://github.com/synmob/generator
- **特点**: 可生成无限规模数据，1秒采样
- **适用场景**: 隐私保护研究、算法测试

### 10. BerlinMod
- **下载地址**: https://github.com/MNTG/BerlinMod
- **特点**: 柏林城市合成轨迹
- **适用场景**: 基准测试

## 📋 数据集处理建议

### 统一数据格式
建议将所有数据集转换为统一的格式：
```
timestamp, user_id, longitude, latitude
```

### 预处理步骤
1. 坐标系统一（WGS84）
2. 异常值过滤
3. 轨迹分段
4. 采样频率统一
5. 隐私保护处理

### 质量评估指标
- 轨迹完整性
- 采样频率稳定性
- 地理覆盖范围
- 时间跨度
- 用户规模

## 🔗 快速开始

### GeoLife 处理示例
```python
# GeoLife数据格式转换示例
def process_geolife_to_tdrive_format(input_dir, output_dir):
    # GeoLife原始数据是plt格式，需要转换为CSV
    # 转换逻辑实现...
    pass
```

### Porto Taxi 处理示例
```python
# Porto Taxi数据格式转换
def process_porto_to_tdrive_format(input_file, output_dir):
    # Porto数据已经是CSV格式，需要字段映射
    # 转换逻辑实现...
    pass
```

## 📚 参考文献
1. Zheng, Y. et al. "GeoLife: A Collaborative Social Networking Service among User, Location and Trajectory." IEEE Data Eng. Bull. 2010.
2. Moreira-Matias, L. et al. "Predicting Taxi–Passenger Demand Using Streaming Data." IEEE TKDE 2013.
3. Krajewski, R. et al. "The highD Dataset: A Drone Dataset of Naturalistic Vehicle Trajectories on German Highways for Validation of Highly Automated Driving Systems." ITSC 2018.