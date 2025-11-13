# BeyondMimic Data Logging

本文档说明如何使用 BeyondMimic 策略的数据记录功能。

## 功能特性

- **异步记录**：主线程将数据放入队列，子线程负责写入，不阻塞控制循环
- **批量写入**：积累多个样本后批量写入，减少 IO 次数
- **自动管理**：根据数据类型自动创建文件，随收随写
- **高效序列化**：使用 pickle 格式，比 JSON 更快且支持 numpy 数组
- **时间戳**：自动为每个样本添加时间戳

## 使用方法

### 1. 基本使用

在创建 BeyondMimic 实例时，启用日志记录：

```python
from policy.beyond_mimic.BeyondMimic import BeyondMimic

# 启用日志记录
policy = BeyondMimic(
    state_cmd=state_cmd,
    policy_output=policy_output,
    enable_logging=True,  # 启用日志
    log_dir="./logs/beyond_mimic"  # 日志保存目录
)
```

### 2. 记录的数据

策略会自动记录以下数据：

#### 观测数据 (observations_YYYYMMDD_HHMMSS.pkl)
- `ref_joint_pos`: 参考关节位置
- `ref_joint_vel`: 参考关节速度
- `motion_anchor_ori`: 运动锚点方向
- `ang_vel`: 角速度
- `joint_pos`: 当前关节位置
- `joint_vel`: 当前关节速度
- `prev_action`: 上一步动作
- `full_obs`: 完整观测向量
- `counter_step`: 步数计数
- `_timestamp`: 时间戳

#### 动作数据 (actions_YYYYMMDD_HHMMSS.pkl)
- `raw_action`: 原始网络输出
- `scaled_action`: 缩放后的动作
- `target_pos_full`: 目标位置（完整）
- `kps`: PD 控制器刚度
- `kds`: PD 控制器阻尼
- `counter_step`: 步数计数
- `_timestamp`: 时间戳

### 3. 读取日志数据

使用提供的工具函数读取日志：

```python
from common.data_logger import load_streaming_pickle

# 读取观测数据
observations = load_streaming_pickle("./logs/beyond_mimic/observations_20251113_143022.pkl")

# 读取动作数据
actions = load_streaming_pickle("./logs/beyond_mimic/actions_20251113_143022.pkl")

# 访问数据
for i, obs in enumerate(observations):
    print(f"Step {i}:")
    print(f"  Joint pos: {obs['joint_pos']}")
    print(f"  Timestamp: {obs['_timestamp']}")
```

### 4. 数据分析示例

```python
import numpy as np
import matplotlib.pyplot as plt
from common.data_logger import load_streaming_pickle

# 加载数据
obs_data = load_streaming_pickle("./logs/beyond_mimic/observations_20251113_143022.pkl")
action_data = load_streaming_pickle("./logs/beyond_mimic/actions_20251113_143022.pkl")

# 提取关节位置时间序列
joint_positions = np.array([obs['joint_pos'] for obs in obs_data])
timestamps = np.array([obs['_timestamp'] for obs in obs_data])

# 相对时间
relative_time = timestamps - timestamps[0]

# 绘制关节 0 的轨迹
plt.figure(figsize=(10, 6))
plt.plot(relative_time, joint_positions[:, 0])
plt.xlabel('Time (s)')
plt.ylabel('Joint 0 Position (rad)')
plt.title('Joint Position Over Time')
plt.grid(True)
plt.show()
```

## 性能优化

### 调整批处理大小

根据控制频率调整批处理参数：

```python
from common.data_logger import StreamingPickleLogger

# 高频控制（50Hz），更大的批处理
logger = StreamingPickleLogger(
    log_dir="./logs",
    batch_size=100,  # 每 100 个样本写一次（2秒）
    flush_interval=5.0,  # 或每 5 秒强制写入
)

# 低频控制（10Hz），更小的批处理
logger = StreamingPickleLogger(
    log_dir="./logs",
    batch_size=20,  # 每 20 个样本写一次（2秒）
    flush_interval=3.0,  # 或每 3 秒强制写入
)
```

### 监控性能

```python
# 在运行过程中检查日志统计
if policy.enable_logging:
    stats = policy.data_logger.get_stats()
    print(f"Total samples logged: {stats['total_samples']}")
    print(f"Queue size: {stats['queue_size']}")
    print(f"Buffer sizes: {stats['buffer_sizes']}")
```

## 注意事项

1. **磁盘空间**：长时间运行会产生大量数据，请确保有足够的磁盘空间
2. **队列溢出**：如果主线程产生数据的速度远超写入速度，队列可能会满，此时会丢弃数据并打印警告
3. **优雅退出**：确保调用 `policy.exit()` 或正常退出 FSM，以便正确刷新所有缓冲数据
4. **文件格式**：使用 pickle 格式存储 numpy 数组，读取时需要 Python 环境

## 文件命名规则

日志文件按照以下规则命名：
- 格式：`{data_name}_{session_id}.pkl`
- `data_name`：数据类型（observations、actions）
- `session_id`：会话 ID（时间戳 YYYYMMDD_HHMMSS）

例如：
- `observations_20251113_143022.pkl`
- `actions_20251113_143022.pkl`

## 故障排查

### 问题：数据没有写入

1. 检查是否启用了日志记录：`enable_logging=True`
2. 检查日志目录是否有写权限
3. 检查是否正常退出（调用了 `exit()` 方法）

### 问题：性能下降

1. 增加 `batch_size` 减少写入频率
2. 增加 `flush_interval` 减少强制刷新频率
3. 检查磁盘 IO 性能

### 问题：队列满告警

1. 增加 `max_queue_size`
2. 增加写入频率（更小的 batch_size）
3. 检查磁盘写入速度是否足够
