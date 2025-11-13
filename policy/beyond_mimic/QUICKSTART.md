# 快速开始 - BeyondMimic 数据记录

## 5 分钟快速上手

### 1. 修改你的部署脚本

如果你使用的是 `FSM/FSM.py`，需要修改它以支持日志参数传递：

```python
# 在 FSM 的 __init__ 方法中，找到创建 BeyondMimic 的地方
# 修改为：

self.beyond_mimic_state = BeyondMimic(
    state_cmd, 
    policy_output,
    enable_logging=enable_logging,  # 新增
    log_dir=log_dir  # 新增
)
```

### 2. 启用日志记录

在创建控制器时启用：

```python
# 在 deploy_real_cpp.py 或类似文件中
from FSM.FSM import FSM

# 启用日志
fsm_controller = FSM(
    state_cmd=state_cmd,
    policy_output=policy_output,
    enable_logging=True,  # 启用
    log_dir="./logs/test_run"  # 指定目录
)
```

### 3. 运行你的程序

```bash
python deploy_real_cpp.py
```

程序运行时会在 `./logs/test_run/` 目录下生成日志文件。

### 4. 分析数据

```bash
cd policy/beyond_mimic
python analyze_logs.py --log-dir ../../logs/test_run
```

这会生成：
- 关节轨迹图 (`joint_trajectories.png`)
- 动作轨迹图 (`action_trajectories.png`)
- IMU 数据图 (`imu_data.png`)

## 完整示例

```python
#!/usr/bin/env python3
"""
完整的部署示例，包含数据记录
"""

from config import Config
from common.ctrlcomp import StateAndCmd, PolicyOutput
from FSM.FSM import FSM
from common.remote_controller import RemoteController, KeyMap
from unitree_cpp import UnitreeController, RobotState

class Controller:
    def __init__(self, config: Config, enable_logging=False):
        self.config = config
        self.num_joints = config.num_joints
        
        # 初始化 unitree 控制器
        self.unitree = UnitreeController(config.unitree.to_dict())
        
        # 创建状态和输出
        self.state_cmd = StateAndCmd(self.num_joints)
        self.policy_output = PolicyOutput(self.num_joints)
        
        # 创建 FSM 控制器，启用日志
        self.FSM_controller = FSM(
            self.state_cmd,
            self.policy_output,
            enable_logging=enable_logging,
            log_dir="./logs/experiment"
        )
        
        self.remote_controller = RemoteController()
        
    def run(self):
        # 获取机器人状态
        robot_state = self.unitree.get_robot_state()
        
        # 更新状态
        self.state_cmd.q = robot_state.motor_state.q
        self.state_cmd.dq = robot_state.motor_state.dq
        
        # 运行 FSM（会自动记录数据）
        self.FSM_controller.run()
        
        # 发送命令
        self.unitree.set_gains(
            self.policy_output.kps.tolist(),
            self.policy_output.kds.tolist()
        )
        self.unitree.step(self.policy_output.actions.tolist())

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='Enable logging')
    args = parser.parse_args()
    
    config = Config()
    controller = Controller(config, enable_logging=args.log)
    
    try:
        while True:
            controller.run()
    except KeyboardInterrupt:
        print("Exiting...")
```

运行时：
```bash
# 不记录
python deploy.py

# 记录数据
python deploy.py --log
```

## 常见问题

### Q: 我的日志文件在哪里？
A: 默认在 `./logs/beyond_mimic/` 目录下，文件名类似 `observations_20251113_143022.pkl`

### Q: 如何查看记录了多少数据？
A: 在程序退出时会打印统计信息，或者使用 `ls -lh logs/beyond_mimic/` 查看文件大小

### Q: 数据记录会影响性能吗？
A: 影响极小（< 0.1ms），因为使用了异步写入

### Q: 如何自定义记录的数据？
A: 修改 `BeyondMimic.py` 中的 `obs_log_data` 和 `action_log_data` 字典

### Q: 可以实时查看数据吗？
A: 当前版本只支持离线分析，未来版本会添加实时可视化

## 下一步

- 阅读完整文档：`README_LOGGING.md`
- 查看数据分析示例：`analyze_logs.py`
- 运行测试：`python common/test_data_logger.py`
- 查看实现总结：`DATA_LOGGING_SUMMARY.md`
