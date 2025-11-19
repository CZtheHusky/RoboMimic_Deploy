<div align="center">
  <h1 align="center">RoboMimic Deploy</h1>
  <p align="center">
    <a href="README.md">🌎 English</a> | <span>🇨🇳 中文</span>
  </p>
</div>

<p align="center">
  🎮🚪 <strong>RoboMimic Deploy 是一个基于有限状态机（FSM）的多策略部署框架，目前策略针对宇树 G1（29 DoF）机器人，支持 Mujoco 仿真与真机部署。</strong> 🚪🎮
</p>

## 写在前面

- 本部署框架仅适用于**具有三自由度腰部的 G1 机器人**，若装有腰部固定件，请按官网教程解锁后再使用。
- 建议拆下手掌，当前舞蹈 / Mimic 动作存在上肢干涉风险。
- 实际部署中出现的大多数问题来自**策略适应性不足**，不必过度怀疑硬件缺陷。
- 强烈建议先在 **Mujoco 仿真** 中熟悉操作，再进行真机实验。
- 视频教程：[Bilibili 链接](https://www.bilibili.com/video/BV1VTKHzSE6C/?vd_source=713b35f59bdf42930757aea07a44e7cb#reply114743994027967)

## 环境与安装

### 1. 创建虚拟环境

建议在虚拟环境中运行训练或部署程序，推荐使用 Conda：

```bash
# Mujoco 仿真（推荐在 PC 上）
conda create -n robomimic python=3.10 -y
conda activate robomimic

# 真机 G1 部署（Jetson / x86 + CUDA）
conda create -n robomimic python=3.8 -y
conda activate robomimic
```

### 2. 安装依赖

```bash
# 2.1 PyTorch
# for mujoco（按自己 CUDA 版本调整）
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# for g1（示例：JetPack 对应的 wheel）
wget https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
pip install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
```

安装 Unitree SDK Python 封装：

```bash
# 2.2 unitree_cpp（真机必需）
# 参考开源仓库进行编译安装：
#   https://github.com/HansZ8/unitree_cpp
```

安装本项目 Python 依赖：

```bash
# 2.3 RoboMimic_Deploy
cd RoboMimic_Deploy
pip install numpy==1.20.0 
pip install onnx==1.17.0 onnxruntime==1.19.2 pygame scipy matplotlib pyyaml pydantic
```

## 脚本与模块功能一览

- **仿真部署**
  - `deploy_mujoco/deploy_mujoco.py`  
    Mujoco 仿真入口脚本：读取 `deploy_mujoco/config/mujoco.yaml`，加载 G1 XML 模型，使用手柄控制机器人，通过有限状态机（FSM）在各个策略间切换，并将仿真轨迹记录到 `logs/`。

- **真机部署与双机状态共享**
  - `deploy_real/deploy_real_cpp.py`  
    真机控制主程序：基于 `unitree_cpp` 获取 G1 状态并下发关节指令，支持可选的**双机点对点状态共享**（通过共享内存 + UDP，适合双人舞 / 对舞等场景）。
  - `deploy_real/state_sender.py` / `deploy_real/state_receiver.py`  
    状态发送 / 接收脚本：从共享内存读取本机状态，经 UDP 发送到对端，或将接收到的对端状态写回共享内存。通常由 `deploy_real_cpp.py` 自动启动，无需手动调用。
  - `deploy_real/shared_state.py`  
    共享内存工具：封装基于 `multiprocessing.shared_memory` 的 seqlock 读写，保证状态快照一致性。
  - `deploy_real/rtt_ping.py`  
    网络 RTT 测试脚本：周期性发送 ping/pong，估计两台机器人之间的往返时延。可由主程序拉起，也可单独运行。

- **策略与状态机**
  - `FSM/FSM.py` / `FSM/FSMState.py`  
    有限状态机实现：根据外部命令（手柄 / 遥控器按键）在各个策略之间切换。
  - `policy/passive/PassiveMode.py`  
    阻尼保护模式：关节刚度清零，仅保留阻尼，便于手动干预或紧急安全。
  - `policy/fixedpose/FixedPose.py`  
    固定姿态：将机器人从当前姿态平滑过渡到默认站立姿态（位置控制恢复）。
  - `policy/loco_mode/LocoMode.py`  
    行走策略：基于 TorchScript 的行走策略，使用摇杆线速度 / 角速度指令控制稳态行走。
  - `policy/skill_cast/SkillCast.py`  
    技能准备：保持下肢+腰部由行走策略平衡，同时将上肢渐变到 Mimic 起始姿态，通常在执行 Mimic 之前使用。
  - `policy/skill_cooldown/SkillCooldown.py`  
    技能冷却：下肢继续维持平衡轨迹，上肢渐变回默认姿态，通常在 Mimic 结束后使用。
  - `policy/beyond_mimic/BeyondMimic.py`  
    BeyondMimic 策略：加载 ONNX 模型，根据参考动作轨迹生成整机控制命令，并可选记录高频观测与动作数据。

- **通用工具**
  - `common/ctrlcomp.py`  
    定义机器人状态与指令结构（`StateAndCmd`）及策略输出结构（`PolicyOutput`）。
  - `common/joystick.py` / `common/remote_controller.py`  
    将 Xbox 手柄 / Unitree 无线遥控器统一为标准按键和摇杆接口。
  - `common/rotation_helper.py` / `common/utils.py` / `common/path_config.py`  
    提供重力方向计算、坐标变换、配置路径管理等辅助函数。
  - `common/data_logger.py`  
    异步数据记录工具：使用后台线程和队列写入 `pickle` / 压缩 `pickle` 日志，并提供 `StreamingPickleLogger` 与 `load_streaming_pickle` 便于之后分析。
  - `policy/beyond_mimic/analyze_logs.py`  
    示例分析脚本：从 `logs/beyond_mimic/...` 读取 `observations` / `actions` 日志，进行时序与频率分析及可视化。

## 运行示例

### 1. Mujoco 仿真

```bash
cd RoboMimic_Deploy
python deploy_mujoco/deploy_mujoco.py
```

操作要点（Xbox 手柄）：

- `START`：切换到 FixedPose（位控恢复默认站立）。
- `L3` 松开：切换到 PassiveMode（阻尼保护）。
- `R1` + `A`：切换到 LocoMode，机器人站起后可用摇杆行走。
- `L1` + `Y`：触发 BeyondMimic 策略。
- `SELECT`：退出仿真程序。

### 2. 真机部署（单机或双机）

开机后将机器人吊起来，按遥控器 `L2 + R2` 进入调试模式。

```bash
cd RoboMimic_Deploy
# 单机控制（不做状态共享）
python deploy_real/deploy_real_cpp.py <robot_uid>

# 双机状态共享示例（两台机器人分别运行）
python deploy_real/deploy_real_cpp.py 0 --peer-ip <robot2_ip> --send-port 50051 --recv-port 50050 --state-rate-hz 100
python deploy_real/deploy_real_cpp.py 1 --peer-ip <robot1_ip> --send-port 50050 --recv-port 50051 --state-rate-hz 100
```

遥控器按键映射（`RemoteController`）：

- `F1`：切换到 PassiveMode（阻尼保护）。
- `START`：切换到 FixedPose（位控站立），此时可将机器人放下。
- `R1` + `A`：切换到 LocoMode，使用左摇杆控制平移，右摇杆控制旋转。
- `L1` + `Y`：触发 BeyondMimic 策略。
- `SELECT`：立即退出控制程序。

## 日志说明

- 在仿真或真机执行 BeyondMimic 时，若开启日志记录，会在 `logs/beyond_mimic/robot_<uid>/` 下生成 `*_observations_*.pkl` 与 `*_actions_*.pkl`。
- 可使用 `policy/beyond_mimic/analyze_logs.py` 对日志进行载入与可视化分析。

## 安全与注意事项

- Mimic / BeyondMimic 策略**不保证 100% 成功率**，在湿滑地面、沙地等复杂场景失败概率会明显升高。
- 如出现失控：
  - 仿真：松开 `L3` 切换到 PassiveMode，或按 `SELECT` 直接退出。
  - 真机：按遥控器 `F1` 进入阻尼保护，必要时立刻扶住或放倒机器人，并按 `SELECT` 退出程序。
- **务必**先在仿真环境充分验证策略与操作流程，再尝试真机部署。

