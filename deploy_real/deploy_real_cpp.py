import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.path_config import PROJECT_ROOT
from common.ctrlcomp import *
from FSM.FSM import *
from typing import Union, Optional
import numpy as np
import time
import os
import yaml
import argparse
import subprocess

from unitree_cpp import UnitreeController, RobotState  # type: ignore

from common.rotation_helper import get_gravity_orientation_real, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config_cpp import RobotConfig
from deploy_real.shared_state import SharedStateBuffer


def compute_state_dim(num_joints: int) -> int:
    """
    Define the length of the shared state vector.
    Layout (float32):
        [0..num_joints)              : qj
        [num_joints..2*num_joints)   : dqj
        [..+4]                       : base quat (w, x, y, z)
        [..+3]                       : ang_vel (x, y, z)
        [..+3]                       : gravity_orientation (x, y, z)
    """
    return 2 * num_joints + 4 + 3 + 3


class Controller:
    def __init__(
        self,
        config: RobotConfig,
        local_state: Optional[SharedStateBuffer] = None,
        remote_state: Optional[SharedStateBuffer] = None,
    ):
        self.config = config
        self.remote_controller = RemoteController()
        self.num_joints = config.num_dofs
        self.control_dt = config.unitree.control_dt
        
        cfg_unitree = config.unitree.to_dict()
        cfg_unitree["num_dofs"] = config.num_dofs
        cfg_unitree["stiffness"] = config.stiffness
        cfg_unitree["damping"] = config.damping
        self.unitree = UnitreeController(cfg_unitree)
        self.robot_state: RobotState = None
        
        self.policy_output_action = np.zeros(self.num_joints, dtype=np.float32)
        self.kps = np.zeros(self.num_joints, dtype=np.float32)
        self.kds = np.zeros(self.num_joints, dtype=np.float32)
        self.qj = np.zeros(self.num_joints, dtype=np.float32)
        self.dqj = np.zeros(self.num_joints, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.gravity_orientation = np.array([0,0,-1], dtype=np.float32)
        
        self.state_cmd = StateAndCmd(self.num_joints)
        self.policy_output = PolicyOutput(self.num_joints)
        self.FSM_controller = FSM(self.state_cmd, self.policy_output, enable_logging=True, log_dir="logs", log_prefix="real", robot_uid = config.robot_uid)
        
        self.running = True
        self.counter_over_time = 0

        # Shared memory buffers for point-to-point state exchange
        self.local_state_buf: Optional[SharedStateBuffer] = local_state
        self.remote_state_buf: Optional[SharedStateBuffer] = remote_state
        self.state_dim = compute_state_dim(self.num_joints)
        self._local_state_vec = (
            np.zeros(self.state_dim, dtype=np.float32)
            if self.local_state_buf is not None
            else None
        )
        self._remote_state_vec = (
            np.zeros(self.state_dim, dtype=np.float32)
            if self.remote_state_buf is not None
            else None
        )
        # Latest snapshot of opponent / peer robot state (float32 vector),
        # or None if no data has been received yet.
        self.oppo_state: Optional[np.ndarray] = None
        self.latency_ms = []
        
        # Wait for connection
        self.wait_for_connection()
        
    def wait_for_connection(self):
        print("Waiting for connection to robot...")
        for _ in range(30):
            time.sleep(0.1)
            if self.unitree.self_check():
                print("Successfully connected to the robot.")
                return
        print("Warning: Connection check failed, but continuing...")

    def update_state(self):
        """Update robot state from unitree controller"""
        self.robot_state = self.unitree.get_robot_state()
        
        # Update joint states
        for i in range(self.num_joints):
            self.qj[i] = self.robot_state.motor_state.q[i]
            self.dqj[i] = self.robot_state.motor_state.dq[i]
        
        # Update IMU state - unitree_cpp returns quaternion as [w, x, y, z]
        quat_wxyz = np.asarray(self.robot_state.imu_state.quaternion, dtype=np.float32)
        self.quat = quat_wxyz  # Keep as [w, x, y, z] format
        self.ang_vel = np.array(self.robot_state.imu_state.gyroscope, dtype=np.float32)
        
        # Update remote controller if wireless remote data is available
        if hasattr(self.robot_state, 'wireless_remote'):
            self.remote_controller.set(self.robot_state.wireless_remote)

        # Read opponent / peer state from shared memory (if enabled)
        if self.remote_state_buf is not None:
            snapshot = self.remote_state_buf.read(out=self._remote_state_vec)
            if snapshot is not None:
                # Keep the latest consistent snapshot as oppo_state
                self.oppo_state = snapshot
                # print("oppo_state[0] =", self.oppo_state[0])

    def _update_local_shared_state(self):
        """Write the current local state snapshot into shared memory."""
        if self.local_state_buf is None or self._local_state_vec is None:
            return

        v = self._local_state_vec
        idx = 0

        # Joint positions and velocities
        v[idx : idx + self.num_joints] = self.qj
        idx += self.num_joints
        v[idx : idx + self.num_joints] = self.dqj
        idx += self.num_joints

        # Base orientation and angular velocity
        v[idx : idx + 4] = self.quat
        idx += 4
        v[idx : idx + 3] = self.ang_vel
        idx += 3

        # Gravity orientation
        v[idx : idx + 3] = self.gravity_orientation

        self.local_state_buf.write(v)
        
    def run(self):
        try:
            loop_start_time = time.time()
            
            # Update robot state and peer state from shared memory
            self.update_state()
            
            # Handle remote controller commands
            if self.remote_controller.is_button_pressed(KeyMap.F1):
                self.state_cmd.skill_cmd = FSMCommand.PASSIVE
            if self.remote_controller.is_button_pressed(KeyMap.start):
                self.state_cmd.skill_cmd = FSMCommand.POS_RESET
            if self.remote_controller.is_button_pressed(KeyMap.A) and self.remote_controller.is_button_pressed(KeyMap.R1):
                self.state_cmd.skill_cmd = FSMCommand.LOCO
            if self.remote_controller.is_button_pressed(KeyMap.Y) and self.remote_controller.is_button_pressed(KeyMap.L1):
                self.state_cmd.skill_cmd = FSMCommand.SKILL_4
            # Set velocity commands from joystick
            self.state_cmd.vel_cmd[0] = self.remote_controller.ly
            self.state_cmd.vel_cmd[1] = self.remote_controller.lx * -1
            self.state_cmd.vel_cmd[2] = self.remote_controller.rx * -1

            # Calculate gravity orientation
            gravity_orientation = get_gravity_orientation_real(self.quat)
            
            # Update state command
            self.state_cmd.q = self.qj.copy()
            self.state_cmd.dq = self.dqj.copy()
            self.state_cmd.gravity_ori = gravity_orientation.copy()
            self.state_cmd.ang_vel = self.ang_vel.astype(np.float64).copy()
            self.state_cmd.base_quat = self.quat.copy()
            self.state_cmd.oppo_state = self.oppo_state.copy() if self.oppo_state is not None else None
            # Also keep a float32 copy for local logging / shared memory
            self.gravity_orientation = gravity_orientation.astype(np.float32)

            # Update local shared memory snapshot (for sender process)
            self._update_local_shared_state()
            
            # Run FSM controller
            self.FSM_controller.run()
            policy_output_action = self.policy_output.actions.copy()
            kps = self.policy_output.kps.copy()
            kds = self.policy_output.kds.copy()
            
            # Set gains and send command using unitree_cpp
            self.unitree.set_gains(kps.tolist(), kds.tolist())
            self.unitree.step(policy_output_action.tolist())
            
            # Timing control
            loop_end_time = time.time()
            delta_time = loop_end_time - loop_start_time
            if delta_time < self.control_dt:
                time.sleep(self.control_dt - delta_time)
                self.counter_over_time = 0
            else:
                print("control loop over time.")
                self.counter_over_time += 1
                
        except ValueError as e:
            print(str(e))
        except Exception as e:
            print(f"Error in run loop: {e}")
    
    def shutdown(self):
        """Shutdown the controller gracefully"""
        print("Shutting down controller...")
        # Set damping mode for safe shutdown
        damping_gains = [8.0] * self.num_joints
        zero_gains = [0.0] * self.num_joints
        zero_pos = [0.0] * self.num_joints
        self.unitree.set_gains(zero_gains, damping_gains)
        self.unitree.step(zero_pos)
        self.unitree.shutdown()
        
        
"""
python deploy_real/deploy_real_cpp.py 0 --peer-ip 192.168.31.120 --send-port 50051 --recv-port 50050 --state-rate-hz 100
python deploy_real/deploy_real_cpp.py 2 --peer-ip 192.168.31.87 --send-port 50050 --recv-port 50051 --state-rate-hz 100
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real robot controller with state sharing")
    parser.add_argument("robot_uid", type=int, help="Unique ID of the robot")
    parser.add_argument(
        "--peer-ip",
        type=str,
        default=None,
        help="Peer robot IP address for point-to-point state exchange",
    )
    parser.add_argument(
        "--send-port",
        type=int,
        default=50051,
        help="UDP port on peer robot to send state to",
    )
    parser.add_argument(
        "--recv-port",
        type=int,
        default=50050,
        help="Local UDP port to receive peer state on",
    )
    parser.add_argument(
        "--state-rate-hz",
        type=float,
        default=50,
        help="Sender process rate limit in Hz (0 = as fast as possible)",
    )
    parser.add_argument(
        "--shm-prefix",
        type=str,
        default=None,
        help="Prefix for shared memory names (defaults to robot_<uid>)",
    )
    args = parser.parse_args()

    config = RobotConfig()
    config.robot_uid = args.robot_uid

    # Shared memory + sender/receiver processes (optional)
    state_dim = compute_state_dim(config.num_dofs)
    shm_prefix = args.shm_prefix or f"robot_{config.robot_uid}"
    local_shm_name = f"{shm_prefix}_local_state"
    remote_shm_name = f"{shm_prefix}_remote_state"

    local_buf: Optional[SharedStateBuffer] = None
    remote_buf: Optional[SharedStateBuffer] = None
    sender_proc: Optional[subprocess.Popen] = None
    receiver_proc: Optional[subprocess.Popen] = None
    rtt_proc: Optional[subprocess.Popen] = None

    if args.peer_ip:
        print(
            f"[main] Enabling state sharing. "
            f"state_dim={state_dim}, local_shm={local_shm_name}, remote_shm={remote_shm_name}"
        )
        # Create shared memory segments
        local_buf = SharedStateBuffer.create(local_shm_name, state_dim)
        remote_buf = SharedStateBuffer.create(remote_shm_name, state_dim)

        py = sys.executable
        base_dir = Path(__file__).parent

        # Sender: local -> peer
        sender_cmd = [
            py,
            str(base_dir / "state_sender.py"),
            "--shm-name",
            local_shm_name,
            "--state-dim",
            str(state_dim),
            "--peer-ip",
            args.peer_ip,
            "--peer-port",
            str(args.send_port),
            "--rate-hz",
            str(args.state_rate_hz),
        ]
        # Receiver: peer -> local
        receiver_cmd = [
            py,
            str(base_dir / "state_receiver.py"),
            "--shm-name",
            remote_shm_name,
            "--state-dim",
            str(state_dim),
            "--listen-port",
            str(args.recv_port),
        ]

        print(f"[main] Starting state_sender: {' '.join(sender_cmd)}")
        sender_proc = subprocess.Popen(sender_cmd)
        print(f"[main] Starting state_receiver: {' '.join(receiver_cmd)}")
        receiver_proc = subprocess.Popen(receiver_cmd)

        # RTT ping/pong helper (optional, for monitoring network RTT)
        rtt_cmd = [
            py,
            str(base_dir / "rtt_ping.py"),
            "--peer-ip",
            args.peer_ip,
        ]
        print(f"[main] Starting rtt_ping: {' '.join(rtt_cmd)}")
        rtt_proc = subprocess.Popen(rtt_cmd)
    else:
        print("[main] Peer IP not provided; state sharing is disabled.")

    controller = Controller(
        config,
        local_state=local_buf,
        remote_state=remote_buf,
    )

    try:
        while True:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.is_button_pressed(KeyMap.select):
                break
    except KeyboardInterrupt:
        pass
    finally:
        controller.shutdown()

        # Stop child processes and clean shared memory
        if sender_proc is not None:
            sender_proc.terminate()
        if receiver_proc is not None:
            receiver_proc.terminate()
        if rtt_proc is not None:
            rtt_proc.terminate()
        if local_buf is not None:
            local_buf.close()
            try:
                local_buf.unlink()
            except FileNotFoundError:
                pass
        if remote_buf is not None:
            remote_buf.close()
            try:
                remote_buf.unlink()
            except FileNotFoundError:
                pass

        print("Exit")
