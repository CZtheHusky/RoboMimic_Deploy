import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.path_config import PROJECT_ROOT
from common.ctrlcomp import *
from FSM.FSM import *
from typing import Union
import numpy as np
import time
import os
import yaml

from unitree_cpp import UnitreeController, RobotState  # type: ignore

from common.rotation_helper import get_gravity_orientation_real, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config_cpp import RobotConfig


class Controller:
    def __init__(self, config: RobotConfig):
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
        
    def run(self):
        try:
            loop_start_time = time.time()
            
            # Update robot state
            self.update_state()
            
            # Handle remote controller commands
            if self.remote_controller.is_button_pressed(KeyMap.F1):
                self.state_cmd.skill_cmd = FSMCommand.PASSIVE
            if self.remote_controller.is_button_pressed(KeyMap.start):
                self.state_cmd.skill_cmd = FSMCommand.POS_RESET
            if self.remote_controller.is_button_pressed(KeyMap.A) and self.remote_controller.is_button_pressed(KeyMap.R1):
                self.state_cmd.skill_cmd = FSMCommand.LOCO
            if self.remote_controller.is_button_pressed(KeyMap.X) and self.remote_controller.is_button_pressed(KeyMap.R1):
                self.state_cmd.skill_cmd = FSMCommand.SKILL_1
            if self.remote_controller.is_button_pressed(KeyMap.Y) and self.remote_controller.is_button_pressed(KeyMap.R1):
                self.state_cmd.skill_cmd = FSMCommand.SKILL_2
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
        
        
if __name__ == "__main__":
    config = RobotConfig()
    
    controller = Controller(config)
    
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
        print("Exit")
    
