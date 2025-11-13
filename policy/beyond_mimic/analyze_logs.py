"""
Example script for analyzing BeyondMimic logged data.

This script demonstrates how to load and analyze the observation and action data
logged during policy execution.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))
from common.data_logger import load_streaming_pickle


def load_latest_logs(log_dir: str = "./logs/beyond_mimic"):
    """
    Load the most recent observation and action logs.
    
    Args:
        log_dir: Directory containing log files
        
    Returns:
        Tuple of (observations, actions) lists
    """
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f"Log directory not found: {log_dir}")
        return None, None
    
    # Find latest observation and action files
    obs_files = sorted(log_path.glob("observations_*.pkl"))
    action_files = sorted(log_path.glob("actions_*.pkl"))
    
    if not obs_files or not action_files:
        print("No log files found")
        return None, None
    
    # Load latest files
    obs_file = obs_files[-1]
    action_file = action_files[-1]
    
    print(f"Loading observations from: {obs_file.name}")
    print(f"Loading actions from: {action_file.name}")
    
    observations = load_streaming_pickle(str(obs_file))
    actions = load_streaming_pickle(str(action_file))
    
    print(f"Loaded {len(observations)} observation samples")
    print(f"Loaded {len(actions)} action samples")
    
    return observations, actions


def analyze_timing(data: List[Dict[str, Any]]):
    """Analyze timing statistics of logged data."""
    if not data:
        return
    
    timestamps = np.array([d['_timestamp'] for d in data])
    relative_time = timestamps - timestamps[0]
    
    # Calculate intervals
    intervals = np.diff(timestamps)
    
    print("\n=== Timing Analysis ===")
    print(f"Total duration: {relative_time[-1]:.2f} seconds")
    print(f"Number of samples: {len(data)}")
    print(f"Average interval: {np.mean(intervals)*1000:.2f} ms")
    print(f"Std dev interval: {np.std(intervals)*1000:.2f} ms")
    print(f"Min interval: {np.min(intervals)*1000:.2f} ms")
    print(f"Max interval: {np.max(intervals)*1000:.2f} ms")
    print(f"Average frequency: {1/np.mean(intervals):.2f} Hz")
    
    return relative_time, intervals


def plot_joint_trajectories(observations: List[Dict], num_joints: int = 5):
    """Plot joint position and velocity trajectories."""
    if not observations:
        return
    
    # Extract data
    joint_pos = np.array([obs['joint_pos'] for obs in observations])
    joint_vel = np.array([obs['joint_vel'] for obs in observations])
    timestamps = np.array([obs['_timestamp'] for obs in observations])
    relative_time = timestamps - timestamps[0]
    
    # Plot first few joints
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Joint positions
    for i in range(min(num_joints, joint_pos.shape[1])):
        axes[0].plot(relative_time, joint_pos[:, i], label=f'Joint {i}', alpha=0.7)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Joint Position (rad)')
    axes[0].set_title('Joint Positions Over Time')
    axes[0].legend()
    axes[0].grid(True)
    
    # Joint velocities
    for i in range(min(num_joints, joint_vel.shape[1])):
        axes[1].plot(relative_time, joint_vel[:, i], label=f'Joint {i}', alpha=0.7)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Joint Velocity (rad/s)')
    axes[1].set_title('Joint Velocities Over Time')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('joint_trajectories.png', dpi=150)
    print("Saved joint trajectories to: joint_trajectories.png")
    plt.show()


def plot_actions(actions: List[Dict], num_joints: int = 5):
    """Plot action trajectories."""
    if not actions:
        return
    
    # Extract data
    raw_actions = np.array([act['raw_action'] for act in actions])
    scaled_actions = np.array([act['scaled_action'] for act in actions])
    timestamps = np.array([act['_timestamp'] for act in actions])
    relative_time = timestamps - timestamps[0]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Raw actions
    for i in range(min(num_joints, raw_actions.shape[1])):
        axes[0].plot(relative_time, raw_actions[:, i], label=f'Joint {i}', alpha=0.7)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Raw Action')
    axes[0].set_title('Raw Network Output Over Time')
    axes[0].legend()
    axes[0].grid(True)
    
    # Scaled actions
    for i in range(min(num_joints, scaled_actions.shape[1])):
        axes[1].plot(relative_time, scaled_actions[:, i], label=f'Joint {i}', alpha=0.7)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Target Position (rad)')
    axes[1].set_title('Scaled Actions (Target Positions) Over Time')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('action_trajectories.png', dpi=150)
    print("Saved action trajectories to: action_trajectories.png")
    plt.show()


def plot_imu_data(observations: List[Dict]):
    """Plot IMU data (angular velocity)."""
    if not observations:
        return
    
    ang_vel = np.array([obs['ang_vel'] for obs in observations])
    timestamps = np.array([obs['_timestamp'] for obs in observations])
    relative_time = timestamps - timestamps[0]
    
    plt.figure(figsize=(12, 6))
    plt.plot(relative_time, ang_vel[:, 0], label='Roll rate', alpha=0.7)
    plt.plot(relative_time, ang_vel[:, 1], label='Pitch rate', alpha=0.7)
    plt.plot(relative_time, ang_vel[:, 2], label='Yaw rate', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('IMU Angular Velocity Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('imu_data.png', dpi=150)
    print("Saved IMU data to: imu_data.png")
    plt.show()


def compute_statistics(observations: List[Dict], actions: List[Dict]):
    """Compute and print various statistics."""
    print("\n=== Data Statistics ===")
    
    # Observation statistics
    if observations:
        joint_pos = np.array([obs['joint_pos'] for obs in observations])
        joint_vel = np.array([obs['joint_vel'] for obs in observations])
        
        print("\nJoint Positions:")
        print(f"  Mean: {np.mean(joint_pos, axis=0)}")
        print(f"  Std:  {np.std(joint_pos, axis=0)}")
        print(f"  Min:  {np.min(joint_pos, axis=0)}")
        print(f"  Max:  {np.max(joint_pos, axis=0)}")
        
        print("\nJoint Velocities:")
        print(f"  Mean: {np.mean(joint_vel, axis=0)}")
        print(f"  Std:  {np.std(joint_vel, axis=0)}")
        print(f"  Max abs: {np.max(np.abs(joint_vel), axis=0)}")
    
    # Action statistics
    if actions:
        raw_actions = np.array([act['raw_action'] for act in actions])
        scaled_actions = np.array([act['scaled_action'] for act in actions])
        
        print("\nRaw Actions:")
        print(f"  Mean: {np.mean(raw_actions, axis=0)}")
        print(f"  Std:  {np.std(raw_actions, axis=0)}")
        
        print("\nScaled Actions:")
        print(f"  Mean: {np.mean(scaled_actions, axis=0)}")
        print(f"  Std:  {np.std(scaled_actions, axis=0)}")


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze BeyondMimic logged data')
    parser.add_argument('--log-dir', type=str, default='./logs/beyond_mimic',
                       help='Directory containing log files')
    parser.add_argument('--num-joints', type=int, default=5,
                       help='Number of joints to plot')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting (only print statistics)')
    
    args = parser.parse_args()
    
    # Load data
    observations, actions = load_latest_logs(args.log_dir)
    
    if observations is None or actions is None:
        print("Failed to load data")
        return
    
    # Timing analysis
    print("\nObservations:")
    analyze_timing(observations)
    
    print("\nActions:")
    analyze_timing(actions)
    
    # Statistics
    compute_statistics(observations, actions)
    
    # Plotting
    if not args.no_plot:
        print("\nGenerating plots...")
        plot_joint_trajectories(observations, args.num_joints)
        plot_actions(actions, args.num_joints)
        plot_imu_data(observations)
        print("\nAll plots saved!")


if __name__ == "__main__":
    main()
