"""
Example script for analyzing BeyondMimic logged data.

This script demonstrates how to load and analyze the observation and action data
logged during policy execution.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import yaml
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))
from common.data_logger import load_streaming_pickle


def load_config():
    """Load configuration from BeyondMimic.yaml"""
    config_path = Path(__file__).parent / "config" / "BeyondMimic.yaml"
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


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
        return None, None
    
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


def plot_multi_joint_data(data: List[Dict], joint_names: List[str], data_key: str,
                          ylabel: str, title: str, filename: str, color: str = 'blue'):
    """
    Generic function to plot multiple joints in subplots.
    
    Args:
        data: List of data dictionaries
        joint_names: List of joint names for labeling
        data_key: Key to extract data from dictionary (e.g., 'raw_qj', 'raw_qvel')
        ylabel: Y-axis label (e.g., 'Position (rad)', 'Velocity (rad/s)')
        title: Overall plot title
        filename: Output filename
        color: Line color
    """
    if not data:
        return
    
    # Extract data
    values = np.array([d[data_key] for d in data])
    timestamps = np.array([d['_timestamp'] for d in data])
    relative_time = timestamps - timestamps[0]
    
    num_joints = values.shape[1]
    
    # Create subplots (6 rows x 5 columns for up to 30 joints)
    fig, axes = plt.subplots(6, 5, figsize=(20, 18))
    axes = axes.flatten()
    
    for i in range(num_joints):
        axes[i].plot(relative_time, values[:, i], linewidth=1.5, color=color)
        axes[i].set_title(joint_names[i], fontsize=9)
        axes[i].set_xlabel('Time (s)', fontsize=8)
        axes[i].set_ylabel(ylabel, fontsize=8)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(labelsize=7)
    
    # Hide unused subplots
    for i in range(num_joints, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    # Ensure output directory exists
    try:
        out_path = Path(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
        print(f"Saved {title.lower()} to: {out_path}")
    finally:
        plt.close()
    


def plot_multi_component_data(data: List[Dict], data_key: str, component_labels: List[str],
                              ylabel: str, title: str, filename: str, colors: List[str] = None):
    """
    Generic function to plot multiple components (e.g., quaternion, angular velocity).
    
    Args:
        data: List of data dictionaries
        data_key: Key to extract data from dictionary
        component_labels: Labels for each component
        ylabel: Y-axis label
        title: Overall plot title
        filename: Output filename
        colors: List of colors for each component
    """
    if not data:
        return
    
    # Extract data
    values = np.array([d[data_key] for d in data])
    timestamps = np.array([d['_timestamp'] for d in data])
    relative_time = timestamps - timestamps[0]
    
    num_components = len(component_labels)
    
    if colors is None:
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown'][:num_components]
    
    fig, axes = plt.subplots(num_components, 1, figsize=(12, 3 * num_components + 1))
    
    # Handle single component case
    if num_components == 1:
        axes = [axes]
    
    for i in range(num_components):
        axes[i].plot(relative_time, values[:, i], label=component_labels[i], 
                    color=colors[i], linewidth=1.5)
        axes[i].set_ylabel(ylabel if isinstance(ylabel, str) else ylabel[i], fontsize=10)
        axes[i].set_title(f'{component_labels[i]} Over Time', fontsize=11)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    axes[-1].set_xlabel('Time (s)', fontsize=10)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    # Ensure output directory exists
    try:
        out_path = Path(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
        print(f"Saved {title.lower()} to: {out_path}")
    finally:
        plt.close()


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


def plot_all_joint_positions(observations: List[Dict], joint_names: List[str], output_dir):
    """Plot all 29 joint positions, each in its own subplot."""
    output_file = Path(output_dir) / 'all_joint_positions.png'
    plot_multi_joint_data(observations, joint_names, 'raw_qj',
                         'Position (rad)', 'All Joint Positions Over Time',
                         str(output_file), color='blue')


def plot_all_joint_velocities(observations: List[Dict], joint_names: List[str], output_dir):
    """Plot all 29 joint velocities, each in its own subplot."""
    output_file = Path(output_dir) / 'all_joint_velocities.png'
    plot_multi_joint_data(observations, joint_names, 'raw_qvel',
                         'Velocity (rad/s)', 'All Joint Velocities Over Time',
                         str(output_file), color='orange')


def plot_torso_orientation(observations: List[Dict], output_dir):
    """Plot torso yaw, roll, pitch over time."""
    if not observations:
        return
    
    # Combine yaw, roll, pitch into a single array
    yaw = np.array([obs['base_troso_yaw'] for obs in observations])
    roll = np.array([obs['base_troso_roll'] for obs in observations])
    pitch = np.array([obs['base_troso_pitch'] for obs in observations])
    combined = np.column_stack([yaw, roll, pitch])
    
    # Create temporary data structure for plot_multi_component_data
    temp_data = [{'torso_orientation': combined[i], '_timestamp': obs['_timestamp']} 
                 for i, obs in enumerate(observations)]
    
    output_file = Path(output_dir) / 'torso_orientation.png'
    plot_multi_component_data(temp_data, 'torso_orientation',
                             ['Yaw', 'Roll', 'Pitch'],
                             'Angle (rad)', 'Torso Orientation (Yaw, Roll, Pitch)',
                             str(output_file),
                             colors=['blue', 'green', 'red'])


def plot_robot_quaternion(observations: List[Dict], output_dir):
    """Plot robot quaternion components over time."""
    output_file = Path(output_dir) / 'robot_quaternion.png'
    plot_multi_component_data(observations, 'robot_quat',
                             ['q_w', 'q_x', 'q_y', 'q_z'],
                             'Quaternion Component', 'Robot Quaternion [w, x, y, z]',
                             str(output_file),
                             colors=['blue', 'green', 'red', 'purple'])


def plot_angular_velocity(observations: List[Dict], output_dir):
    """Plot robot angular velocity components over time."""
    output_file = Path(output_dir) / 'angular_velocity.png'
    plot_multi_component_data(observations, 'ang_vel',
                             ['Roll rate (ω_x)', 'Pitch rate (ω_y)', 'Yaw rate (ω_z)'],
                             'Angular Vel (rad/s)', 'Robot Angular Velocity',
                             str(output_file),
                             colors=['blue', 'green', 'red'])


def plot_target_positions(actions: List[Dict], joint_names: List[str], output_dir):
    """Plot all 29 target positions from actions, each in its own subplot."""
    output_file = Path(output_dir) / 'target_positions.png'
    plot_multi_joint_data(actions, joint_names, 'target_pos_mj',
                         'Target Pos (rad)', 'Target Joint Positions (Policy Output)',
                         str(output_file), color='purple')


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
    parser.add_argument('--log-dir', type=str, default='./mujoco_logs',
                       help='Directory containing log files')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting (only print statistics)')
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config()
    mj2lab = np.array(config['mj2lab'], dtype=np.int32)
    joint_names = config['mj_joint_names']
    
    print(f"Loaded {len(joint_names)} joint names")
    print(f"mj2lab mapping: {mj2lab}")
    
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
    # compute_statistics(observations, actions)
    
    # Plotting
    if not args.no_plot:
        print("\n" + "="*60)
        print("Generating plots...")
        print("="*60)
        # Ensure analyse folder under log dir exists
        analyze_dir = Path(args.log_dir) / 'analyse'
        analyze_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. All joint positions (raw_qj)
        print("\n1. Plotting all joint positions...")
        plot_all_joint_positions(observations, joint_names, analyze_dir)
        
        # 2. All joint velocities (raw_qvel)
        print("2. Plotting all joint velocities...")
        plot_all_joint_velocities(observations, joint_names, analyze_dir)
        
        # 3. Torso orientation (yaw, roll, pitch)
        print("3. Plotting torso orientation...")
        plot_torso_orientation(observations, analyze_dir)
        
        # 4. Robot quaternion
        print("4. Plotting robot quaternion...")
        plot_robot_quaternion(observations, analyze_dir)
        
        # 5. Angular velocity
        print("5. Plotting angular velocity...")
        plot_angular_velocity(observations, analyze_dir)
        
        # 6. Target positions from actions
        print("6. Plotting target positions...")
        plot_target_positions(actions, joint_names, analyze_dir)
        
        print("\n" + "="*60)
        print("All plots saved successfully!")
        print("="*60)
        print("\nGenerated files in:")
        print(f"  {analyze_dir}")
        print("  - all_joint_positions.png")
        print("  - all_joint_velocities.png")
        print("  - torso_orientation.png")
        print("  - robot_quaternion.png")
        print("  - angular_velocity.png")
        print("  - target_positions.png")


if __name__ == "__main__":
    main()
