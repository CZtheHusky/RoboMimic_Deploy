"""
Example script for analyzing BeyondMimic logged data.

This script demonstrates how to load and analyze the observation and action data
logged during policy execution.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import sys
import yaml
from typing import List, Dict, Any, Optional, Tuple

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
    obs_files = sorted(log_path.glob("*observations_*.pkl"))
    action_files = sorted(log_path.glob("*actions_*.pkl"))
    
    if not obs_files or not action_files:
        print("No log files found")
        return None, None
    
    # Load latest files
    # Match latest session that has both files
    obs_map = {}
    for f in obs_files:
        stem = f.stem
        key = "observations_"
        idx = stem.rfind(key)
        if idx >= 0:
            sid = stem[idx+len(key):]
            obs_map[sid] = f
    act_map = {}
    for f in action_files:
        stem = f.stem
        key = "actions_"
        idx = stem.rfind(key)
        if idx >= 0:
            sid = stem[idx+len(key):]
            act_map[sid] = f
    session_ids = sorted(set(obs_map.keys()) & set(act_map.keys()))
    if not session_ids:
        print("No matching observation/action log pairs found")
        return None, None
    latest_sid = session_ids[-1]
    obs_file = obs_map[latest_sid]
    action_file = act_map[latest_sid]
    
    print(f"Loading observations from: {obs_file.name}")
    print(f"Loading actions from: {action_file.name}")
    
    observations = load_streaming_pickle(str(obs_file))
    actions = load_streaming_pickle(str(action_file))
    
    print(f"Loaded {len(observations)} observation samples")
    print(f"Loaded {len(actions)} action samples")
    
    return observations, actions


def load_all_logs(log_dir: str) -> List[Dict[str, Any]]:
    """
    Load all observation/action log pairs in a directory and group by session.

    Returns a list of dicts: { 'session_id', 'obs_file', 'action_file', 'observations', 'actions' }
    Only sessions that have both observations and actions are returned.
    """
    log_path = Path(log_dir)
    results: List[Dict[str, Any]] = []
    if not log_path.exists():
        print(f"Log directory not found: {log_dir}")
        return results

    obs_files = sorted(log_path.glob("*observations_*.pkl"))
    action_files = sorted(log_path.glob("*actions_*.pkl"))

    # Map session_id -> file (session_id may contain an underscore)
    obs_map = {}
    for f in obs_files:
        stem = f.stem
        key = "observations_"
        idx = stem.rfind(key)
        if idx >= 0:
            sid = stem[idx+len(key):]
            obs_map[sid] = f
    act_map = {}
    for f in action_files:
        stem = f.stem
        key = "actions_"
        idx = stem.rfind(key)
        if idx >= 0:
            sid = stem[idx+len(key):]
            act_map[sid] = f

    # Intersect sessions
    session_ids = sorted(set(obs_map.keys()) & set(act_map.keys()))
    if not session_ids:
        print("No matching observation/action log pairs found")
        return results
    min_len = 1e9
    for sid in session_ids:
        of = obs_map[sid]
        af = act_map[sid]
        stem = of.stem
        key = "observations_"
        idx = stem.rfind(key)
        pre = stem[:idx] if idx >= 0 else ""
        pre = pre.rstrip("_")
        label = f"{pre}-{sid}" if pre else sid
        try:
            observations = load_streaming_pickle(str(of))
            actions = load_streaming_pickle(str(af))
            print(f"Loaded session {sid}: obs={len(observations)}, actions={len(actions)}")
            min_len = min(min_len, len(observations), len(actions))
            results.append({
                'session_id': sid,
                'obs_file': of,
                'action_file': af,
                'observations': observations,
                'actions': actions,
                'label': label,
            })
        except Exception as e:
            print(f"Warning: failed to load session {sid}: {e}")
    print(f"Truncating all runs to minimum length: {min_len} samples")
    for i in range(len(results)):
        results[i]['observations'] = results[i]['observations'][:min_len]
        results[i]['actions'] = results[i]['actions'][:min_len]
    return results


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
    # Use counter_step as x-axis; fallback to sequential index if missing
    def _get_step(x):
        v = x.get('counter_step', None)
        if v is None:
            return None
        try:
            return int(np.atleast_1d(v)[0])
        except Exception:
            return None
    steps = np.array([_get_step(d) for d in data], dtype=object)
    if any(s is None for s in steps):
        steps = np.arange(values.shape[0])
    
    num_joints = values.shape[1]
    
    # Create subplots (6 rows x 5 columns for up to 30 joints)
    fig, axes = plt.subplots(6, 5, figsize=(20, 18))
    axes = axes.flatten()
    
    for i in range(num_joints):
        axes[i].plot(steps, values[:, i], linewidth=1.5, color=color)
        axes[i].set_title(joint_names[i], fontsize=9)
        axes[i].set_xlabel('Step', fontsize=8)
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


def plot_multi_joint_overlay(runs: List[Dict[str, Any]], source: str, joint_names: List[str], data_key: str,
                             ylabel: str, title: str, filename: str):
    """
    Overlay multiple runs' joint series on one figure (subplots per joint).
    source: 'observations' or 'actions'
    """
    if not runs:
        return

    # Determine number of joints from the first run
    first_values = np.array([runs[0][source][0][data_key]])
    num_joints = first_values.shape[1]

    fig, axes = plt.subplots(6, 5, figsize=(20, 18))
    axes = axes.flatten()

    # Color cycle
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(len(runs))]

    # Truncate all series to shortest length among runs
    lengths = [len(run[source]) for run in runs]
    if any(l <= 0 for l in lengths):
        print("One or more runs are empty; skipping overlay plot.")
        return
    min_len = min(lengths)
    if min_len <= 0:
        print("No data to plot in overlay.")
        return

    for ridx, run in enumerate(runs):
        data = run[source]
        values = np.array([d[data_key] for d in data[:min_len]])
        # Counter steps as x-axis; fallback to sequential index if missing
        def _get_step(x):
            v = x.get('counter_step', None)
            if v is None:
                return None
            try:
                return int(np.atleast_1d(v)[0])
            except Exception:
                return None
        steps = np.array([_get_step(d) for d in data[:min_len]], dtype=object)
        if any(s is None for s in steps):
            steps = np.arange(values.shape[0])
        for j in range(min(num_joints, len(axes))):
            axes[j].plot(steps, values[:, j], linewidth=1.2, color=colors[ridx], alpha=0.85)

    for j in range(num_joints):
        axes[j].set_title(joint_names[j], fontsize=9)
        axes[j].set_xlabel('Step', fontsize=8)
        axes[j].set_ylabel(ylabel, fontsize=8)
        axes[j].grid(True, alpha=0.3)
        axes[j].tick_params(labelsize=7)
    for j in range(num_joints, len(axes)):
        axes[j].axis('off')

    # Create a figure-level legend with session_ids
    try:
        labels = [run.get('label', run['session_id']) for run in runs]
        # Create invisible handles for legend
        handles = [Line2D([0], [0], color=colors[i], lw=2) for i in range(len(runs))]
        fig.legend(handles, labels, loc='upper center', ncol=min(len(runs), 5), fontsize=9)
    except Exception:
        pass

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    try:
        out_path = Path(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
        print(f"Saved overlay to: {out_path}")
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
    # Use counter_step as x-axis; fallback to sequential index if missing
    def _get_step(x):
        v = x.get('counter_step', None)
        if v is None:
            return None
        try:
            return int(np.atleast_1d(v)[0])
        except Exception:
            return None
    steps = np.array([_get_step(d) for d in data], dtype=object)
    if any(s is None for s in steps):
        steps = np.arange(values.shape[0])
    
    num_components = len(component_labels)
    
    if colors is None:
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown'][:num_components]
    
    fig, axes = plt.subplots(num_components, 1, figsize=(12, 3 * num_components + 1))
    
    # Handle single component case
    if num_components == 1:
        axes = [axes]
    
    for i in range(num_components):
        axes[i].plot(steps, values[:, i], label=component_labels[i], 
                    color=colors[i], linewidth=1.5)
        axes[i].set_ylabel(ylabel if isinstance(ylabel, str) else ylabel[i], fontsize=10)
        axes[i].set_title(f'{component_labels[i]} Over Time', fontsize=11)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    axes[-1].set_xlabel('Step', fontsize=10)
    
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


def plot_multi_component_overlay(runs: List[Dict[str, Any]], source: str, data_key: str, component_labels: List[str],
                                 ylabel: str, title: str, filename: str):
    """
    Overlay multiple runs for multi-component signals (e.g., yaw/roll/pitch, quat, ang vel).
    Creates one subplot per component, with a line per run.
    """
    if not runs:
        return

    num_components = len(component_labels)
    fig, axes = plt.subplots(num_components, 1, figsize=(12, 3 * num_components + 1))
    if num_components == 1:
        axes = [axes]

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(runs))]

    # Truncate all series to shortest length among runs
    lengths = [len(run[source]) for run in runs]
    if any(l <= 0 for l in lengths):
        print("One or more runs are empty; skipping overlay plot.")
        return
    min_len = min(lengths)
    if min_len <= 0:
        print("No data to plot in overlay.")
        return

    for ridx, run in enumerate(runs):
        data = run[source]
        values = np.array([d[data_key] for d in data[:min_len]])
        # Counter steps as x-axis; fallback to sequential index if missing
        def _get_step(x):
            v = x.get('counter_step', None)
            if v is None:
                return None
            try:
                return int(np.atleast_1d(v)[0])
            except Exception:
                return None
        steps = np.array([_get_step(d) for d in data[:min_len]], dtype=object)
        if any(s is None for s in steps):
            steps = np.arange(values.shape[0])
        for i in range(num_components):
            axes[i].plot(steps, values[:, i], linewidth=1.4, color=colors[ridx], alpha=0.9)

    for i in range(num_components):
        axes[i].set_ylabel(ylabel if isinstance(ylabel, str) else ylabel[i], fontsize=10)
        axes[i].set_title(f'{component_labels[i]} Over Time', fontsize=11)
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel('Step', fontsize=10)

    # Legend with session ids
    labels = [run.get('label', run['session_id']) for run in runs]
    handles = [Line2D([0], [0], color=colors[i], lw=2) for i in range(len(runs))]
    fig.legend(handles, labels, loc='upper center', ncol=min(len(runs), 5), fontsize=9)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    try:
        out_path = Path(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
        print(f"Saved overlay to: {out_path}")
    finally:
        plt.close()


def plot_joint_trajectories(observations: List[Dict], num_joints: int = 5):
    """Plot joint position and velocity trajectories."""
    if not observations:
        return
    
    # Extract data
    joint_pos = np.array([obs['joint_pos'] for obs in observations])
    joint_vel = np.array([obs['joint_vel'] for obs in observations])
    # Use counter_step as x-axis; fallback to sequential index
    steps = np.array([int(np.atleast_1d(obs.get('counter_step', [i]))[0]) for i, obs in enumerate(observations)])
    
    # Plot first few joints
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Joint positions
    for i in range(min(num_joints, joint_pos.shape[1])):
        axes[0].plot(steps, joint_pos[:, i], label=f'Joint {i}', alpha=0.7)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Joint Position (rad)')
    axes[0].set_title('Joint Positions Over Time')
    axes[0].legend()
    axes[0].grid(True)
    
    # Joint velocities
    for i in range(min(num_joints, joint_vel.shape[1])):
        axes[1].plot(steps, joint_vel[:, i], label=f'Joint {i}', alpha=0.7)
    axes[1].set_xlabel('Step')
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
    temp_data = [{'torso_orientation': combined[i], 'counter_step': int(np.atleast_1d(obs.get('counter_step', [i]))[0])} 
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


def plot_all_joint_positions_overlay(runs: List[Dict[str, Any]], joint_names: List[str], output_dir):
    output_file = Path(output_dir) / 'overlay_all_joint_positions.png'
    plot_multi_joint_overlay(runs, 'observations', joint_names, 'raw_qj',
                             'Position (rad)', 'All Joint Positions Over Time (All Runs)',
                             str(output_file))


def plot_all_joint_velocities_overlay(runs: List[Dict[str, Any]], joint_names: List[str], output_dir):
    output_file = Path(output_dir) / 'overlay_all_joint_velocities.png'
    plot_multi_joint_overlay(runs, 'observations', joint_names, 'raw_qvel',
                             'Velocity (rad/s)', 'All Joint Velocities Over Time (All Runs)',
                             str(output_file))


def plot_torso_orientation_overlay(runs: List[Dict[str, Any]], output_dir):
    # Build temporary arrays per run [yaw, roll, pitch]
    aug_runs = []
    for run in runs:
        obs = run['observations']
        yaw = np.array([o['base_troso_yaw'] for o in obs])
        roll = np.array([o['base_troso_roll'] for o in obs])
        pitch = np.array([o['base_troso_pitch'] for o in obs])
        combined = np.column_stack([yaw, roll, pitch])
        # Inject into a proxy key to reuse overlay plotter
        proxy = [{'counter_step': int(np.atleast_1d(o.get('counter_step', [i]))[0]), 'torso_orientation': combined[i]} for i, o in enumerate(obs)]
        aug_runs.append({'session_id': run['session_id'], 'observations': proxy})

    output_file = Path(output_dir) / 'overlay_torso_orientation.png'
    plot_multi_component_overlay(aug_runs, 'observations', 'torso_orientation',
                                 ['Yaw', 'Roll', 'Pitch'], 'Angle (rad)',
                                 'Torso Orientation (All Runs)', str(output_file))


def plot_robot_quaternion_overlay(runs: List[Dict[str, Any]], output_dir):
    output_file = Path(output_dir) / 'overlay_robot_quaternion.png'
    plot_multi_component_overlay(runs, 'observations', 'robot_quat',
                                 ['q_w', 'q_x', 'q_y', 'q_z'], 'Quaternion Component',
                                 'Robot Quaternion [w, x, y, z] (All Runs)', str(output_file))


def plot_angular_velocity_overlay(runs: List[Dict[str, Any]], output_dir):
    output_file = Path(output_dir) / 'overlay_angular_velocity.png'
    plot_multi_component_overlay(runs, 'observations', 'ang_vel',
                                 ['Roll rate (ω_x)', 'Pitch rate (ω_y)', 'Yaw rate (ω_z)'],
                                 'Angular Vel (rad/s)', 'Robot Angular Velocity (All Runs)',
                                 str(output_file))


def plot_target_positions_overlay(runs: List[Dict[str, Any]], joint_names: List[str], output_dir):
    output_file = Path(output_dir) / 'overlay_target_positions.png'
    plot_multi_joint_overlay(runs, 'actions', joint_names, 'target_pos_mj',
                             'Target Pos (rad)', 'Target Joint Positions (All Runs)',
                             str(output_file))


def plot_motion_anchor_overlay(runs: List[Dict[str, Any]], output_dir):
    """Overlay comparison for observation motion_anchor_ori components across runs."""
    output_file = Path(output_dir) / 'overlay_motion_anchor_ori.png'
    # motion_anchor_ori is a flattened 3x2 (first two columns), so 6 components
    labels = ['col0_x', 'col0_y', 'col0_z', 'col1_x', 'col1_y', 'col1_z']
    plot_multi_component_overlay(runs, 'observations', 'motion_anchor_ori',
                                 labels, 'Value', 'Motion Anchor Orientation Components (All Runs)',
                                 str(output_file))


def plot_raw_actions_overlay(runs: List[Dict[str, Any]], joint_names_lab: List[str], output_dir):
    """Overlay raw_action (29D) across runs, labeled with joint names (lab order)."""
    output_file = Path(output_dir) / 'overlay_raw_actions.png'
    plot_multi_joint_overlay(runs, 'actions', joint_names_lab, 'raw_action',
                             'Raw Action', 'Raw Actions (All Runs)',
                             str(output_file))


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
    parser.add_argument('--compare-all', action='store_true',
                       help='Overlay and compare all runs in the directory')
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config()
    mj2lab = np.array(config['mj2lab'], dtype=np.int32)
    joint_names = config['mj_joint_names']
    
    print(f"Loaded {len(joint_names)} joint names")
    print(f"mj2lab mapping: {mj2lab}")
    
    analyze_dir = Path(args.log_dir) / 'analyse'
    analyze_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_all:
        # Load all runs and overlay
        runs = load_all_logs(args.log_dir)
        if not runs:
            print("No runs to compare.")
            return
        if not args.no_plot:
            print("\n" + "="*60)
            print("Generating overlay plots for all runs...")
            print("="*60)
            plot_all_joint_positions_overlay(runs, joint_names, analyze_dir)
            plot_all_joint_velocities_overlay(runs, joint_names, analyze_dir)
            # motion_anchor_ori components
            plot_motion_anchor_overlay(runs, analyze_dir)
            plot_torso_orientation_overlay(runs, analyze_dir)
            plot_robot_quaternion_overlay(runs, analyze_dir)
            plot_angular_velocity_overlay(runs, analyze_dir)
            plot_target_positions_overlay(runs, joint_names, analyze_dir)
            # raw actions overlay (lab order names)
            # Build lab-order joint names using mj2lab mapping
            joint_names_lab = [joint_names[idx] for idx in mj2lab]
            plot_raw_actions_overlay(runs, joint_names_lab, analyze_dir)
            print("\nOverlay plots saved in:")
            print(f"  {analyze_dir}")
        return

    # Default: load latest only
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
