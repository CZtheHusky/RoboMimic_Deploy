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
    return results


def enforce_length_ratio(
    runs: List[Dict[str, Any]],
    source: str,
    max_ratio: float = 2.0,
) -> None:
    """
    Ensure that within a group of runs, the longest sequence length for the given
    source key does not exceed max_ratio * shortest_length.
    If 2 * min_len >= max_len, no truncation is applied.
    """
    if not runs:
        return

    lengths = []
    for run in runs:
        data = run.get(source, None)
        if data:
            lengths.append(len(data))

    if not lengths:
        return

    min_len = min(lengths)
    max_len = max(lengths)

    if min_len <= 0:
        return

    allowed_max = int(max_ratio * min_len)

    # If allowed_max already covers the longest sequence, nothing to do
    if allowed_max >= max_len:
        return

    print(
        f"Truncating '{source}' runs to at most {allowed_max} samples "
        f"(min={min_len}, original max={max_len})"
    )

    for run in runs:
        seq = run.get(source, None)
        if not seq:
            continue
        if len(seq) > allowed_max:
            run[source] = seq[:allowed_max]


def extract_steps(data: List[Dict[str, Any]]) -> np.ndarray:
    """
    Extract step indices from a sequence of dicts using the 'counter_step' key.
    Falls back to sequential indices [0..N-1] if anything looks invalid.
    """
    if not data:
        return np.array([], dtype=int)

    steps: List[int] = []
    valid = True
    for i, d in enumerate(data):
        if not isinstance(d, dict):
            valid = False
            break
        v = d.get("counter_step", None)
        if v is None:
            valid = False
            break
        try:
            steps.append(int(np.atleast_1d(v)[0]))
        except Exception:
            valid = False
            break

    if not valid or not steps:
        return np.arange(len(data), dtype=int)

    return np.array(steps, dtype=int)


def get_run_color(
    label: str,
    idx: int,
    color_map: Optional[Dict[str, Any]],
    cmap_name: str,
):
    """
    Resolve a color for a given run label:
    - Use color_map[label] if available (e.g., fixed red for sim runs).
    - Otherwise pick from the given matplotlib colormap using idx.
    """
    if color_map is not None:
        c = color_map.get(label, None)
        if c is not None:
            return c
    cmap = plt.get_cmap(cmap_name)
    # colormap objects usually have an 'N' attribute (number of base colors)
    n = getattr(cmap, "N", 10)
    return cmap(idx % n)


def analyze_timing(
    data: List[Dict[str, Any]],
    label: Optional[str] = None,
    stats_target: Optional[Dict[str, Any]] = None,
    stats_key: Optional[str] = None,
):
    """
    Analyze timing statistics of logged data.

    If stats_target and stats_key are provided, store the interval statistics
    into stats_target[stats_key] as a dict, in addition to printing them.
    Returns (relative_time, intervals, summary_dict).
    """
    if not data:
        return None, None, None
    
    timestamps = np.array([d['_timestamp'] for d in data])
    relative_time = timestamps - timestamps[0]
    
    # Calculate intervals
    intervals = np.diff(timestamps)

    header = "=== Timing Analysis ==="
    if label:
        header = f"=== Timing Analysis: {label} ==="
    print(f"\n{header}")
    print(f"Total duration: {relative_time[-1]:.2f} seconds")
    print(f"Number of samples: {len(data)}")
    avg_interval_ms = float(np.mean(intervals) * 1000.0)
    std_interval_ms = float(np.std(intervals) * 1000.0)
    min_interval_ms = float(np.min(intervals) * 1000.0)
    max_interval_ms = float(np.max(intervals) * 1000.0)
    avg_freq_hz = float(1.0 / np.mean(intervals)) if len(intervals) > 0 else 0.0
    print(f"Average interval: {avg_interval_ms:.2f} ms")
    print(f"Std dev interval: {std_interval_ms:.2f} ms")
    print(f"Min interval: {min_interval_ms:.2f} ms")
    print(f"Max interval: {max_interval_ms:.2f} ms")
    print(f"Average frequency: {avg_freq_hz:.2f} Hz")

    summary = {
        "total_duration_sec": float(relative_time[-1]),
        "num_samples": int(len(data)),
        "average_interval_ms": avg_interval_ms,
        "std_interval_ms": std_interval_ms,
        "min_interval_ms": min_interval_ms,
        "max_interval_ms": max_interval_ms,
        "average_frequency_hz": avg_freq_hz,
    }

    if stats_target is not None and stats_key is not None:
        stats_target[stats_key] = summary
    
    return relative_time, intervals, summary


def has_detailed_timestamps(data: List[Dict[str, Any]]) -> bool:
    """
    Check whether observation data contains the newer, more detailed
    timing fields (inf_start / inf_end).
    """
    if not data:
        return False
    first = data[0]
    if not isinstance(first, dict):
        return False
    return ("inf_start" in first) and ("inf_end" in first)


def analyze_detailed_timing(
    obs_data: List[Dict[str, Any]],
    label: str,
    stats_target: Dict[str, Any],
) -> Optional[Dict[str, np.ndarray]]:
    """
    Analyze detailed per-step timing based on the newer fields recorded in
    observations:
      - inf_start -> inf_end: total time spent inside the BeyondMimic.run step
      - before_nn -> after_nn: neural network forward pass time
      - inf_end(t) -> inf_start(t+1): external time between two control steps

    Stores summary statistics (in milliseconds) into stats_target['detailed'] and
    returns a dict mapping
        { 'step_duration': np.ndarray[ms],
          'nn_duration':   np.ndarray[ms],
          'outer_duration': np.ndarray[ms] }
    for use in plotting. Any missing series will be absent in the dict.
    """
    if not obs_data:
        return None

    # Collect raw timestamps as float arrays
    inf_start = np.array([d.get("inf_start", np.nan) for d in obs_data], dtype=np.float64)
    inf_end = np.array([d.get("inf_end", np.nan) for d in obs_data], dtype=np.float64)
    before_nn = np.array([d.get("before_nn", np.nan) for d in obs_data], dtype=np.float64)
    after_nn = np.array([d.get("after_nn", np.nan) for d in obs_data], dtype=np.float64)

    # Basic sanity: require finite values, otherwise skip detailed analysis
    if not (np.isfinite(inf_start).all() and np.isfinite(inf_end).all()):
        print(f"Warning: {label}: inf_start/inf_end contain non-finite values, skipping detailed timing.")
        return None

    # Per-step durations (seconds)
    step_duration = inf_end - inf_start
    nn_duration = None
    if np.isfinite(before_nn).all() and np.isfinite(after_nn).all():
        nn_duration = after_nn - before_nn

    outer_duration = None
    if len(inf_start) > 1:
        outer_duration = inf_start[1:] - inf_end[:-1]

    def _summarize(name: str, arr: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
        if arr is None or len(arr) == 0:
            return None
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        # Convert to milliseconds for reporting
        arr_ms = arr * 1000.0
        return {
            "mean_ms": float(np.mean(arr_ms)),
            "std_ms": float(np.std(arr_ms)),
            "min_ms": float(np.min(arr_ms)),
            "max_ms": float(np.max(arr_ms)),
        }

    detailed_stats: Dict[str, Any] = {}
    step_stats = _summarize("step_duration", step_duration)
    if step_stats is not None:
        detailed_stats["step_duration"] = step_stats
    nn_stats = _summarize("nn_duration", nn_duration)
    if nn_stats is not None:
        detailed_stats["nn_duration"] = nn_stats
    outer_stats = _summarize("outer_duration", outer_duration)
    if outer_stats is not None:
        detailed_stats["outer_duration"] = outer_stats

    # Build ms-series for plotting (only finite values are kept)
    series: Dict[str, np.ndarray] = {}
    if step_duration is not None and len(step_duration) > 0:
        series["step_duration"] = step_duration * 1000.0
    if nn_duration is not None and len(nn_duration) > 0:
        series["nn_duration"] = nn_duration * 1000.0
    if outer_duration is not None and len(outer_duration) > 0:
        series["outer_duration"] = outer_duration * 1000.0

    if detailed_stats:
        print(f"\n=== Detailed Timing ({label}) ===")
        for k, v in detailed_stats.items():
            print(
                f"{k}: mean={v['mean_ms']:.2f} ms, std={v['std_ms']:.2f} ms, "
                f"min={v['min_ms']:.2f} ms, max={v['max_ms']:.2f} ms"
            )
        stats_target["detailed"] = detailed_stats

    return series if series else None


def plot_timing_overview(
    analyze_dir: Path,
    color_map: Dict[str, Any],
    interval_series: List[Tuple[str, np.ndarray]],
    step_series: List[Tuple[str, np.ndarray]],
    nn_series: List[Tuple[str, np.ndarray]],
    outer_series: List[Tuple[str, np.ndarray]],
) -> None:
    """
    Plot a 4x1 timing overview figure:
      1) Action sample intervals (from _timestamp on actions)
      2) Step duration (inf_start -> inf_end)
      3) NN forward duration (before_nn -> after_nn)
      4) Outer-loop duration (inf_end[t] -> inf_start[t+1])

    Any metric with no data will have its subplot hidden.
    """
    if not (interval_series or step_series or nn_series or outer_series):
        return

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    metric_rows = [
        ("Action sample intervals over runs", interval_series, "Interval (ms)"),
        ("Step duration (run() total)", step_series, "Time (ms)"),
        ("NN forward duration", nn_series, "Time (ms)"),
        ("Outer-loop duration (between steps)", outer_series, "Time (ms)"),
    ]

    for row_idx, (title, series_list, ylabel) in enumerate(metric_rows):
        ax = axes[row_idx]
        if not series_list:
            ax.axis("off")
            continue

        for s_idx, (label, arr) in enumerate(series_list):
            # Action intervals are in seconds; convert to ms here.
            if row_idx == 0:
                arr_ms = arr * 1000.0
            else:
                arr_ms = arr  # already stored in ms by analyze_detailed_timing

            if len(arr_ms) == 0:
                continue
            x = np.arange(len(arr_ms))
            c = get_run_color(label, s_idx, color_map, cmap_name="tab10")
            ax.plot(x, arr_ms, label=label, color=c, linewidth=1.4, alpha=0.9)

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step index")

    # Put a legend once, using the first non-empty axis
    for row_idx, (title, series_list, _) in enumerate(metric_rows):
        if not series_list:
            continue
        ax = axes[row_idx]
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center',
                       ncol=min(len(labels), 5), fontsize=8)
            break

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    timing_plot_path = analyze_dir / "timing_overview.png"
    try:
        timing_plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(timing_plot_path), dpi=150, bbox_inches='tight')
        print(f"Saved timing overview plot to: {timing_plot_path}")
    finally:
        plt.close(fig)

def plot_multi_joint_overlay(runs: List[Dict[str, Any]], source: str, joint_names: List[str], data_key: str,
                             ylabel: str, title: str, filename: str,
                             color_map: Optional[Dict[str, Any]] = None):
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

    lengths = [len(run[source]) for run in runs]
    if any(l <= 0 for l in lengths):
        print("One or more runs are empty; skipping overlay plot.")
        return

    for ridx, run in enumerate(runs):
        data = run[source]
        if not data:
            continue
        values = np.array([d[data_key] for d in data])
        steps = extract_steps(data)
        # Choose color via shared helper (uses color_map if provided)
        label = run.get('label', run['session_id'])
        c = get_run_color(label, ridx, color_map, cmap_name="tab20")
        for j in range(min(num_joints, len(axes))):
            axes[j].plot(steps, values[:, j], linewidth=1.2, color=c, alpha=0.85)

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
        handles = []
        for idx, run in enumerate(runs):
            label = run.get('label', run['session_id'])
            c = get_run_color(label, idx, color_map, cmap_name="tab20")
            handles.append(Line2D([0], [0], color=c, lw=2))
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
    

def plot_multi_component_overlay(runs: List[Dict[str, Any]], source: str, data_key: str, component_labels: List[str],
                                 ylabel: str, title: str, filename: str,
                                 color_map: Optional[Dict[str, Any]] = None):
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

    lengths = [len(run[source]) for run in runs]
    if any(l <= 0 for l in lengths):
        print("One or more runs are empty; skipping overlay plot.")
        return

    for ridx, run in enumerate(runs):
        data = run[source]
        if not data:
            continue
        values = np.array([d[data_key] for d in data])
        steps = extract_steps(data)
        label = run.get('label', run['session_id'])
        c = get_run_color(label, ridx, color_map, cmap_name="tab10")
        for i in range(num_components):
            axes[i].plot(steps, values[:, i], linewidth=1.4, color=c, alpha=0.9)

    for i in range(num_components):
        axes[i].set_ylabel(ylabel if isinstance(ylabel, str) else ylabel[i], fontsize=10)
        axes[i].set_title(f'{component_labels[i]} Over Time', fontsize=11)
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel('Step', fontsize=10)

    # Legend with session ids
    labels = [run.get('label', run['session_id']) for run in runs]
    handles = []
    for idx, run in enumerate(runs):
        label = run.get('label', run['session_id'])
        c = get_run_color(label, idx, color_map, cmap_name="tab10")
        handles.append(Line2D([0], [0], color=c, lw=2))
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


def plot_all_joint_positions_overlay(runs: List[Dict[str, Any]], joint_names: List[str], output_dir,
                                     color_map: Optional[Dict[str, Any]] = None):
    output_file = Path(output_dir) / 'overlay_all_joint_positions.png'
    plot_multi_joint_overlay(runs, 'observations', joint_names, 'raw_qj',
                             'Position (rad)', 'All Joint Positions Over Time (All Runs)',
                             str(output_file), color_map=color_map)


def plot_all_joint_velocities_overlay(runs: List[Dict[str, Any]], joint_names: List[str], output_dir,
                                      color_map: Optional[Dict[str, Any]] = None):
    output_file = Path(output_dir) / 'overlay_all_joint_velocities.png'
    plot_multi_joint_overlay(runs, 'observations', joint_names, 'raw_qvel',
                             'Velocity (rad/s)', 'All Joint Velocities Over Time (All Runs)',
                             str(output_file), color_map=color_map)


def plot_torso_orientation_overlay(runs: List[Dict[str, Any]], output_dir,
                                   color_map: Optional[Dict[str, Any]] = None):
    # Build temporary arrays per run [yaw, roll, pitch]
    aug_runs = []
    for run in runs:
        obs = run['observations']
        yaw = np.array([o['base_troso_yaw'] for o in obs])
        roll = np.array([o['base_troso_roll'] for o in obs])
        pitch = np.array([o['base_troso_pitch'] for o in obs])
        combined = np.column_stack([yaw, roll, pitch])
        steps = extract_steps(obs)
        # Inject into a proxy key to reuse overlay plotter
        proxy = [
            {'counter_step': steps[i], 'torso_orientation': combined[i]}
            for i in range(len(obs))
        ]
        aug_runs.append({'session_id': run['session_id'], 'observations': proxy})

    output_file = Path(output_dir) / 'overlay_torso_orientation.png'
    plot_multi_component_overlay(aug_runs, 'observations', 'torso_orientation',
                                 ['Yaw', 'Roll', 'Pitch'], 'Angle (rad)',
                                 'Torso Orientation (All Runs)', str(output_file),
                                 color_map=color_map)


def plot_robot_quaternion_overlay(runs: List[Dict[str, Any]], output_dir,
                                  color_map: Optional[Dict[str, Any]] = None):
    output_file = Path(output_dir) / 'overlay_robot_quaternion.png'
    plot_multi_component_overlay(runs, 'observations', 'robot_quat',
                                 ['q_w', 'q_x', 'q_y', 'q_z'], 'Quaternion Component',
                                 'Robot Quaternion [w, x, y, z] (All Runs)', str(output_file),
                                 color_map=color_map)


def plot_angular_velocity_overlay(runs: List[Dict[str, Any]], output_dir,
                                  color_map: Optional[Dict[str, Any]] = None):
    output_file = Path(output_dir) / 'overlay_angular_velocity.png'
    plot_multi_component_overlay(runs, 'observations', 'ang_vel',
                                 ['Roll rate (ω_x)', 'Pitch rate (ω_y)', 'Yaw rate (ω_z)'],
                                 'Angular Vel (rad/s)', 'Robot Angular Velocity (All Runs)',
                                 str(output_file), color_map=color_map)


def plot_target_positions_overlay(runs: List[Dict[str, Any]], joint_names: List[str], output_dir,
                                  color_map: Optional[Dict[str, Any]] = None):
    output_file = Path(output_dir) / 'overlay_target_positions.png'
    plot_multi_joint_overlay(runs, 'actions', joint_names, 'target_pos_mj',
                             'Target Pos (rad)', 'Target Joint Positions (All Runs)',
                             str(output_file), color_map=color_map)


def plot_motion_anchor_overlay(runs: List[Dict[str, Any]], output_dir,
                               color_map: Optional[Dict[str, Any]] = None):
    """Overlay comparison for observation motion_anchor_ori components across runs."""
    output_file = Path(output_dir) / 'overlay_motion_anchor_ori.png'
    # motion_anchor_ori is a flattened 3x2 (first two columns), so 6 components
    labels = ['col0_x', 'col0_y', 'col0_z', 'col1_x', 'col1_y', 'col1_z']
    plot_multi_component_overlay(runs, 'observations', 'motion_anchor_ori',
                                 labels, 'Value', 'Motion Anchor Orientation Components (All Runs)',
                                 str(output_file), color_map=color_map)


def plot_raw_actions_overlay(runs: List[Dict[str, Any]], joint_names_lab: List[str], output_dir,
                             color_map: Optional[Dict[str, Any]] = None):
    """Overlay raw_action (29D) across runs, labeled with joint names (lab order)."""
    output_file = Path(output_dir) / 'overlay_raw_actions.png'
    plot_multi_joint_overlay(runs, 'actions', joint_names_lab, 'raw_action',
                             'Raw Action', 'Raw Actions (All Runs)',
                             str(output_file), color_map=color_map)


def _compute_error_stats(deltas: np.ndarray, joint_names: List[str]) -> Dict[str, Any]:
    """
    Compute overall and per-joint statistics for tracking errors.
    deltas: shape (T, J)
    """
    stats: Dict[str, Any] = {}
    if deltas.size == 0:
        return stats

    flat = deltas.reshape(-1)
    abs_flat = np.abs(flat)

    stats["mean"] = float(np.mean(flat))
    stats["std"] = float(np.std(flat))
    stats["min"] = float(np.min(flat))
    stats["max"] = float(np.max(flat))
    stats["mean_abs"] = float(np.mean(abs_flat))
    stats["max_abs"] = float(np.max(abs_flat))

    per_joint: Dict[str, Any] = {}
    for j, name in enumerate(joint_names):
        col = deltas[:, j]
        abs_col = np.abs(col)
        per_joint[name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "mean_abs": float(np.mean(abs_col)),
            "max_abs": float(np.max(abs_col)),
        }
    stats["per_joint"] = per_joint
    return stats


def analyze_tracking_error(
    runs: List[Dict[str, Any]],
    joint_names: List[str],
    output_dir: Path,
    color_map: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Compute and plot tracking error:
      delta(t, j) = raw_qj[t+1, j] - target_pos_mj[t, j]
    for each run, then:
      - Save an overlay plot over all runs (per joint subplots)
      - Save overall and per-run statistics to YAML.
    """
    if not runs:
        return

    num_joints = len(joint_names)
    tracking_runs: List[Dict[str, Any]] = []
    all_deltas: List[np.ndarray] = []
    run_stats_list: List[Dict[str, Any]] = []

    for run in runs:
        obs = run.get("observations", None)
        acts = run.get("actions", None)
        if not obs or not acts:
            continue

        # Need obs[t+1] and acts[t] pairs
        max_t = min(len(acts), len(obs) - 1)
        if max_t <= 0:
            continue

        deltas = []
        steps = []
        for t in range(max_t):
            o_next = obs[t + 1]
            a_t = acts[t]
            try:
                q_next = np.array(o_next["raw_qj"], dtype=np.float32)
                target_t = np.array(a_t["target_pos_mj"], dtype=np.float32)
                if q_next.shape[0] != num_joints or target_t.shape[0] != num_joints:
                    continue
                deltas.append(q_next - target_t)
                # Prefer action counter_step as the reference step index
                cs = a_t.get("counter_step", t)
                try:
                    steps.append(int(np.atleast_1d(cs)[0]))
                except Exception:
                    steps.append(t)
            except KeyError:
                continue

        if not deltas:
            continue

        deltas_arr = np.vstack(deltas)  # shape (T, J)
        all_deltas.append(deltas_arr)

        # Stats for this run
        run_label = run.get("label", run.get("session_id", "unknown"))
        session_id = run.get("session_id", "")
        stats = _compute_error_stats(deltas_arr, joint_names)
        stats["label"] = run_label
        stats["session_id"] = session_id
        run_stats_list.append(stats)

        # Build proxy structure for overlay plotting
        tracking_series = [
            {"counter_step": steps[i], "tracking_delta": deltas_arr[i]}
            for i in range(deltas_arr.shape[0])
        ]
        tracking_runs.append(
            {
                "session_id": session_id,
                "label": run_label,
                "tracking": tracking_series,
            }
        )

    if not tracking_runs or not all_deltas:
        print("No valid tracking error data to analyze.")
        return

    # Overall statistics across all runs
    all_concat = np.vstack(all_deltas)
    overall_stats = _compute_error_stats(all_concat, joint_names)

    # Organize per-run stats as a dict keyed by run label (preferred),
    # falling back to session_id if needed.
    run_stats_map: Dict[str, Any] = {}
    for s in run_stats_list:
        key = s.get("label") or s.get("session_id") or f"run_{len(run_stats_map)}"
        run_stats_map[key] = s

    tracking_stats: Dict[str, Any] = {
        "overall": overall_stats,
        "runs": run_stats_map,
    }

    # Save statistics to YAML
    stats_path = output_dir / "tracking_error_stats.yaml"
    try:
        with open(stats_path, "w") as f:
            yaml.safe_dump(tracking_stats, f, sort_keys=False)
        print(f"Tracking error statistics saved to: {stats_path}")
    except Exception as e:
        print(f"Warning: failed to write tracking error statistics YAML: {e}")

    # Plot overlay of tracking error per joint
    output_file = output_dir / "overlay_tracking_error.png"
    plot_multi_joint_overlay(
        tracking_runs,
        source="tracking",
        joint_names=joint_names,
        data_key="tracking_delta",
        ylabel="Tracking Error (rad)",
        title="Tracking Error raw_qj[t+1] - target_pos_mj[t] (All Runs)",
        filename=str(output_file),
        color_map=color_map,
    )

def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze BeyondMimic logged data')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Base directory containing log files '
                            '(sim_* files in this dir, real_* files in robot_{robot_id}/)')
    parser.add_argument('--robot-id', type=int, default=None,
                       help='Robot id to compare against sim logs '
                            '(reads from {log_dir}/robot_{robot_id})')
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config()
    mj2lab = np.array(config['mj2lab'], dtype=np.int32)
    joint_names = config['mj_joint_names']
    
    print(f"Loaded {len(joint_names)} joint names")
    print(f"mj2lab mapping: {mj2lab}")
    
    base_log_path = Path(args.log_dir)

    # sim_* logs live directly under base_log_path
    sim_dir = base_log_path
    # real_* logs for this robot live under base_log_path / robot_{robot_id}
    robot_dir = base_log_path / f"robot_{args.robot_id}"
    analyze_dir = robot_dir / 'analyse'
    analyze_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading sim logs from: {sim_dir}")
    print(f"Loading robot {args.robot_id} logs from: {robot_dir}")

    sim_runs = load_all_logs(sim_dir)
    robot_runs = load_all_logs(robot_dir)

    if not sim_runs:
        print("No sim runs found in base log dir.")
    if not robot_runs:
        print(f"No robot runs found for robot_{args.robot_id}.")
    if not sim_runs or not robot_runs:
        print("Nothing to compare (need both sim_* and real_* runs).")
        return

    # Assign stable, descriptive labels to runs (used consistently in stats,
    # timing plots, and all overlay figures).
    for r in sim_runs:
        base_label = r.get('label', r['session_id'])
        # Ensure we have a clear "sim-" prefix, but avoid double-prefixing.
        if isinstance(base_label, str) and base_label.startswith("sim-"):
            r['label'] = base_label
        else:
            r['label'] = f"sim-{base_label}"
    for r in robot_runs:
        base_label = r.get('label', r['session_id'])
        r['label'] = f"robot{args.robot_id}-{base_label}"

    # After both sim and real runs are labeled, treat them as a single group
    # and enforce that the longest sequence is at most 2x the shortest one.
    all_runs = sim_runs + robot_runs
    # enforce_length_ratio(all_runs, 'observations', max_ratio=2.0)
    # enforce_length_ratio(all_runs, 'actions', max_ratio=2.0)

    # Per-run timing analysis and interval aggregation (for observations/actions)
    timing_stats: Dict[str, Dict[str, Any]] = {}
    interval_series: List[Tuple[str, np.ndarray]] = []
    # For detailed timing (new logs): store per-run series in milliseconds
    step_series: List[Tuple[str, np.ndarray]] = []
    nn_series: List[Tuple[str, np.ndarray]] = []
    outer_series: List[Tuple[str, np.ndarray]] = []

    for run in all_runs:
        run_label = run.get('label', run['session_id'])
        if run_label not in timing_stats:
            timing_stats[run_label] = {}

        # Observations timing
        obs = run.get('observations', None)
        if obs:
            # If this run has the newer detailed timing fields, analyze them
            if has_detailed_timestamps(obs):
                detailed = analyze_detailed_timing(
                    obs,
                    label=f"{run_label} (observations)",
                    stats_target=timing_stats[run_label],
                )
                if detailed is not None:
                    if "step_duration" in detailed:
                        step_series.append((run_label, detailed["step_duration"]))
                    if "nn_duration" in detailed:
                        nn_series.append((run_label, detailed["nn_duration"]))
                    if "outer_duration" in detailed:
                        outer_series.append((run_label, detailed["outer_duration"]))

            analyze_timing(
                obs,
                label=f"{run_label} (observations)",
                stats_target=timing_stats[run_label],
                stats_key='observations',
            )

        # Actions timing (also used for interval plot)
        acts = run.get('actions', None)
        if acts:
            _, intervals_act, _ = analyze_timing(
                acts,
                label=f"{run_label} (actions)",
                stats_target=timing_stats[run_label],
                stats_key='actions',
            )
            if intervals_act is not None and len(intervals_act) > 0:
                interval_series.append((run_label, intervals_act))

    # Save timing statistics to YAML in analyse directory
    timing_yaml_path = analyze_dir / "timing_stats.yaml"
    try:
        with open(timing_yaml_path, "w") as f:
            yaml.safe_dump(timing_stats, f, sort_keys=False)
        print(f"\nTiming statistics saved to: {timing_yaml_path}")
    except Exception as e:
        print(f"Warning: failed to write timing statistics YAML: {e}")

    # Build a consistent color map for all runs: sim_* in red, others auto.
    def _is_sim_run(run: Dict[str, Any]) -> bool:
        obs_file = run.get('obs_file', None)
        name = ""
        if isinstance(obs_file, Path):
            name = obs_file.name
        elif isinstance(obs_file, str):
            name = Path(obs_file).name
        return name.startswith("sim_")

    color_map: Dict[str, Any] = {}
    # Non-sim runs get distinct colors from a colormap
    base_cmap = plt.get_cmap('tab10')
    color_idx = 0
    for run in all_runs:
        label = run.get('label', run['session_id'])
        if label in color_map:
            continue
        if _is_sim_run(run):
            color_map[label] = 'red'
        else:
            color_map[label] = base_cmap(color_idx % 10)
            color_idx += 1

    # Unified timing overview figure (action intervals + detailed metrics)
    plot_timing_overview(
        analyze_dir=analyze_dir,
        color_map=color_map,
        interval_series=interval_series,
        step_series=step_series,
        nn_series=nn_series,
        outer_series=outer_series,
    )

    # Prepare runs list for overlay plots (shallow copies to avoid mutation).
    runs = [dict(r) for r in all_runs]

    # Tracking error analysis: raw_qj[t+1] - target_pos_mj[t]
    print("\n" + "="*60)
    print("Analyzing tracking error (raw_qj[t+1] - target_pos_mj[t])...")
    print("="*60)
    analyze_tracking_error(runs, joint_names, analyze_dir, color_map=color_map)

    print("\n" + "="*60)
    print(f"Generating overlay plots for sim vs robot_{args.robot_id}...")
    print("="*60)
    # Pass the shared color_map so that sim runs stay red
    # and each non-sim run keeps a consistent color across figures.
    plot_all_joint_positions_overlay(runs, joint_names, analyze_dir, color_map=color_map)
    plot_all_joint_velocities_overlay(runs, joint_names, analyze_dir, color_map=color_map)
    # motion_anchor_ori components
    plot_motion_anchor_overlay(runs, analyze_dir, color_map=color_map)
    plot_torso_orientation_overlay(runs, analyze_dir, color_map=color_map)
    plot_robot_quaternion_overlay(runs, analyze_dir, color_map=color_map)
    plot_angular_velocity_overlay(runs, analyze_dir, color_map=color_map)
    plot_target_positions_overlay(runs, joint_names, analyze_dir, color_map=color_map)
    # raw actions overlay (lab order names)
    joint_names_lab = [joint_names[idx] for idx in mj2lab]
    plot_raw_actions_overlay(runs, joint_names_lab, analyze_dir, color_map=color_map)
    print("\nOverlay plots saved in:")
    print(f"  {analyze_dir}")

if __name__ == "__main__":
    main()
