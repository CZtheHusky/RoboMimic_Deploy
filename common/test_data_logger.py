"""
Test script for the data logging functionality.

This script tests the StreamingPickleLogger independently to ensure it works correctly.
"""

import numpy as np
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))
from common.data_logger import StreamingPickleLogger, load_streaming_pickle


def test_basic_logging():
    """Test basic logging functionality."""
    print("=== Test 1: Basic Logging ===")
    
    # Create logger
    logger = StreamingPickleLogger(
        log_dir="./test_logs",
        batch_size=10,
        flush_interval=2.0,
        max_queue_size=1000
    )
    
    logger.start()
    
    # Simulate logging data
    num_samples = 50
    for i in range(num_samples):
        obs_data = {
            "joint_pos": np.random.randn(29).astype(np.float32),
            "joint_vel": np.random.randn(29).astype(np.float32),
            "imu_quat": np.random.randn(4).astype(np.float32),
            "step": np.array([i], dtype=np.int32),
        }
        
        action_data = {
            "target_pos": np.random.randn(29).astype(np.float32),
            "kp": np.ones(29).astype(np.float32) * 100,
            "kd": np.ones(29).astype(np.float32) * 5,
            "step": np.array([i], dtype=np.int32),
        }
        
        logger.log("observations", obs_data)
        logger.log("actions", action_data)
        
        time.sleep(0.02)  # Simulate 50Hz control
        
        if i % 10 == 0:
            stats = logger.get_stats()
            print(f"Step {i}: queue_size={stats['queue_size']}, "
                  f"buffers={stats['buffer_sizes']}")
    
    # Stop logger
    print("\nStopping logger...")
    logger.stop()
    
    final_stats = logger.get_stats()
    print(f"Final stats: {final_stats}")
    print("Test 1 completed!\n")
    
    return logger.log_dir, logger.session_id


def test_data_loading(log_dir, session_id):
    """Test loading logged data."""
    print("=== Test 2: Data Loading ===")
    
    log_path = Path(log_dir)
    
    # Find log files
    obs_file = log_path / f"observations_{session_id}.pkl"
    action_file = log_path / f"actions_{session_id}.pkl"
    
    print(f"Loading from: {obs_file}")
    print(f"Loading from: {action_file}")
    
    # Load data
    observations = load_streaming_pickle(str(obs_file))
    actions = load_streaming_pickle(str(action_file))
    
    print(f"Loaded {len(observations)} observation samples")
    print(f"Loaded {len(actions)} action samples")
    
    # Verify data
    if observations:
        first_obs = observations[0]
        print(f"\nFirst observation keys: {list(first_obs.keys())}")
        print(f"Joint positions shape: {np.array(first_obs['joint_pos']).shape}")
        print(f"Has timestamp: {'_timestamp' in first_obs}")
    
    if actions:
        first_action = actions[0]
        print(f"\nFirst action keys: {list(first_action.keys())}")
        print(f"Target positions shape: {np.array(first_action['target_pos']).shape}")
    
    print("Test 2 completed!\n")


def test_high_frequency():
    """Test with high-frequency data (100Hz)."""
    print("=== Test 3: High-Frequency Logging (100Hz) ===")
    
    logger = StreamingPickleLogger(
        log_dir="./test_logs_high_freq",
        batch_size=100,  # Write every 100 samples (1 second at 100Hz)
        flush_interval=5.0,
        max_queue_size=2000
    )
    
    logger.start()
    
    num_samples = 500
    start_time = time.time()
    
    for i in range(num_samples):
        data = {
            "value": np.random.randn(10).astype(np.float32),
            "step": np.array([i], dtype=np.int32),
        }
        
        logger.log("high_freq_data", data)
        
        time.sleep(0.01)  # 100Hz
    
    elapsed = time.time() - start_time
    logger.stop()
    
    stats = logger.get_stats()
    print(f"Logged {stats['total_samples']} samples in {elapsed:.2f} seconds")
    print(f"Average rate: {stats['total_samples']/elapsed:.2f} Hz")
    print("Test 3 completed!\n")


def test_stress():
    """Stress test with many samples."""
    print("=== Test 4: Stress Test ===")
    
    logger = StreamingPickleLogger(
        log_dir="./test_logs_stress",
        batch_size=200,
        flush_interval=10.0,
        max_queue_size=5000
    )
    
    logger.start()
    
    num_samples = 1000
    start_time = time.time()
    
    print(f"Logging {num_samples} samples as fast as possible...")
    
    for i in range(num_samples):
        data = {
            "large_array": np.random.randn(100).astype(np.float32),
            "step": np.array([i], dtype=np.int32),
        }
        
        logger.log("stress_test", data)
        
        if i % 100 == 0:
            stats = logger.get_stats()
            print(f"Progress: {i}/{num_samples}, queue: {stats['queue_size']}")
    
    elapsed = time.time() - start_time
    logger.stop()
    
    stats = logger.get_stats()
    print(f"\nStress test completed!")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Average rate: {stats['total_samples']/elapsed:.2f} samples/sec")
    print(f"Total samples: {stats['total_samples']}")
    print("Test 4 completed!\n")


def main():
    """Run all tests."""
    print("Starting Data Logger Tests\n")
    print("=" * 50)
    
    try:
        # Test 1: Basic logging
        log_dir, session_id = test_basic_logging()
        
        # Test 2: Data loading
        test_data_loading(log_dir, session_id)
        
        # Test 3: High frequency
        test_high_frequency()
        
        # Test 4: Stress test
        test_stress()
        
        print("=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
