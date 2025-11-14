"""
Efficient data logging system for recording policy observations and actions.
Uses a background thread with queue-based buffering for minimal performance impact.
"""

import threading
import queue
import json
import pickle
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import gzip


class DataLogger:
    """
    Asynchronous data logger that records data to files in a background thread.
    
    Features:
    - Non-blocking: main thread only enqueues data
    - Batch writing: accumulates data before writing to disk
    - Multiple formats: supports JSON, JSONL, pickle, and compressed formats
    - Automatic file management: creates files based on data keys
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        format: str = "pickle",  # "jsonl", "pickle", "pickle_gz"
        batch_size: int = 100,  # Write every N samples
        flush_interval: float = 5.0,  # Flush every N seconds
        max_queue_size: int = 10000,
        compress: bool = False,
        file_prefix: str = "",
    ):
        """
        Initialize the data logger.
        
        self.file_prefix = file_prefix
        Args:
            log_dir: Directory to store log files
            format: File format ("jsonl", "pickle", "pickle_gz")
            batch_size: Number of samples to accumulate before writing
            flush_interval: Time interval (seconds) to force flush
            max_queue_size: Maximum queue size before blocking
            compress: Whether to compress pickle files (if format is "pickle")
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.format = format
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.compress = compress
        
        # Queue for passing data from main thread to logger thread
        self.data_queue = queue.Queue(maxsize=max_queue_size)
        
        # Internal buffers for each data category
        self.buffers: Dict[str, list] = {}
        self.file_handles: Dict[str, Any] = {}
        
        # Control flags
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Statistics
        self.total_samples = 0
        self.last_flush_time = time.time()
        
        # Timestamp for unique session ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def start(self):
        """Start the logging thread."""
        if self.running:
            print("DataLogger is already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._logging_loop, daemon=True)
        self.thread.start()
        print(f"DataLogger started. Logging to: {self.log_dir}")
        
    def stop(self, timeout: float = 10.0):
        """
        Stop the logging thread and flush all remaining data.
        
        Args:
            timeout: Maximum time to wait for thread to finish
        """
        if not self.running:
            return
        
        print("Stopping DataLogger...")
        self.running = False
        
        # Wait for thread to finish
        if self.thread:
            self.thread.join(timeout=timeout)
        
        # Flush any remaining data
        self._flush_all_buffers()
        
        # Close all file handles
        self._close_all_files()
        
        print(f"DataLogger stopped. Total samples logged: {self.total_samples}")
        
    def stop_async(self, timeout: float = 10.0, on_complete: Optional[Callable[[Dict[str, Any]], None]] = None):
        """Stop the logger without blocking the caller.
        Spawns a helper thread that calls stop() and then invokes on_complete(stats)
        if provided. Returns immediately.
        """
        if not self.running:
            return None
        
        def _do_stop():
            try:
                self.stop(timeout=timeout)
            finally:
                if on_complete is not None:
                    try:
                        on_complete(self.get_stats())
                    except Exception:
                        pass
        t = threading.Thread(target=_do_stop, daemon=True)
        t.start()
        return t


    def log(self, data_name: str, data: Dict[str, np.ndarray]):
        """
        Log data asynchronously.
        
        Args:
            data_name: Name/category of the data (e.g., "observations", "actions")
            data: Dictionary mapping field names to numpy arrays
        """
        if not self.running:
            print("Warning: DataLogger is not running. Call start() first.")
            return
        
        try:
            # Convert numpy arrays to lists for serialization
            serializable_data = {}
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    serializable_data[key] = value.tolist()
                else:
                    serializable_data[key] = value
            
            # Add timestamp
            serializable_data["_timestamp"] = time.time()
            
            # Put in queue (non-blocking with timeout)
            self.data_queue.put((data_name, serializable_data), timeout=0.01)
            
        except queue.Full:
            print(f"Warning: DataLogger queue is full, dropping sample for {data_name}")
        except Exception as e:
            print(f"Error logging data: {e}")
    
    def _logging_loop(self):
        """Main loop running in the background thread."""
        while self.running or not self.data_queue.empty():
            try:
                # Get data from queue with timeout
                try:
                    data_name, data = self.data_queue.get(timeout=0.1)
                except queue.Empty:
                    # Check if we should flush based on time
                    if time.time() - self.last_flush_time > self.flush_interval:
                        self._flush_all_buffers()
                        self.last_flush_time = time.time()
                    continue
                
                # Initialize buffer if needed
                if data_name not in self.buffers:
                    self.buffers[data_name] = []
                
                # Add to buffer
                self.buffers[data_name].append(data)
                self.total_samples += 1
                
                # Check if we should flush this buffer
                if len(self.buffers[data_name]) >= self.batch_size:
                    self._flush_buffer(data_name)
                
                # Check if we should flush all buffers based on time
                if time.time() - self.last_flush_time > self.flush_interval:
                    self._flush_all_buffers()
                    self.last_flush_time = time.time()
                    
            except Exception as e:
                print(f"Error in logging loop: {e}")
                import traceback
                traceback.print_exc()
    
    def _get_file_path(self, data_name: str) -> Path:
        """Get the file path for a given data name."""
        if self.format == "jsonl":
            ext = ".jsonl"
        elif self.format == "pickle":
            ext = ".pkl.gz" if self.compress else ".pkl"
        elif self.format == "pickle_gz":
            ext = ".pkl.gz"
        else:
            ext = ".dat"
        
        return self.log_dir / f"{data_name}_{self.session_id}{ext}"
    
    def _flush_buffer(self, data_name: str):
        """Flush a specific buffer to disk."""
        if data_name not in self.buffers or len(self.buffers[data_name]) == 0:
            return
        
        file_path = self._get_file_path(data_name)
        
        try:
            if self.format == "jsonl":
                # Append to JSONL file
                with open(file_path, 'a') as f:
                    for sample in self.buffers[data_name]:
                        json.dump(sample, f)
                        f.write('\n')
            
            elif self.format in ["pickle", "pickle_gz"]:
                # For pickle, we need to load existing data, append, and save
                # This is less efficient but maintains compatibility
                existing_data = []
                if file_path.exists():
                    try:
                        if self.format == "pickle_gz" or self.compress:
                            with gzip.open(file_path, 'rb') as f:
                                existing_data = pickle.load(f)
                        else:
                            with open(file_path, 'rb') as f:
                                existing_data = pickle.load(f)
                    except:
                        existing_data = []
                
                # Append new data
                existing_data.extend(self.buffers[data_name])
                
                # Save all data
                if self.format == "pickle_gz" or self.compress:
                    with gzip.open(file_path, 'wb') as f:
                        pickle.dump(existing_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open(file_path, 'wb') as f:
                        pickle.dump(existing_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Clear buffer
            self.buffers[data_name] = []
            
        except Exception as e:
            print(f"Error flushing buffer for {data_name}: {e}")
    
    def _flush_all_buffers(self):
        """Flush all buffers to disk."""
        for data_name in list(self.buffers.keys()):
            self._flush_buffer(data_name)
    
    def _close_all_files(self):
        """Close all open file handles."""
        for fh in self.file_handles.values():
            try:
                fh.close()
            except:
                pass
        self.file_handles = {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            "total_samples": self.total_samples,
            "queue_size": self.data_queue.qsize(),
            "buffer_sizes": {name: len(buf) for name, buf in self.buffers.items()},
            "session_id": self.session_id,
            "log_dir": str(self.log_dir),
        }


class StreamingPickleLogger:
    """
    Alternative implementation that streams pickle data more efficiently.
    Each file is a sequence of pickled objects, avoiding the need to load all data.
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        batch_size: int = 100,
        flush_interval: float = 5.0,
        max_queue_size: int = 10000,
        file_prefix: str = "",
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.file_prefix = file_prefix
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        self.data_queue = queue.Queue(maxsize=max_queue_size)
        self.buffers: Dict[str, list] = {}
        self.file_handles: Dict[str, Any] = {}
        
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        self.total_samples = 0
        self.last_flush_time = time.time()
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def start(self):
        """Start the logging thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._logging_loop, daemon=True)
        self.thread.start()
        print(f"StreamingPickleLogger started. Logging to: {self.log_dir}")
        
    def stop(self, timeout: float = 10.0):
        """Stop the logging thread and flush all remaining data."""
        if not self.running:
            return
        
        print("Stopping StreamingPickleLogger...")
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=timeout)
        
        self._flush_all_buffers()
        self._close_all_files()
        
        print(f"StreamingPickleLogger stopped. Total samples: {self.total_samples}")
        

    def stop_async(self, timeout: float = 10.0, on_complete: Optional[Callable[[Dict[str, Any]], None]] = None):
        """Stop the logger without blocking the caller.
        Spawns a helper thread that calls stop() and then invokes on_complete(stats)
        if provided. Returns immediately.
        """
        if not self.running:
            return None
        
        def _do_stop():
            try:
                self.stop(timeout=timeout)
            finally:
                if on_complete is not None:
                    try:
                        on_complete(self.get_stats())
                    except Exception:
                        pass
        t = threading.Thread(target=_do_stop, daemon=True)
        t.start()
        return t

    def log(self, data_name: str, data: Dict[str, np.ndarray]):
        """Log data asynchronously."""
        if not self.running:
            return
        
        try:
            # Add timestamp
            data_with_ts = dict(data)
            data_with_ts["_timestamp"] = time.time()
            
            self.data_queue.put((data_name, data_with_ts), timeout=0.01)
            
        except queue.Full:
            print(f"Warning: Queue full, dropping sample for {data_name}")
        except Exception as e:
            print(f"Error logging data: {e}")
    
    def _logging_loop(self):
        """Main loop running in the background thread."""
        while self.running or not self.data_queue.empty():
            try:
                try:
                    data_name, data = self.data_queue.get(timeout=0.1)
                except queue.Empty:
                    if time.time() - self.last_flush_time > self.flush_interval:
                        self._flush_all_buffers()
                        self.last_flush_time = time.time()
                    continue
                
                if data_name not in self.buffers:
                    self.buffers[data_name] = []
                
                self.buffers[data_name].append(data)
                self.total_samples += 1
                
                if len(self.buffers[data_name]) >= self.batch_size:
                    self._flush_buffer(data_name)
                
                if time.time() - self.last_flush_time > self.flush_interval:
                    self._flush_all_buffers()
                    self.last_flush_time = time.time()
                    
            except Exception as e:
                print(f"Error in logging loop: {e}")
    
    def _get_file_handle(self, data_name: str):
        """Get or create file handle for a data name."""
        if data_name not in self.file_handles:
            prefix = f"{self.file_prefix}_" if self.file_prefix else ""
            file_path = self.log_dir / f"{prefix}{data_name}_{self.session_id}.pkl"
            self.file_handles[data_name] = open(file_path, 'ab')  # Append binary mode
        return self.file_handles[data_name]
    
    def _flush_buffer(self, data_name: str):
        """Flush a specific buffer to disk using streaming pickle."""
        if data_name not in self.buffers or len(self.buffers[data_name]) == 0:
            return
        
        try:
            fh = self._get_file_handle(data_name)
            
            # Write each sample as a separate pickle object
            for sample in self.buffers[data_name]:
                pickle.dump(sample, fh, protocol=pickle.HIGHEST_PROTOCOL)
            
            fh.flush()  # Ensure data is written to disk
            
            self.buffers[data_name] = []
            
        except Exception as e:
            print(f"Error flushing buffer for {data_name}: {e}")
    
    def _flush_all_buffers(self):
        """Flush all buffers to disk."""
        for data_name in list(self.buffers.keys()):
            self._flush_buffer(data_name)
    
    def _close_all_files(self):
        """Close all open file handles."""
        for fh in self.file_handles.values():
            try:
                fh.close()
            except:
                pass
        self.file_handles = {}
    
    def get_stats(self):
        """Get logging statistics."""
        return {
            "total_samples": self.total_samples,
            "queue_size": self.data_queue.qsize(),
            "buffer_sizes": {name: len(buf) for name, buf in self.buffers.items()},
            "session_id": self.session_id,
            "log_dir": str(self.log_dir),
        }


def load_streaming_pickle(file_path: str):
    """
    Load data from a streaming pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        List of all samples in the file
    """
    data = []
    with open(file_path, 'rb') as f:
        while True:
            try:
                sample = pickle.load(f)
                data.append(sample)
            except EOFError:
                break
    return data


# Example usage
if __name__ == "__main__":
    # Test the logger
    logger = StreamingPickleLogger(log_dir="./test_logs", batch_size=10)
    logger.start()
    
    # Simulate logging data
    for i in range(100):
        obs_data = {
            "joint_pos": np.random.randn(29),
            "joint_vel": np.random.randn(29),
            "imu": np.random.randn(4),
        }
        action_data = {
            "target_pos": np.random.randn(29),
        }
        
        logger.log("observations", obs_data)
        logger.log("actions", action_data)
        
        time.sleep(0.01)
    
    logger.stop()
    print("Test complete")
