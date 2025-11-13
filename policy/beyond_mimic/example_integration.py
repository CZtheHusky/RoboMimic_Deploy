"""
Example of how to integrate data logging into the deployment script.

This shows how to modify deploy_real_cpp.py or similar deployment scripts
to enable data logging for BeyondMimic policy.
"""

# Example 1: Enable logging when creating the FSM controller

from FSM.FSM import FSM
from common.ctrlcomp import StateAndCmd, PolicyOutput

# Create state and command objects
state_cmd = StateAndCmd(num_joints=29)
policy_output = PolicyOutput(num_joints=29)

# Create FSM with logging enabled
fsm_controller = FSM(
    state_cmd=state_cmd, 
    policy_output=policy_output,
    enable_logging=True,  # Enable logging
    log_dir="./logs/experiment_001"  # Custom log directory
)


# Example 2: Enable logging via environment variable

import os

# Set environment variable before running
enable_logging = os.getenv("ENABLE_LOGGING", "false").lower() == "true"
log_dir = os.getenv("LOG_DIR", "./logs/default")

fsm_controller = FSM(
    state_cmd=state_cmd,
    policy_output=policy_output,
    enable_logging=enable_logging,
    log_dir=log_dir
)

# Then run with:
# ENABLE_LOGGING=true LOG_DIR=./logs/test_run python deploy_real_cpp.py


# Example 3: Add logging control to existing deployment script

class Controller:
    def __init__(self, config, enable_logging=False, log_dir="./logs"):
        self.config = config
        self.num_joints = config.num_joints
        
        # ... existing initialization ...
        
        # Create FSM controller with optional logging
        self.state_cmd = StateAndCmd(self.num_joints)
        self.policy_output = PolicyOutput(self.num_joints)
        self.FSM_controller = FSM(
            self.state_cmd, 
            self.policy_output,
            enable_logging=enable_logging,
            log_dir=log_dir
        )
    
    def shutdown(self):
        """Shutdown the controller gracefully"""
        print("Shutting down controller...")
        
        # FSM will handle logging shutdown internally
        # Just make sure to exit cleanly
        
        # ... rest of shutdown code ...


# Example 4: Command-line argument for logging

import argparse

def main():
    parser = argparse.ArgumentParser(description='Deploy robot controller')
    parser.add_argument('--enable-logging', action='store_true',
                       help='Enable data logging')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Directory to save logs')
    args = parser.parse_args()
    
    # Create controller with logging options
    controller = Controller(
        config=config,
        enable_logging=args.enable_logging,
        log_dir=args.log_dir
    )
    
    try:
        while True:
            controller.run()
            if controller.should_exit():
                break
    except KeyboardInterrupt:
        pass
    finally:
        controller.shutdown()

# Then run with:
# python deploy_real_cpp.py --enable-logging --log-dir ./logs/experiment_001


# Example 5: Automatic logging with timestamp

from datetime import datetime

def create_timestamped_log_dir(base_dir="./logs"):
    """Create a unique log directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{base_dir}/run_{timestamp}"
    return log_dir

# Use in controller
controller = Controller(
    config=config,
    enable_logging=True,
    log_dir=create_timestamped_log_dir("./logs/experiments")
)
