import argparse
import socket
import time
import sys
from pathlib import Path

import numpy as np

# Ensure we can import shared_state.py from the same folder when launched as a script
sys.path.append(str(Path(__file__).parent.absolute()))
from shared_state import SharedStateBuffer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Send local robot state over UDP from shared memory"
    )
    parser.add_argument(
        "--shm-name",
        type=str,
        required=True,
        help="Name of the shared memory segment for local state",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        required=True,
        help="Length of the state vector (float32)",
    )
    parser.add_argument(
        "--peer-ip",
        type=str,
        required=True,
        help="Peer robot IP address",
    )
    parser.add_argument(
        "--peer-port",
        type=int,
        required=True,
        help="Peer robot UDP port to send state to",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=0.0,
        help="Optional send rate limit in Hz (0 = send as fast as possible)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    state_dim = args.state_dim
    shm_name = args.shm_name

    buf = SharedStateBuffer.attach(shm_name, state_dim)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    peer_addr = (args.peer_ip, args.peer_port)

    interval = 1.0 / args.rate_hz if args.rate_hz > 0 else 0.0

    print(
        f"[state_sender] Starting. shm_name={shm_name}, state_dim={state_dim}, "
        f"peer={peer_addr}, rate_hz={args.rate_hz or 'unlimited'}"
    )

    try:
        last_send_time = 0.0
        while True:
            now = time.time()
            if interval > 0 and (now - last_send_time) < interval:
                # Simple rate limiting
                time.sleep(max(0.0, interval - (now - last_send_time)))
                now = time.time()

            arr = buf.read()
            if arr is None:
                # No consistent snapshot; wait briefly and retry
                time.sleep(0.001)
                continue

            try:
                sock.sendto(arr.tobytes(), peer_addr)
                last_send_time = now
            except OSError as e:
                # Network errors should not crash the process; just report and retry
                print(f"[state_sender] send error: {e}")
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("[state_sender] Interrupted, exiting.")
    finally:
        buf.close()
        sock.close()


if __name__ == "__main__":
    main()
