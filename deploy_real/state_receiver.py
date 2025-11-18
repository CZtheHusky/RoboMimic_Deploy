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
        description="Receive peer robot state over UDP into shared memory"
    )
    parser.add_argument(
        "--shm-name",
        type=str,
        required=True,
        help="Name of the shared memory segment for peer state",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        required=True,
        help="Length of the state vector (float32)",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        required=True,
        help="UDP port to listen on for peer state",
    )
    parser.add_argument(
        "--bind-ip",
        type=str,
        default="0.0.0.0",
        help="Local IP to bind the UDP socket to",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=30,
        help="Seconds between latency printouts (0 = disable)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    state_dim = args.state_dim
    shm_name = args.shm_name

    buf = SharedStateBuffer.attach(shm_name, state_dim)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_ip, args.listen_port))
    sock.settimeout(1.0)

    print(
        f"[state_receiver] Listening on {args.bind_ip}:{args.listen_port}, "
        f"shm_name={shm_name}, state_dim={state_dim}"
    )

    last_print_time = time.time()
    sample_count = 0
    last_latency_ms = None

    try:
        while True:
            try:
                data, addr = sock.recvfrom(state_dim * 4 + 64)
            except socket.timeout:
                # Periodic latency print even if no packets
                now = time.time()
                if args.print_interval > 0 and (now - last_print_time) >= args.print_interval:
                    print("[state_receiver] No packets received in the last interval.")
                    last_print_time = now
                    sample_count = 0
                continue

            if len(data) < state_dim * 4:
                # Unexpected packet size; ignore
                continue

            arr = np.frombuffer(data[: state_dim * 4], dtype=np.float32, count=state_dim)
            sample_count += 1

            # Write full state into shared memory (including timestamp)
            try:
                buf.write(arr)
            except ValueError as e:
                print(f"[state_receiver] write error: {e}")

            # Periodic latency reporting
            if args.print_interval > 0:
                now = time.time()
                if (now - last_print_time) >= args.print_interval:
                    hz = sample_count / max(now - last_print_time, 1e-6)
                    print(f"[state_receiver] recv_rate={hz:.1f} Hz")
                    last_print_time = now
                    sample_count = 0

    except KeyboardInterrupt:
        print("[state_receiver] Interrupted, exiting.")
    finally:
        buf.close()
        sock.close()


if __name__ == "__main__":
    main()
