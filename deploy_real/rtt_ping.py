import argparse
import socket
import struct
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Application-layer RTT ping/pong between two robots"
    )
    parser.add_argument(
        "--peer-ip",
        type=str,
        required=True,
        help="Peer robot IP address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50060,
        help="UDP port used for ping/pong on both robots",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=1.0,
        help="Ping send rate in Hz (default: 1 Hz)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", args.port))
    sock.setblocking(False)

    peer_addr = (args.peer_ip, args.port)
    seq = 0
    interval = 1.0 / args.rate_hz if args.rate_hz > 0 else 1.0
    last_ping_time = 0.0

    print(f"[rtt_ping] Listening on 0.0.0.0:{args.port}, peer={peer_addr}, rate={args.rate_hz} Hz")

    try:
        while True:
            now = time.monotonic()

            # Receive any pending messages
            while True:
                try:
                    data, addr = sock.recvfrom(64)
                except BlockingIOError:
                    break
                except OSError:
                    break
                if not data:
                    continue
                msg_type = data[0:1]
                if msg_type == b"P":
                    # Incoming ping: reply with pong
                    sock.sendto(b"O" + data[1:], addr)
                elif msg_type == b"O":
                    # Pong: compute RTT
                    if len(data) >= 1 + 4 + 8:
                        try:
                            send_time = struct.unpack("!d", data[5:13])[0]
                            rtt_ms = (time.monotonic() - send_time) * 1000.0
                            print(f"[rtt_ping] RTT: {rtt_ms:.2f} ms")
                        except Exception as e:
                            print(f"[rtt_ping] failed to parse pong: {e}")

            # Periodically send ping
            if (now - last_ping_time) >= interval:
                seq += 1
                send_time = time.monotonic()
                payload = b"P" + seq.to_bytes(4, "big") + struct.pack("!d", send_time)
                try:
                    sock.sendto(payload, peer_addr)
                    last_ping_time = send_time
                except OSError as e:
                    print(f"[rtt_ping] send error: {e}")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("[rtt_ping] Interrupted, exiting.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()

