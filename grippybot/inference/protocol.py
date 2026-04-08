"""
TCP protocol helpers for Pi ↔ Mac inference communication.

Uses length-prefixed JSON messages for headers/actions,
and raw bytes for JPEG image data.
"""

import io
import json
import struct


def send_msg(sock, data):
    """Send a length-prefixed JSON message."""
    raw = json.dumps(data).encode()
    sock.sendall(struct.pack(">I", len(raw)) + raw)


def recv_msg(sock):
    """Receive a length-prefixed JSON message."""
    raw_len = recv_exact(sock, 4)
    if not raw_len:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    raw = recv_exact(sock, msg_len)
    return json.loads(raw.decode())


def recv_exact(sock, n):
    """Receive exactly n bytes."""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def send_frame(sock, jpeg_bytes, state, gripper):
    """Send image + state to Mac. Used by inference server."""
    header = {
        "state": state,
        "gripper": gripper,
        "jpeg_size": len(jpeg_bytes),
    }
    send_msg(sock, header)
    sock.sendall(jpeg_bytes)
