"""Tests for inference protocol — verifies message encoding/decoding."""

import json
import socket
import struct
import threading
import pytest
from grippybot.inference.protocol import send_msg, recv_msg, recv_exact


class TestProtocol:
    def _create_socket_pair(self):
        """Create a connected pair of sockets for testing."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", port))
        conn, _ = server.accept()
        server.close()
        return client, conn

    def test_send_recv_msg(self):
        """Message round-trips correctly."""
        client, conn = self._create_socket_pair()
        try:
            data = {"base": 45.0, "gripper": 1.0, "step": 42}
            send_msg(client, data)
            received = recv_msg(conn)
            assert received == data
        finally:
            client.close()
            conn.close()

    def test_recv_exact(self):
        """recv_exact gets exactly n bytes."""
        client, conn = self._create_socket_pair()
        try:
            client.sendall(b"hello world")
            result = recv_exact(conn, 5)
            assert result == b"hello"
            result2 = recv_exact(conn, 6)
            assert result2 == b" world"
        finally:
            client.close()
            conn.close()

    def test_recv_msg_on_closed_connection(self):
        """recv_msg returns None when connection is closed."""
        client, conn = self._create_socket_pair()
        client.close()
        result = recv_msg(conn)
        assert result is None
        conn.close()

    def test_multiple_messages(self):
        """Multiple messages in sequence work correctly."""
        client, conn = self._create_socket_pair()
        try:
            for i in range(5):
                send_msg(client, {"step": i})
            for i in range(5):
                msg = recv_msg(conn)
                assert msg == {"step": i}
        finally:
            client.close()
            conn.close()
