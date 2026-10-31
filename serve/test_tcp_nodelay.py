"""Streamed replies set TCP_NODELAY (Project Maya #626): the token frames are small and flushed as they come, and
Nagle + the peer's delayed ACK would hold one ~40 ms.    python -m unittest serve.test_tcp_nodelay"""
import socket
import unittest


class NoDelayStreaming(unittest.TestCase):
    def test_sse_sets_tcp_nodelay(self):
        from serve.server import make_handler
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        client = socket.create_connection(listener.getsockname())
        conn, _ = listener.accept()
        try:
            handler_cls = make_handler(None)
            handler = handler_cls.__new__(handler_cls)
            handler.connection = conn
            for name in ("send_response", "send_header", "end_headers", "_note", "_cors"):
                setattr(handler, name, lambda *a, **k: None)
            self.assertEqual(conn.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY), 0)
            handler._sse()
            self.assertEqual(conn.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY), 1)
        finally:
            conn.close(), client.close(), listener.close()


if __name__ == "__main__":
    unittest.main()
