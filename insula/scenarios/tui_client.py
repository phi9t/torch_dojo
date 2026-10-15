"""Private local transport for the Emacs scenario worker."""

import json
import os
from pathlib import Path
import socket
import threading


class ScenarioError(RuntimeError):
    pass


class OutcomeUnknown(ScenarioError):
    pass


class ScenarioClient:
    def __init__(self, socket_path: Path, timeout: float = 90):
        self.socket_path = Path(socket_path)
        self.timeout = timeout
        self._lock = threading.Lock()
        self._next_id = 0
        self.last_snapshot = None

    def request(self, command: str, args: dict | None = None) -> dict:
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            payload = json.dumps({'id': request_id, 'command': command,
                                  'args': args or {}}, ensure_ascii=False).encode() + b'\n'
            sent = False
            try:
                with socket.socket(socket.AF_UNIX) as conn:
                    conn.settimeout(self.timeout)
                    directory_fd = os.open(self.socket_path.parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        conn.connect(f'/proc/self/fd/{directory_fd}/{self.socket_path.name}')
                    finally:
                        os.close(directory_fd)
                    sent = True
                    conn.sendall(payload)
                    data = bytearray()
                    while not data.endswith(b'\n'):
                        chunk = conn.recv(65536)
                        if not chunk:
                            raise ConnectionError('worker closed the connection')
                        data.extend(chunk)
                        if len(data) > 2 * 1024 * 1024:
                            raise ValueError('worker response too large')
                response = json.loads(data)
                if response.get('id') != request_id:
                    raise ValueError('worker response ID mismatch')
                if response.get('ok') is not True:
                    raise ScenarioError(response.get('error', 'worker rejected request'))
                snapshot = response['snapshot']
                self.last_snapshot = snapshot
                return snapshot
            except (OSError, ValueError, KeyError) as exc:
                if sent and command in {'set_source', 'next', 'restart', 'quit'}:
                    raise OutcomeUnknown('Worker connection lost; outcome unknown') from exc
                raise ScenarioError(f'Worker unavailable: {type(exc).__name__}') from exc
