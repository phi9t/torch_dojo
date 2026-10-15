"""Read bounded, redacted process output for the scenario TUI."""

from pathlib import Path
import re


_QUERY_TOKEN = re.compile(r'(?i)([?&]token=)[^\s&]+')
_AUTH_TOKEN = re.compile(r'(?i)(authorization:\s*token\s+)\S+')
_ESCAPE = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')


def redact_text(value: str, token_path: Path | None = None) -> str:
    """Remove token spellings from any text shown in the TUI."""
    value = _ESCAPE.sub('', str(value))
    value = _QUERY_TOKEN.sub(r'\1[redacted]', value)
    value = _AUTH_TOKEN.sub(r'\1[redacted]', value)
    if token_path:
        try:
            token = Path(token_path).read_text().strip()
            if token:
                value = value.replace(token, '[redacted]')
        except OSError:
            pass
    return value


class LogTail:
    def __init__(self, path: Path, *, token_path: Path | None = None,
                 start_offset: int = 0):
        self.path = Path(path)
        self.token_path = Path(token_path) if token_path else None
        self._position = start_offset
        self._fragment = ''
        self._initial = True

    def poll(self) -> list[str]:
        try:
            size = self.path.stat().st_size
            if size < self._position:
                self._position = 0
                self._fragment = ''
            with self.path.open('rb') as stream:
                if size - self._position > 65536:
                    stream.seek(size - 65536)
                    stream.readline()
                    self._fragment = ''
                else:
                    stream.seek(self._position)
                data = stream.read(65536)
                self._position = stream.tell()
        except OSError:
            return []
        text = self._fragment + data.decode('utf-8', errors='replace')
        parts = text.split('\n')
        self._fragment = parts.pop()
        if self._initial:
            self._initial = False
            parts = parts[-12:]
        return [self._redact(line.rstrip('\r')) for line in parts if line]

    def _redact(self, line: str) -> str:
        return redact_text(line, self.token_path)
