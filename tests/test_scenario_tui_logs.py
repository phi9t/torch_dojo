import tempfile
import unittest
from pathlib import Path

from insula.scenarios.tui_logs import LogTail


class LogTailTest(unittest.TestCase):
    def test_incremental_output_and_token_redaction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / 'server.log'
            token = root / 'token'
            token.write_text('private-token')
            log.write_text('Kernel started: abc\nOpen /?token=old-token\n')
            tail = LogTail(log, token_path=token)
            first = tail.poll()
            self.assertIn('Kernel started: abc', first)
            self.assertNotIn('old-token', '\n'.join(first))
            self.assertIn('[redacted]', '\n'.join(first))
            with log.open('a') as stream:
                stream.write('Authorization: token private-token\nKernel connected\n')
            second = tail.poll()
            self.assertEqual(len(second), 2)
            self.assertIn('Kernel connected', second)
            self.assertNotIn('private-token', '\n'.join(second))
            self.assertEqual(tail.poll(), [])

    def test_start_offset_shows_only_the_current_server_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / 'server.log'
            log.write_text('Old kernel started\n')
            offset = log.stat().st_size
            with log.open('a') as stream:
                stream.write('Current kernel started\n')
            self.assertEqual(LogTail(log, start_offset=offset).poll(),
                             ['Current kernel started'])


if __name__ == '__main__':
    unittest.main()
