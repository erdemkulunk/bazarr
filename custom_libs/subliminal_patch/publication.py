"""Publish complete subtitle bytes without replacing an independently created destination."""

import os
import secrets


def create_file(path, content, chmod=None):
    """Atomically create the final name only after all content is written on the same filesystem."""
    temporary = os.path.join(os.path.dirname(path), '.bazarr-' + secrets.token_hex(16) + '.tmp')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(content)
            if chmod is not None:
                os.fchmod(stream.fileno(), chmod)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
