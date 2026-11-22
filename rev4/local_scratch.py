"""Node-local temporary images with bounded retries for transient cleanup errors."""
from contextlib import contextmanager
import errno
import os
from pathlib import Path
import tempfile
import time


@contextmanager
def local_directory(prefix):
    # Shared datastore scratch can race with network-filesystem unlink handling.
    # Do not change generated pixels, filenames, or the frozen detector pipeline.
    root = Path(os.environ.get('SLURM_TMPDIR') or ('/tmp' if os.name == 'posix' else tempfile.gettempdir()))
    root.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix=prefix, dir=root)
    try:
        yield Path(temporary.name)
    finally:
        for attempt in range(4):
            try:
                temporary.cleanup()
                break
            except OSError as exc:
                if exc.errno not in (errno.ENOTEMPTY, errno.EBUSY) or attempt == 3:
                    raise
                time.sleep(.05 * 2**attempt)
