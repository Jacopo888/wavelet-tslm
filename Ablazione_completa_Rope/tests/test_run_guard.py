from __future__ import annotations

import pytest

from wavelet_rope.errors import ActiveRunError
from wavelet_rope.run_guard import RunLease


def test_run_lease_blocks_second_worker_and_can_be_reacquired(tmp_path) -> None:
    first = RunLease(tmp_path, "a" * 64, job_id="1.server")
    second = RunLease(tmp_path, "a" * 64, job_id="2.server")
    owner = first.acquire()
    assert owner["job_id"] == "1.server"
    with pytest.raises(ActiveRunError, match="active lease"):
        second.acquire()
    first.release()
    second.acquire()
    second.release()

