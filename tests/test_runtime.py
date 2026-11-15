import multiprocessing as mp
import time

import pytest

from push_lab.runtime import Deadline, RunStopped


def test_a_deadline_stops_while_a_worker_is_busy() -> None:
    """A check made only when a result arrives would wait for the whole minute."""
    with mp.get_context("spawn").Pool(1) as pool:
        assert list(Deadline(60).results(pool.imap(abs, [-1, -2]))) == [1, 2]
        start = time.perf_counter()
        with pytest.raises(RunStopped, match="at the stop line of 2 s"):
            list(Deadline(2).results(pool.imap(time.sleep, [60])))
        assert time.perf_counter() - start < 30


def test_a_deadline_counts_the_seconds_spent_before_it() -> None:
    Deadline(10, spent=9).check()
    with pytest.raises(RunStopped):
        Deadline(10, spent=10).check()
