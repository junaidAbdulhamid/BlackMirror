"""Capturing TRIBE's pre-filter segment total from its LogRecord.

TRIBE drops event-free segments and returns only survivors; the total reaches us
only through a log call. We read the structured args, never the rendered string,
and refuse the number whenever it cannot be trusted.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

tribe_backend = pytest.importorskip(
    "blackmirror.inference.tribe_backend", reason="needs the tribe backend module"
)

_capture_segment_counts = tribe_backend._capture_segment_counts

TRIBE_MESSAGE = "Predicted %d / %d segments (%.1f%% kept)"


def _emit(*args: object) -> None:
    logging.getLogger("tribev2.demo_utils").info(TRIBE_MESSAGE, *args)


def test_counts_are_read_from_the_log_record() -> None:
    """The real observed case: 3 of 100 segments kept."""
    with _capture_segment_counts() as counts:
        _emit(3, 100, 3.0)
    assert counts.kept == 3
    assert counts.total == 100


def test_unrelated_log_lines_are_ignored() -> None:
    with _capture_segment_counts() as counts:
        logging.getLogger("tribev2.demo_utils").info("Loading model from %s", "/tmp/x")
    assert counts.total is None


def _handler(counts: object) -> object:
    """A handler bound to the current thread, as the capture creates it."""
    return tribe_backend._SegmentCountHandler(counts, threading.get_ident())


def _feed(handler: logging.Handler, message: str, args: tuple[object, ...]) -> None:
    """Push a record straight at our handler.

    Deliberately bypasses `Logger.info`: these cases are malformed on purpose,
    and routing them through the logging machinery would make *other* handlers
    (pytest's capture, the console) try to render them and raise. The unit under
    test is the argument parsing, not Python's formatter.
    """
    handler.emit(
        logging.LogRecord(
            name="tribev2.demo_utils",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=message,
            args=args,
            exc_info=None,
        )
    )


@pytest.mark.parametrize(
    "args",
    [
        (None, None, None),      # upstream stopped passing integers
        ("3", "100", 3.0),       # passed as strings
        (5,),                    # fewer args than expected
        (),                      # no args
    ],
)
def test_malformed_args_are_ignored(args: tuple[object, ...]) -> None:
    """A changed upstream signature must yield None, never a wrong number."""
    counts = tribe_backend._SegmentCounts()
    _feed(_handler(counts), TRIBE_MESSAGE, args)
    assert counts.total is None


def test_numpy_integers_are_accepted() -> None:
    """Regression: upstream passes `keep.sum()`, a np.int64, not a Python int.

    `isinstance(np.int64(3), int)` is False, so a strict int check silently
    discarded every real tally while the log line still printed "3 / 100".
    """
    import numpy as np

    counts = tribe_backend._SegmentCounts()
    _feed(
        _handler(counts),
        TRIBE_MESSAGE,
        (np.int64(3), 100, 3.0),
    )
    assert counts.kept == 3
    assert counts.total == 100
    assert isinstance(counts.kept, int)


def test_booleans_are_not_counts() -> None:
    counts = tribe_backend._SegmentCounts()
    _feed(_handler(counts), TRIBE_MESSAGE, (True, True, 1.0))
    assert counts.total is None


def test_impossible_tallies_are_rejected() -> None:
    """kept > total cannot be right, so it is discarded."""
    counts = tribe_backend._SegmentCounts()
    _feed(_handler(counts), TRIBE_MESSAGE, (100, 3, 3.0))
    assert counts.total is None


def test_a_different_upstream_message_is_ignored() -> None:
    counts = tribe_backend._SegmentCounts()
    _feed(
        _handler(counts),
        "Finished %d of %d batches",
        (3, 100),
    )
    assert counts.total is None


def test_capture_works_when_logging_is_quiet() -> None:
    """A user running at WARNING must not silently lose the tally."""
    tribe_logger = logging.getLogger("tribev2.demo_utils")
    tribe_logger.setLevel(logging.WARNING)
    try:
        with _capture_segment_counts() as counts:
            _emit(3, 100, 3.0)
        assert counts.total == 100
        # The caller's level is restored.
        assert tribe_logger.level == logging.WARNING
    finally:
        tribe_logger.setLevel(logging.NOTSET)


def test_handler_is_removed_afterwards() -> None:
    """No handler may leak onto TRIBE's logger between runs."""
    tribe_logger = logging.getLogger("tribev2.demo_utils")
    before = list(tribe_logger.handlers)
    with _capture_segment_counts():
        assert len(tribe_logger.handlers) == len(before) + 1
    assert tribe_logger.handlers == before


def test_handler_is_removed_even_on_error() -> None:
    tribe_logger = logging.getLogger("tribev2.demo_utils")
    before = list(tribe_logger.handlers)
    with pytest.raises(RuntimeError), _capture_segment_counts():
        raise RuntimeError("inference blew up")
    assert tribe_logger.handlers == before


def test_only_the_last_tally_is_kept() -> None:
    with _capture_segment_counts() as counts:
        _emit(1, 10, 10.0)
        _emit(3, 100, 3.0)
    assert (counts.kept, counts.total) == (3, 100)


def test_concurrent_captures_do_not_cross_talk() -> None:
    """Two predicts in one process must not steal each other's tallies.

    `tribev2.demo_utils` has a single module-level logger, so without thread
    affinity both handlers would see both records and one run would be
    attributed the other's segment counts — a wrong number, silently.
    """
    barrier = threading.Barrier(2)

    def run(kept: int, total: int) -> tuple[int | None, int | None]:
        with _capture_segment_counts() as counts:
            barrier.wait(timeout=5)  # ensure both captures are open at once
            logging.getLogger("tribev2.demo_utils").info(TRIBE_MESSAGE, kept, total, 0.0)
            barrier.wait(timeout=5)  # hold both open until both have logged
        return counts.kept, counts.total

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run, 3, 100)
        second = pool.submit(run, 7, 200)
        assert first.result(timeout=10) == (3, 100)
        assert second.result(timeout=10) == (7, 200)


def test_nested_captures_restore_the_logger_once() -> None:
    """Reference counting: the inner exit must not restore while the outer runs."""
    tribe_logger = logging.getLogger("tribev2.demo_utils")
    tribe_logger.setLevel(logging.WARNING)
    try:
        with _capture_segment_counts() as outer:
            with _capture_segment_counts() as inner:
                _emit(3, 100, 3.0)
                assert inner.total == 100
            # The inner exit must NOT have restored WARNING yet.
            _emit(4, 100, 4.0)
            assert outer.total == 100
        assert tribe_logger.level == logging.WARNING
    finally:
        tribe_logger.setLevel(logging.NOTSET)
