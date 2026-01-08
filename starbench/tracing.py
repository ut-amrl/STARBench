# evaluation/tracing.py
import time
import threading
from typing import Any, Callable, Dict, Optional, Set
from functools import wraps

_trace_lock = threading.Lock()
_current = threading.local()
_trace: Dict[str, list] = {}  # task_id -> list[dict]


class TerminateEpisode(RuntimeError):
    """Raised when an action is attempted/completed that should terminate the episode."""
    pass


def set_current_task(
    task_id: str,
    *,
    sink: Optional[Callable[[dict], None]] = None,
    stop_on: Optional[Set[str]] = None,
    stop_after: Optional[Set[str]] = None,
):
    """
    Set task-local tracing settings for the current thread.

    Args:
      task_id: group all trace records under this id.
      sink: optional callback(rec_dict) called immediately for every record.
      stop_on: set of action names that should terminate immediately when attempted (BEFORE call).
      stop_after: set of action names that should terminate immediately after completion (AFTER call).
    """
    _current.task_id = task_id
    _current.sink = sink
    _current.stop_on = set(stop_on) if stop_on else set()
    _current.stop_after = set(stop_after) if stop_after else set()
    _current.seq = 0  # step counter for this task/thread


def clear_current_task():
    for k in ("task_id", "sink", "stop_on", "stop_after", "seq"):
        if hasattr(_current, k):
            delattr(_current, k)


def get_trace() -> Dict[str, list]:
    return _trace


def get_task_trace(task_id: str) -> list:
    return _trace.get(task_id, [])


def clear_task_trace(task_id: str):
    with _trace_lock:
        _trace.pop(task_id, None)


def trace_call(action_name: str):
    """
    Decorator for action_utils functions.
    Stores RAW args/kwargs/result/exception (no summarization).
    """
    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            task_id = getattr(_current, "task_id", "NO_TASK")
            sink = getattr(_current, "sink", None)
            stop_on = getattr(_current, "stop_on", set())
            stop_after = getattr(_current, "stop_after", set())
            seq = getattr(_current, "seq", 0)

            # Pre-log + terminate *before* calling the underlying function
            if action_name in stop_on:
                rec = {
                    "task_id": task_id,
                    "seq": seq,
                    "action": action_name,
                    "fn": fn.__name__,
                    "ok": False,
                    "terminated": True,
                    "terminate_phase": "pre",
                    "err": TerminateEpisode(f"action '{action_name}' attempted (pre)"),
                    "dt_sec": 0.0,
                    "args": args,        # RAW
                    "kwargs": kwargs,    # RAW
                    "result": None,
                }
                _current.seq = seq + 1

                with _trace_lock:
                    _trace.setdefault(task_id, []).append(rec)
                if sink:
                    sink(rec)

                raise rec["err"]

            t0 = time.time()
            ok = False
            err: Optional[BaseException] = None
            result: Any = None

            try:
                result = fn(*args, **kwargs)
                ok = True
                return result
            except BaseException as e:
                err = e
                raise
            finally:
                t1 = time.time()
                rec = {
                    "task_id": task_id,
                    "seq": seq,
                    "action": action_name,
                    "fn": fn.__name__,
                    "ok": ok,
                    "terminated": False,
                    "terminate_phase": None,
                    "err": err,          # RAW exception object (or None)
                    "dt_sec": t1 - t0,
                    "args": args,        # RAW
                    "kwargs": kwargs,    # RAW
                    "result": result,    # RAW (can be huge / ROS msgs / images)
                }
                _current.seq = seq + 1

                with _trace_lock:
                    _trace.setdefault(task_id, []).append(rec)
                if sink:
                    sink(rec)

                # Terminate AFTER call (even if it failed, we still terminate if requested)
                if action_name in stop_after:
                    raise TerminateEpisode(f"action '{action_name}' completed (post)")
        return wrapped
    return deco


class task_context:
    """
    Context manager for per-task tracing.
    """
    def __init__(self, task_id: str, *, sink=None, stop_on=None, stop_after=None):
        self.task_id = task_id
        self.sink = sink
        self.stop_on = stop_on
        self.stop_after = stop_after

    def __enter__(self):
        set_current_task(
            self.task_id,
            sink=self.sink,
            stop_on=self.stop_on,
            stop_after=self.stop_after,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        clear_current_task()
        return False  # don't suppress exceptions
