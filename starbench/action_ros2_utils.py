from __future__ import annotations

import threading
import time
from functools import wraps
from typing import List

import rclpy
from rclpy.node import Node

from amrl_msgs.srv import (
    GetImageSrv,
    GetImageAtPoseSrv,
    PickObjectSrv,
    DetectVirtualHomeObjectSrv,
    OpenVirtualHomeObjectSrv,
)

from starbench.tracing import trace_call

_ros2_ctx = {"inited": False, "node": None}
_skill_timing_lock = threading.Lock()
_skill_timing_print_every = 10
_skill_timing_stats = {
    "per_fn": {},
    "total_calls": 0,
    "total_time": 0.0,
}


def _print_skill_timing_summary_locked() -> None:
    total_calls = _skill_timing_stats["total_calls"]
    total_time = _skill_timing_stats["total_time"]
    per_fn = _skill_timing_stats["per_fn"]

    overall_avg = (total_time / total_calls) if total_calls else 0.0
    parts = [f"overall={overall_avg:.3f}s ({total_calls})"]

    for fn_name in sorted(per_fn):
        fn_calls = per_fn[fn_name]["count"]
        fn_total = per_fn[fn_name]["total_time"]
        fn_avg = (fn_total / fn_calls) if fn_calls else 0.0
        parts.append(f"{fn_name}={fn_avg:.3f}s ({fn_calls})")

    print("[skill_avg] " + " | ".join(parts))


def _track_skill_timing(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            with _skill_timing_lock:
                per_fn = _skill_timing_stats["per_fn"]
                if fn.__name__ not in per_fn:
                    per_fn[fn.__name__] = {"count": 0, "total_time": 0.0}

                per_fn[fn.__name__]["count"] += 1
                per_fn[fn.__name__]["total_time"] += dt
                _skill_timing_stats["total_calls"] += 1
                _skill_timing_stats["total_time"] += dt

                if _skill_timing_stats["total_calls"] % _skill_timing_print_every == 0:
                    _print_skill_timing_summary_locked()

    return wrapped

def _ensure_ros2() -> Node:
    if not _ros2_ctx["inited"]:
        if not rclpy.ok():
            rclpy.init(args=None)
        _ros2_ctx["inited"] = True
    if _ros2_ctx["node"] is None:
        _ros2_ctx["node"] = rclpy.create_node("action_ros2_utils")
    return _ros2_ctx["node"]

def _wait_for_service(node: Node, client, service_name: str) -> None:
    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info(f"Waiting for service {service_name}...")

def _call_service(node: Node, client, request):
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.exception() is not None:
        raise future.exception()
    return future.result()

@trace_call("navigate_then_observe")
@_track_skill_timing
def navigate(pos: List[float], theta: float) -> GetImageAtPoseSrv.Response:
    node = _ensure_ros2()
    service_name = "/moma/navigate"
    client = node.create_client(GetImageAtPoseSrv, service_name)
    _wait_for_service(node, client, service_name)

    req = GetImageAtPoseSrv.Request()
    req.x = float(pos[0])
    req.y = float(pos[1])
    if len(pos) == 3:
        req.z = float(pos[2])
    req.theta = float(theta)

    return _call_service(node, client, req)

@trace_call("observe")
@_track_skill_timing
def observe() -> GetImageSrv.Response:
    node = _ensure_ros2()
    service_name = "/moma/observe"
    client = node.create_client(GetImageSrv, service_name)
    _wait_for_service(node, client, service_name)

    req = GetImageSrv.Request()
    return _call_service(node, client, req)

@trace_call("pick")
@_track_skill_timing
def pick_by_instance_id(instance_id: str) -> PickObjectSrv.Response:
    node = _ensure_ros2()
    service_name = "/moma/pick_object"
    client = node.create_client(PickObjectSrv, service_name)
    _wait_for_service(node, client, service_name)

    req = PickObjectSrv.Request()
    req.instance_id = instance_id
    return _call_service(node, client, req)

@trace_call("open")
@_track_skill_timing
def open_by_instance_id(instance_id: str) -> OpenVirtualHomeObjectSrv.Response:
    node = _ensure_ros2()
    service_name = "/moma/open_object"
    client = node.create_client(OpenVirtualHomeObjectSrv, service_name)
    _wait_for_service(node, client, service_name)

    req = OpenVirtualHomeObjectSrv.Request()
    req.instance_id = instance_id
    return _call_service(node, client, req)

@trace_call("detect")
@_track_skill_timing
def detect_virtual_home_object(query_text: str) -> DetectVirtualHomeObjectSrv.Response:
    node = _ensure_ros2()
    service_name = "/moma/detect_virtual_home_object"
    client = node.create_client(DetectVirtualHomeObjectSrv, service_name)
    _wait_for_service(node, client, service_name)

    req = DetectVirtualHomeObjectSrv.Request()
    req.query_text = query_text
    return _call_service(node, client, req)