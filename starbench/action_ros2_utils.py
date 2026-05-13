from __future__ import annotations

import sys
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

_ros2_ctx = {"inited": False, "nodes": {}, "executors": {}, "clients": {}}
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

    sys.stdout.write("\n[skill_avg] " + " | ".join(parts) + "\n")
    sys.stdout.flush()


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

def _ensure_ros2(portnum=None) -> Node:
    if not _ros2_ctx["inited"]:
        if not rclpy.ok():
            rclpy.init(args=None)
        _ros2_ctx["inited"] = True
    if portnum not in _ros2_ctx["nodes"]:
        node_name = "action_ros2_utils" if portnum is None else f"action_ros2_utils_{portnum}"
        node = rclpy.create_node(node_name)
        _ros2_ctx["nodes"][portnum] = node
        
        from rclpy.executors import SingleThreadedExecutor
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        _ros2_ctx["executors"][portnum] = executor
    return _ros2_ctx["nodes"][portnum]

def _wait_for_service(node: Node, client, service_name: str) -> None:
    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info(f"Waiting for service {service_name}...")

def _get_client(node: Node, srv_type, service_name: str, portnum=None):
    key = (portnum, service_name)
    if key not in _ros2_ctx["clients"]:
        client = node.create_client(srv_type, service_name)
        _wait_for_service(node, client, service_name)
        _ros2_ctx["clients"][key] = client
    return _ros2_ctx["clients"][key]

def _call_service(node: Node, client, request, portnum=None):
    future = client.call_async(request)
    if "executors" in _ros2_ctx and portnum in _ros2_ctx["executors"]:
        _ros2_ctx["executors"][portnum].spin_until_future_complete(future)
    else:
        rclpy.spin_until_future_complete(node, future)
        
    if future.exception() is not None:
        raise future.exception()
    return future.result()

@trace_call("navigate_then_observe")
@_track_skill_timing
def navigate(pos: List[float], theta: float, portnum=None) -> GetImageAtPoseSrv.Response:
    node = _ensure_ros2(portnum)
    service_prefix = f"moma_{portnum}" if portnum is not None else "moma"
    service_name = f"/{service_prefix}/navigate"
    client = _get_client(node, GetImageAtPoseSrv, service_name, portnum=portnum)

    req = GetImageAtPoseSrv.Request()
    req.x = float(pos[0])
    req.y = float(pos[1])
    if len(pos) == 3:
        req.z = float(pos[2])
    req.theta = float(theta)

    return _call_service(node, client, req, portnum=portnum)

@trace_call("observe")
@_track_skill_timing
def observe(portnum=None) -> GetImageSrv.Response:
    node = _ensure_ros2(portnum)
    service_prefix = f"moma_{portnum}" if portnum is not None else "moma"
    service_name = f"/{service_prefix}/observe"
    client = _get_client(node, GetImageSrv, service_name, portnum=portnum)

    req = GetImageSrv.Request()
    return _call_service(node, client, req, portnum=portnum)

@trace_call("pick")
@_track_skill_timing
def pick_by_instance_id(instance_id: str, portnum=None) -> PickObjectSrv.Response:
    node = _ensure_ros2(portnum)
    service_prefix = f"moma_{portnum}" if portnum is not None else "moma"
    service_name = f"/{service_prefix}/pick_object"
    client = _get_client(node, PickObjectSrv, service_name, portnum=portnum)

    req = PickObjectSrv.Request()
    req.instance_id = instance_id
    return _call_service(node, client, req, portnum=portnum)

@trace_call("open")
@_track_skill_timing
def open_by_instance_id(instance_id: str, portnum=None) -> OpenVirtualHomeObjectSrv.Response:
    node = _ensure_ros2(portnum)
    service_prefix = f"moma_{portnum}" if portnum is not None else "moma"
    service_name = f"/{service_prefix}/open_object"
    client = _get_client(node, OpenVirtualHomeObjectSrv, service_name, portnum=portnum)

    req = OpenVirtualHomeObjectSrv.Request()
    req.instance_id = instance_id
    return _call_service(node, client, req, portnum=portnum)

@trace_call("detect")
@_track_skill_timing
def detect_virtual_home_object(query_text: str, portnum=None) -> DetectVirtualHomeObjectSrv.Response:
    node = _ensure_ros2(portnum)
    service_prefix = f"moma_{portnum}" if portnum is not None else "moma"
    service_name = f"/{service_prefix}/detect_virtual_home_object"
    client = _get_client(node, DetectVirtualHomeObjectSrv, service_name, portnum=portnum)

    req = DetectVirtualHomeObjectSrv.Request()
    req.query_text = query_text
    return _call_service(node, client, req, portnum=portnum)