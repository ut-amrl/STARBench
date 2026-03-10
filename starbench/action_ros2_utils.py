from __future__ import annotations

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
def observe() -> GetImageSrv.Response:
    node = _ensure_ros2()
    service_name = "/moma/observe"
    client = node.create_client(GetImageSrv, service_name)
    _wait_for_service(node, client, service_name)

    req = GetImageSrv.Request()
    return _call_service(node, client, req)

@trace_call("pick")
def pick_by_instance_id(instance_id: str) -> PickObjectSrv.Response:
    node = _ensure_ros2()
    service_name = "/moma/pick_object"
    client = node.create_client(PickObjectSrv, service_name)
    _wait_for_service(node, client, service_name)

    req = PickObjectSrv.Request()
    req.instance_id = instance_id
    return _call_service(node, client, req)

@trace_call("open")
def open_by_instance_id(instance_id: str) -> OpenVirtualHomeObjectSrv.Response:
    node = _ensure_ros2()
    service_name = "/moma/open_object"
    client = node.create_client(OpenVirtualHomeObjectSrv, service_name)
    _wait_for_service(node, client, service_name)

    req = OpenVirtualHomeObjectSrv.Request()
    req.instance_id = instance_id
    return _call_service(node, client, req)

@trace_call("detect")
def detect_virtual_home_object(query_text: str) -> DetectVirtualHomeObjectSrv.Response:
    node = _ensure_ros2()
    service_name = "/moma/detect_virtual_home_object"
    client = node.create_client(DetectVirtualHomeObjectSrv, service_name)
    _wait_for_service(node, client, service_name)

    req = DetectVirtualHomeObjectSrv.Request()
    req.query_text = query_text
    return _call_service(node, client, req)