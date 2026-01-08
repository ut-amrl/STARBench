from typing import List

import rospy
import roslib; roslib.load_manifest('amrl_msgs')
from amrl_msgs.srv import (
    GetImageSrv,
    GetImageSrvResponse,
    GetImageAtPoseSrv, 
    GetImageAtPoseSrvRequest, 
    GetImageAtPoseSrvResponse, 
    PickObjectSrv, 
    PickObjectSrvResponse,
    DetectVirtualHomeObjectSrv,
    DetectVirtualHomeObjectSrvRequest,
    DetectVirtualHomeObjectSrvResponse,
    OpenVirtualHomeObjectSrv,
)
from starbench.tracing import trace_call

@trace_call("navigate_then_observe")
def navigate(pos: List[float], theta: float) -> GetImageAtPoseSrvResponse:
    """
    Navigate to a specific position and orientation.
    """
    rospy.wait_for_service("/moma/navigate")
    try:
        navigate_service = rospy.ServiceProxy("/moma/navigate", GetImageAtPoseSrv)
        request = GetImageAtPoseSrvRequest()
        request.x = pos[0]
        request.y = pos[1]
        if len(pos) == 3:
            request.z = pos[2]
        request.theta = theta
        response = navigate_service(request)
        return response
    except rospy.ServiceException as e:
        print("Service call failed:", e)
        
@trace_call("observe")
def observe() -> GetImageSrvResponse:
    """
    Observe the current environment.
    """
    rospy.wait_for_service("/moma/observe")
    try:
        observe_service = rospy.ServiceProxy("/moma/observe", GetImageSrv)
        response = observe_service()
        return response
    except rospy.ServiceException as e:
        print("Service call failed:", e)
        
@trace_call("pick")
def pick_by_instance_id(instance_id: str) -> PickObjectSrvResponse:
    """
    Pick an object by its instance ID.
    """
    rospy.wait_for_service("/moma/pick_object")
    try:
        pick_service = rospy.ServiceProxy("/moma/pick_object", PickObjectSrv)
        response = pick_service(instance_id=instance_id)
        return response
    except rospy.ServiceException as e:
        print("Service call failed:", e)
        
@trace_call("open")
def open_by_instance_id(instance_id: str) -> PickObjectSrvResponse:
    """
    Open an object by its instance ID.
    """
    rospy.wait_for_service("/moma/open_object")
    try:
        open_service = rospy.ServiceProxy("/moma/open_object", OpenVirtualHomeObjectSrv)
        response = open_service(instance_id=instance_id)
        return response
    except rospy.ServiceException as e:
        print("Service call failed:", e)
        
@trace_call("detect")
def detect_virtual_home_object(query_text: str) -> DetectVirtualHomeObjectSrvResponse:
    """
    Detect a virtual home object by its class label.
    """
    rospy.wait_for_service("/moma/detect_virtual_home_object")
    try:
        detect_service = rospy.ServiceProxy("/moma/detect_virtual_home_object", DetectVirtualHomeObjectSrv)
        req = DetectVirtualHomeObjectSrvRequest()
        req.query_text = query_text
        response = detect_service(req)
        return response
    except rospy.ServiceException as e:
        print("Service call failed:", e)
        
