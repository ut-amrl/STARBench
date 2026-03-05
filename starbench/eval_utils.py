import os
import pandas as pd
import glob
from pathlib import Path
import json  
from datetime import datetime, timezone
import re
from tqdm import tqdm
import sys
import importlib
import importlib.util
from typing import Any

ROS_VERSION = os.environ.get("ROS_VERSION", "").strip()
if ROS_VERSION == "2":
    USE_ROS2 = True
elif ROS_VERSION == "1":
    USE_ROS2 = False
else:
    USE_ROS2 = importlib.util.find_spec("rclpy") is not None

if USE_ROS2:
    rclpy = importlib.import_module("rclpy")
    Node = importlib.import_module("rclpy.node").Node
    ChangeVirtualHomeGraphSrv = importlib.import_module("amrl_msgs.srv").ChangeVirtualHomeGraphSrv
else:
    rospy = importlib.import_module("rospy")
    roslib = importlib.import_module("roslib")
    roslib.load_manifest("amrl_msgs")
    amrl_msgs_srv = importlib.import_module("amrl_msgs.srv")
    ChangeVirtualHomeGraphSrv = amrl_msgs_srv.ChangeVirtualHomeGraphSrv
    ChangeVirtualHomeGraphSrvRequest = amrl_msgs_srv.ChangeVirtualHomeGraphSrvRequest

_ros2_ctx = {"inited": False, "node": None}


def _ensure_ros2_node() -> Any:
    if not _ros2_ctx["inited"]:
        rclpy.init(args=None)
        _ros2_ctx["inited"] = True
    if _ros2_ctx["node"] is None:
        _ros2_ctx["node"] = rclpy.create_node("eval_utils")
    return _ros2_ctx["node"]


def _set_virtualhome_scene_ros2(graph_path: str, scene_id: int = None) -> bool:
    node = _ensure_ros2_node()
    service_name = "/moma/change_virtualhome_graph"
    client = node.create_client(ChangeVirtualHomeGraphSrv, service_name)
    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info(f"Waiting for service {service_name}...")

    request = ChangeVirtualHomeGraphSrv.Request()
    request.graph_path = graph_path
    if scene_id is not None:
        request.scene_id = scene_id

    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.exception() is not None:
        print("Service call failed:", future.exception())
        return False
    response = future.result()
    return bool(response.success)

def set_virtulhome_scene(graph_path: str, scene_id: int = None) -> bool:
    """
    Set the virtual home scene by changing the graph.
    """
    if USE_ROS2:
        return _set_virtualhome_scene_ros2(graph_path=graph_path, scene_id=scene_id)

    rospy.wait_for_service("/moma/change_virtualhome_graph", timeout=10)
    try:
        change_graph_service = rospy.ServiceProxy("/moma/change_virtualhome_graph", ChangeVirtualHomeGraphSrv)
        request = ChangeVirtualHomeGraphSrvRequest()
        request.graph_path = graph_path
        if scene_id is not None:
            request.scene_id = scene_id
        response = change_graph_service(request)
        return response.success
    except rospy.ServiceException as e:
        print("Service call failed:", e)
        return False
    
def import_ros_service_module():
    """
    Dynamically adds the virtualhome demo folder to sys.path 
    and imports the module.
    """
    # 1. Get the path of the current file (eval_utils.py)
    current_file = Path(__file__).resolve()
    
    # 2. Go up the tree to find the Project Root. 
    # Based on your image: outputs_test -> evaluation -> ROOT
    project_root = current_file.parents[2] 
    
    # 3. Construct path to the demo folder
    # Structure: root/virtualhome/virtualhome/demo
    demo_path = os.path.join(project_root, "virtualhome", "virtualhome", "demo")
    
    if not os.path.exists(demo_path):
        raise FileNotFoundError(f"Could not find virtualhome demo path at: {demo_path}")

    # 4. Add to sys.path so Python can find 'start_ros_service' AND its sibling 'ros_utils'
    sys.path.append(str(demo_path))
    
    # 5. Import the module
    import start_ros_service
    return start_ros_service

class BaseRobot:
    def __init__(self, actions):
        self.actions = actions
        self.action_trace = []

    def on_action(self, rec: dict):
        self.action_trace.append(rec)

    def run(self, task):
        raise NotImplementedError

class Evaluator:
    def __init__(self, 
                 agent_name: str,
                 benchmark_dir: str, 
                 data_dir: str, 
                 output_dir: str, 
                 task_file: str, 
                 caption_type: str,
                 captioner_type: str,
                 task_families: list = ["visible", "interactive", "common_sense"],
                 task_types: list = ["class_based", "attribute_based", "spatial", "spatial_temporal", "spatial_frequentist"],
                 force_rerun: bool = False,
                 task_uids_to_include: list = None # If specified, only include tasks with these uids
    ):
        """
        Initialize the Evaluator with configuration parameters and load metadata.
        Args:
            benchmark_dir (str): Path to the benchmark directory containing task definitions
            data_dir (str): Path to the directory containing evaluation data
            output_dir (str): Path to the directory where evaluation results will be saved
            task_file (str): Path to the task configuration file
            task_families (list): ["visible", "interactsive", "common_sense"] 
            task_types (list): ["class_based", "attribute_based", "spatial", "spatial_temporal", "spatial_frequentist"]
            caption_type (str): "oracle" or "realistic" caption type
            captioner_type (str): "gpt4o" or "molmo" captioner type
            force_rerun (bool): Whether to force re-running evaluations even if results exist
        Sets up the evaluator instance by storing configuration parameters and loading
        both task metadata and data metadata based on the provided parameters.
        """
        
        self.benchmark_dir = benchmark_dir
        if not os.path.exists(benchmark_dir):
            raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        self.task_file = task_file
        if not os.path.exists(task_file):
            raise FileNotFoundError(f"Task file not found: {task_file}")
        
        self.task_families = task_families
        self.task_types = task_types
        
        self.caption_type = caption_type
        if caption_type not in ["oracle", "realistic"]:
            raise ValueError(f"Invalid caption_type: {caption_type}. Must be 'oracle' or 'realistic'.")
        self.captioner_type = captioner_type
        if captioner_type not in ["gpt4o", "molmo"]:
            raise ValueError(f"Invalid captioner_type: {captioner_type}. Must be 'gpt4o' or 'molmo'.")
        
        self.force_rerun = force_rerun
        
        self.task_metadata = Evaluator.load_task_metadata(
            task_file_path=self.task_file, 
            benchmark_dir=self.benchmark_dir,
            task_families=self.task_families,
            task_types=self.task_types,
            task_uids_to_include=task_uids_to_include
        )
        self.data_metadata = Evaluator.load_data_metadata(
            data_dir=self.data_dir,
            caption_type=self.caption_type,
            captioner_type=self.captioner_type
        )
        
        self.agent_name = agent_name
        
        if self.force_rerun:
            for task_type, task_paths in self.task_metadata.items():
                for task_path in task_paths:
                    task_id = Path(task_path).stem
                    results_dir = os.path.join(self.output_dir, task_type, task_id)
                    # Patterns to match
                    patterns = [
                        f"results_{self.agent_name}_*.json",
                        f"{self.agent_name}_*.log"
                    ]
                    if os.path.exists(results_dir):
                        for pattern in patterns:
                            files_to_delete = glob.glob(os.path.join(results_dir, pattern))
                            for f in files_to_delete:
                                try:
                                    os.remove(f)
                                    print(f"🗑️ Deleted: {f}")
                                except Exception as e:
                                    print(f"⚠️ Failed to delete {f}: {e}")
        
        self._task_indices = self._all_task_paths()
                                    
    def __len__(self):
        """Returns the total number of tasks across all groups."""
        total_tasks = 0
        for task_list in self.task_metadata.values():
            total_tasks += len(task_list)
        return total_tasks
    
    def __iter__(self):
        for task_t, task_path in self._task_indices:
            task_m = json.load(open(task_path, 'r'))
            
            obs_stream = []
            for dataname, (_, (start_date, start_time)) in zip(task_m["bagnames"], task_m["bag_time_mapping"]):
                
                dt = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M")
                dt = dt.replace(tzinfo=timezone.utc)
                unix_time = dt.timestamp()
                
                if dataname not in self.data_metadata:
                    raise ValueError(f"Data {dataname} not found in data metadata. Please double check data integrity.")
                data_meta_path = self.data_metadata[dataname]
                if os.path.exists(data_meta_path):
                    data_m = json.load(open(data_meta_path, 'r'))   
                else:
                    raise FileNotFoundError(f"Data metadata file not found: {data_meta_path}. Please double check data integrity.")
                
                for i, dm in enumerate(data_m):
                    pos = dm["base_position"]
                    pos = [pos[0], pos[2], pos[1]] # convert from (x, z, y) to (x, y, z)
                    caption = dm["base_caption"]
                    caption_embed = dm["base_caption_embedding"]
                    
                    frame = dm["start_frame"]
                    image_path = os.path.join(self.data_dir, dataname, "0", f"Action_{frame}_0_normal.png")
                    
                    time = unix_time + float(i)
                    
                    obs = {
                        "time": time,
                        "position": pos,
                        "caption": caption,
                        "caption_embedding": caption_embed,
                        "image_path": image_path
                    }
                    obs_stream.append(obs)
                
                if len(task_m["tasks"]) != 1:
                    raise NotImplementedError("Currently only support single-task loading.")
                
            task = task_m["tasks"][0]
            
            task_desc = task["task"]
            target_obj_cls = task["instance_class"]
            target_obj_uid = task["instance_name"]
            
            bagname_current = task["bagname_current"]
            match = re.match(r'scene(\d+)_', bagname_current)
            if match:
                scene_id = int(match.group(1))
            else:
                raise ValueError(f"Could not extract scene ID from bagname: {bagname_current}")
            graph_path = os.path.join(self.data_dir, bagname_current, "0", "graph.json")
            
            if task_t == "common_sense":
                task_family = "common_sense"
                task_type = None
            elif "interactive" in task_t:
                task_family = "interactive"
                task_type = task_t.replace("_interactive", "")
            else:
                task_family = "visible"
                task_type = task_t
                
            task_uid = Path(task_path).stem
            yield {
                "task_uid": task_uid,
                "obs_stream": obs_stream,
                "task_desc": task_desc,
                "target_obj_cls": target_obj_cls,
                "target_obj_uid": target_obj_uid,
                "scene_id": scene_id,
                "graph_path": graph_path,
                "task_family": task_family,
                "task_type": task_type,
            }
                
    def __getitem__(self, idx: int):
        """Allows indexing into the Evaluator to get a specific task by index."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for Evaluator with {len(self)} tasks.")
        
        task_t, task_path = self._task_indices[idx]
        task_m = json.load(open(task_path, 'r'))
            
        obs_stream = []
        for dataname, (_, (start_date, start_time)) in zip(task_m["bagnames"], task_m["bag_time_mapping"]):
            
            dt = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M")
            dt = dt.replace(tzinfo=timezone.utc)
            unix_time = dt.timestamp()
            
            if dataname not in self.data_metadata:
                raise ValueError(f"Data {dataname} not found in data metadata. Please double check data integrity.")
            data_meta_path = self.data_metadata[dataname]
            if os.path.exists(data_meta_path):
                data_m = json.load(open(data_meta_path, 'r'))   
            else:
                raise FileNotFoundError(f"Data metadata file not found: {data_meta_path}. Please double check data integrity.")
            
            for i, dm in enumerate(data_m):
                pos = dm["base_position"]
                pos = [pos[0], pos[2], pos[1]] # convert from (x, z, y) to (x, y, z)
                caption = dm["base_caption"]
                caption_embed = dm["base_caption_embedding"]
                
                frame = dm["start_frame"]
                image_path = os.path.join(self.data_dir, dataname, "0", f"Action_{frame}_0_normal.png")
                
                time = unix_time + float(i)
                
                obs = {
                    "time": time,
                    "position": pos,
                    "caption": caption,
                    "caption_embedding": caption_embed,
                    "image_path": image_path
                }
                obs_stream.append(obs)
            
            if len(task_m["tasks"]) != 1:
                raise NotImplementedError("Currently only support single-task loading.")
            
        task = task_m["tasks"][0]
        
        task_desc = task["task"]
        target_obj_cls = task["instance_class"]
        if task_t != "common_sense":
            target_obj_uid = task["instance_name"]
        else:
            target_obj_uid = None
        
        bagname_current = task["bagname_current"]
        match = re.match(r'scene(\d+)_', bagname_current)
        if match:
            scene_id = int(match.group(1))
        else:
            raise ValueError(f"Could not extract scene ID from bagname: {bagname_current}")
        graph_path = os.path.join(self.data_dir, bagname_current, "0", "graph.json")
        
        if task_t == "common_sense":
            task_family = "common_sense"
            task_type = None
        elif "interactive" in task_t:
            task_family = "interactive"
            task_type = task_t.replace("_interactive", "")
        else:
            task_family = "visible"
            task_type = task_t
            
        task_uid = Path(task_path).stem
        return {
            "task_uid": task_uid,
            "obs_stream": obs_stream,
            "task_desc": task_desc,
            "target_obj_cls": target_obj_cls,
            "target_obj_uid": target_obj_uid,
            "scene_id": scene_id,
            "graph_path": graph_path,
            "task_family": task_family,
            "task_type": task_type,
        }
        
    def _all_task_paths(self):
        paths = []
        for task_t, task_list in self.task_metadata.items():
            for task_path in task_list:
                paths.append((task_t, task_path))
        return paths
                
    @staticmethod
    def load_task_metadata(
        task_file_path: str, 
        benchmark_dir: str, 
        task_families: list = ["visible", "interactive", "common_sense"], 
        task_types: list = ["class_based", "attribute_based", "spatial", "spatial_temporal", "spatial_frequentist"],
        task_uids_to_include: list = None
    ):
        
        # 1. Initialize allowed keys based on input arguments
        task_groups = []
        if "visible" in task_families and task_types:
            task_groups += task_types
        if "interactive" in task_families and task_types:
            task_groups += [f"{task_type}_interactive" for task_type in task_types]
        if "common_sense" in task_families:
            task_groups += ["common_sense"]
            
        # Initialize dict with specific allowed keys to ensure we only load what is requested
        task_dict = {k: [] for k in task_groups}
            
        # 2. Read CSV
        if not os.path.exists(task_file_path):
            raise FileNotFoundError(f"Task summary file not found: {task_file_path}")
            
        df = pd.read_csv(task_file_path)
        
        # 3. Iterate and populate
        for _, row in df.iterrows():
            uid = row['task_uid']
            if task_uids_to_include is not None and uid not in task_uids_to_include:
                continue
            family = row['task_family']
            t_type = row['task_type']
            
            # Determine the key based on family logic
            key = None
            if family == 'visible':
                key = t_type
            elif family == 'interactive':
                key = f"{t_type}_interactive"
            elif family == 'common_sense':
                key = "common_sense"
            
            # Only add if this key is in our initialized valid groups
            if key in task_dict:
                # Construct the full path: <benchmark_dir>/<task_uid>.json
                json_path = os.path.join(benchmark_dir, key, f"{uid}.json")
                task_dict[key].append(json_path)
                
        return task_dict
    
    @staticmethod
    def load_data_metadata(data_dir: str, caption_type: str, captioner_type: str):
        """_summary_

        Args:
            data_dir (str): path to data directory
            caption_type (str): "gt" (for oracle) or "caption" (for realistic)
            captioner_type (str): "gpt4o" or "molmo"

        Returns:
            dict: mapping from dataname to metadata files
        """
        caption_t = "gt" if caption_type == "oracle" else "nframe1"
        
        result = {}
        for dataname in os.listdir(data_dir):
            data_path = os.path.join(data_dir, dataname)
            if not os.path.isdir(data_path):
                continue  # skip non-directory entries
            simulation_data_dir = os.path.join(data_path, "0")

            caption_file = os.path.join(simulation_data_dir, f"caption_{captioner_type}_{caption_t}.json")
            if os.path.exists(caption_file):
                result[dataname] = caption_file
        return result
    
    def before_evaluate_one_task(self, task: dict):
        if not set_virtulhome_scene(graph_path=task["graph_path"], scene_id=task["scene_id"]):
            raise RuntimeError(f"Failed to set VirtualHome scene for scene_id: {task['scene_id']}. Please make sure the VirtualHome simulator is running and the graph path is correct.")
    
    def after_evaluate_one_task(self, task: dict, result: dict):
        return task["target_obj_uid"] == result.get("picked_obj", "")
    
    def evaluate_one_task(self, robot: BaseRobot, task: dict):
        """
        Evaluate a single task. This method must be implemented by subclasses.
        
        Args:
            task (dict): Task dictionary containing task information
            
        Raises:
            NotImplementedError: This method must be overridden by subclasses
        """
        raise NotImplementedError("Subclasses must implement evaluate_one_task method")
    
    def evaluate(self, robot: BaseRobot):
        """
        Evaluate all tasks in the evaluator.
        """
        n_success = 0
        n_evaluated = 0
        for task in tqdm(self, total=len(self), desc="Evaluating tasks"):
            self.before_evaluate_one_task(task)
            result = self.evaluate_one_task(task)
            success = self.after_evaluate_one_task(task, result)
            n_success += int(success)
            n_evaluated += 1
            success_rate = n_success / n_evaluated * 100
            tqdm.write(f"Success rate so far: {success_rate:.1f}% ({n_success}/{n_evaluated})")
            