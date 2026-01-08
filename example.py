from starbench.action_utils import *
from starbench.eval_utils import Evaluator, BaseRobot

from starbench.tracing import task_context, TerminateEpisode, get_task_trace, clear_task_trace

import argparse
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate benchmark results")
    
    parser.add_argument(
        "--port",
        type=str,
        default="18080",
        help="Port for the server"
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        required=True,
        help="Directory containing benchmark files"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing data files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save output results"
    )
    parser.add_argument(
        "--task-file",
        type=str,
        required=True,
        help="Path to the task summary CSV file"
    )
    parser.add_argument(
        "--caption-type",
        type=str,
        default="oracle",
        help="Type of caption to use"
    )
    parser.add_argument(
        "--captioner-type",
        type=str,
        default="gpt4o",
        help="Type of captioner to use"
    )
    parser.add_argument(
        "--agent-name",
        type=str,
        required=True,
        help="Name of the agent"
    )
    return parser.parse_args()

import math
def _mean(xs):
    xs = list(xs)
    return (sum(xs) / len(xs)) if xs else float("nan")

def extract_counts(r: dict):
    n_nav = r.get("n_navigate", 0)
    n_det = r.get("n_detect", 0)
    n_pick = r.get("n_pick", 0)
    n_open = r.get("n_open", 0)
    return n_nav, n_det, n_pick, n_open

def summarize_bucket(results_list, successes_list):
    # Success rate
    # (expects successes_list contains 0/1 per task; if it's cumulative, fix at collection time)
    succ = [int(x) for x in successes_list]
    success_rate = 100.0 * (_mean(succ)) if succ else float("nan")

    # Action counts
    navs, dets, picks, opens, totals = [], [], [], [], []
    for r in results_list:
        n_nav, n_det, n_pick, n_open = extract_counts(r)
        navs.append(n_nav)
        dets.append(n_det)
        picks.append(n_pick)
        opens.append(n_open)
        totals.append(n_nav + n_det + n_pick + n_open)

    return {
        "n_tasks": len(results_list),
        "success_rate": success_rate,
        "avg_nav": _mean(navs),
        "avg_detect": _mean(dets),
        "avg_pick": _mean(picks),
        "avg_open": _mean(opens),
        "avg_total_actions": _mean(totals),
    }

def print_report(execution_results, execution_successes):
    print("\n=== Execution Summary ===")
    # common_sense
    if "common_sense" in execution_results:
        s = summarize_bucket(
            execution_results["common_sense"],
            execution_successes["common_sense"],
        )
        print(f"\n[common_sense] tasks={s['n_tasks']}  success={s['success_rate']:.1f}%")
        print(f"  avg nav={s['avg_nav']:.2f}  detect={s['avg_detect']:.2f}  pick={s['avg_pick']:.2f}  open={s['avg_open']:.2f}  total={s['avg_total_actions']:.2f}")

    # visible / interactive families (and any other top-level groups besides common_sense)
    for top_key, sub in execution_results.items():
        if top_key == "common_sense":
            continue
        if not isinstance(sub, dict):
            continue

        for task_family, results_list in sub.items():
            successes_list = execution_successes.get(top_key, {}).get(task_family, [])
            s = summarize_bucket(results_list, successes_list)
            print(f"\n[{top_key}/{task_family}] tasks={s['n_tasks']}  success={s['success_rate']:.1f}%")
            print(f"  avg nav={s['avg_nav']:.2f}  detect={s['avg_detect']:.2f}  pick={s['avg_pick']:.2f}  open={s['avg_open']:.2f}  total={s['avg_total_actions']:.2f}")


if __name__ == "__main__":
    args = parse_args()
    
    evaluator = Evaluator(
        agent_name=args.agent_name,
        benchmark_dir=args.benchmark_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        task_file=args.task_file,
        caption_type=args.caption_type,
        captioner_type=args.captioner_type,
    )
    
    actions = {
        "navigate_then_observe": navigate,
        "pick": pick_by_instance_id,
        "open": open_by_instance_id,
        "detect": detect_virtual_home_object,
    }
    
    n_success = 0
    n_evaluated = 0
    
    execution_results = {
        "visible": {
            "class_based": [],
            "attribute_based": [],
            "spatial": [],
            "spatial_temporal": [],
            "spatial_frequentist": []    
        },
        "interactive": {
            "class_based": [],
            "attribute_based": [],
            "spatial": [],
            "spatial_temporal": [],
            "spatial_frequentist": []    
        },
        "common_sense": []
    }
    execution_successes = {
        "visible": {
            "class_based": [],
            "attribute_based": [],
            "spatial": [],
            "spatial_temporal": [],
            "spatial_frequentist": []    
        },
        "interactive": {
            "class_based": [],
            "attribute_based": [],
            "spatial": [],
            "spatial_temporal": [],
            "spatial_frequentist": []    
        },
        "common_sense": []
    }
    
    from itertools import islice
    for task in tqdm(islice(evaluator, 6, 14), total=14-6, desc="Evaluating tasks"):
        task_uid = task["task_uid"]
        
        robot = BaseRobot(actions=actions) # TODO replace with specific algorithm
        task_uid = task["task_uid"]
        
        task_family = task["task_family"]
        task_type = task["task_type"]
        
        result_dir = os.path.join(args.output_dir, task_family, task_type, task_uid)
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, task_family), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, task_family, task_type), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, task_family, task_type, task_uid), exist_ok=True)
        result_path = os.path.join(result_dir, f"results_{args.agent_name}_{task_uid}.json")
        
        try:
            with task_context(task_uid, sink=robot.on_action, stop_after={"pick"}):
                robot.run(task) # TODO replace with specific algorithm
        except TerminateEpisode:
            pass # This is the intended stop condition (pick was attempted)
        
        trace = get_task_trace("NO_TASK")
        
        parsed_result = {}
        if len(trace) == 0:
            pass
        if len(trace) > 0 and trace[-1]["action"] == "pick":
            if trace[-1]["result"].success:
                parsed_result["picked_obj"] = trace[-1]["result"].instance_uid
        parsed_result["n_navigate"] = 0
        parsed_result["n_pick"] = 0
        parsed_result["n_open"] = 0
        parsed_result["n_detect"] = 0
        for step in trace:
            if  "navigate" in step["action"]:
                parsed_result["n_navigate"] += 1
            elif step["action"] == "pick":
                parsed_result["n_pick"] += 1
            elif step["action"] == "open":
                parsed_result["n_open"] += 1
            elif step["action"] == "detect":
                parsed_result["n_detect"] += 1
        
        clear_task_trace("NO_TASK")
        
        success = evaluator.after_evaluate_one_task(task, parsed_result)
        n_success += int(success)
        n_evaluated += 1
        success_rate = n_success / n_evaluated * 100
        tqdm.write(f"Success rate so far: {success_rate:.1f}% ({n_success}/{n_evaluated})")
        
        if task_type == "common_sense":
            execution_successes["common_sense"].append(n_success)
            execution_results["common_sense"].append(parsed_result)
        else:
            execution_successes[task_family][task_type].append(n_success)
            execution_results[task_family][task_type].append(parsed_result)
            
    print_report(execution_results, execution_successes)