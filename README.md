# STARBench: Benchmarking long-horizon spatio-temporal object search

<div align="center">

[![Website](https://img.shields.io/badge/Website-STAR-blue.svg)](https://amrl.cs.utexas.edu/STAR/benchmark.html)
[![arXiv](https://img.shields.io/badge/arXiv-2511.14004-b31b1b.svg)](https://arxiv.org/abs/2511.14004)

<br/><br/>

<img src="assets/benchmark.gif" width="900" alt="STARBench overview">

</div>




## Repo Structure
```
starbench/
├── starbench_data/ # default data download directory
├── starbench_tasks/ # default task metadata download directory
├── starbench/ # Python evaluation package 
├── scripts/ # Setup + data helper scripts
│ ├── install.sh
│ ├── install_virtualhome.sh
│ └── download.sh
├── virtualhome/ # VirtualHome dependency 
├── starbench_example.py # Minimal example 
├── pyproject.toml # Python build + dependencies
└── README.md
```

## Installation

### Prerequisites
- Python (3.10+ recommended)
- (Optional) Docker — only required if you want to run VirtualHome through the provided installation script.


### 1) Clone the repo:
```bash
git clone https://github.com/ut-amrl/STARBench.git
cd STARBench
git submodule update --init --recursive
```

### 2) Install STARBench:
**Option A: quick installation for everything**
```bash
bash scripts/install.sh
```
This script would also install [VirtualHome](https://github.com/xavierpuigf/virtualhome) through docker. If you prefer installing VirtualHome using other method, see more details on their [webpage](https://github.com/xavierpuigf/virtualhome).

**Option B: install STARBench only (no VirtualHome)**
```bash
bash scripts/install_starbench.sh
```

### 3) Download data and tasks
```bash
bash scripts/download_data.sh
```
This script would download data to `starbench_dta` and `starbench_tasks` by default.

## Running the benchmark

### Step 1 - Start the simulator
To start the simulation in docker:
```bash
cd virtualhome
podman run --name virtualhome_container \
      --mount type=bind,source="$(pwd)"/unity_vol,target=/unity_vol/ \
      --mount type=bind,source="$(pwd)"/unity_output,target=/Output/ \
      -p 8080:8080 -it virtualhome
```

If you saw error: `Error: rootlessport listen tcp 0.0.0.0:8080: bind: address already in use`, run:
```bash
lsof -i :8080
```
You'll see output like:
```
COMMAND     PID    USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
your_app   12345  user   ...  TCP  ...           0   LISTEN 0.0.0.0:8080
```
Kill this process by running:
```
kill -9 <PID>
```
and restart the container.

### Step 2 - Running the benchmark
Checkout `example.py` script for details.

**Plug in your algorithm (replace `BaseRobot`)**:`example.py` uses a BaseRobot(actions=...) placeholder. Replace it with your own robot implementation.

Your agent is expected to call the following primitive actions (provided in `starbench.action_utils`):
- navigate_then_observe
- detect
- pick
- open

During evaluation, STARBench traces actions via `task_context(..., sink=robot.on_action, stop_after={"pick"})`, and the episode terminates after pick is attempted.

To start evaluation, in another terminal:
```bash
python example.py \
  --agent-name <NAME_OF_ALGORITHM> \
  --benchmark-dir <PATH_TO_BENCHMARK_DIR such as starbench_tasks> \
  --data-dir <PATH_TO_DATA_DIR such as starbench_data> \
  --task-file <PATH_TO_TASK_SUMMARY_CSV such as starbench_tasks/tasks_summary.csv> \
  --output-dir <PATH_TO_OUTPUT_DIR> \
  --port 18080
```



## Citation
```
@misc{chen2025searchingspacetimeunified,
      title={Searching in Space and Time: Unified Memory-Action Loops for Open-World Object Retrieval}, 
      author={Taijing Chen and Sateesh Kumar and Junhong Xu and George Pavlakos and J oydeep Biswas and Roberto Martín-Martín},
      year={2025},
      eprint={2511.14004},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2511.14004}, 
}
```