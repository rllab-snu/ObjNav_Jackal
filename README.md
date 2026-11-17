# ObjNav_Jackal

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Deployment code for running Object-Goal Navigation on a real Clearpath **Jackal**.
The robot starts from an unknown pose and finds a named object as follows:

1. It localizes itself on a prior map with a particle filter (global localization), spinning in place and moving safely.
2. It picks views to inspect from a topology graph and a per-object prior score.
3. It detects the target object with SAM 3.
4. It approaches the detected object (last mile) and stops.

All computation runs on a **client PC** with a GPU. The client talks to the Jackal over a **single SSH connection**:
it receives odometry, RGB and depth, and sends motion commands (0.25 m forward, 30° left/right turns).
ROS is not needed on the client.

The map (`occupancy_map/`), the topology and the search scores (`topo_data/`) needed to run the robot are included.
Follow Section 5 (occupancy map, needs a Habitat environment) and Section 6 (topology and Qwen scoring) only if
you need to regenerate them.

```
client PC (GPU, conda env)                    Jackal (ROS Melodic)
┌──────────────────────────────┐    SSH     ┌──────────────────────────────┐
│ robot/run_robot.py           │◄──────────►│ robot/jackal_bridge.py       │
│  localizer · planner · SAM 3 │            │  (sent over SSH and run)     │
└──────────────────────────────┘            │  /odometry/filtered, /cmd_vel│
                                            │  dual D455 camera topics     │
                                            └──────────────────────────────┘
```

---

## 1. Requirements

### Client PC
- Linux x86_64, NVIDIA GPU with a driver that supports CUDA 12.8
- conda (Miniconda or Anaconda), git
- Same network as the Jackal
- A display, if you want the live view (`--render`)
- (Only to regenerate the map) an NVIDIA GPU with EGL support, in a separate conda environment on the same PC (Section 5)

### Jackal (robot)
Nothing has to be installed by this repository, but the robot must provide the following.

| Item | Default | Notes |
|---|---|---|
| ROS | `/opt/ros/melodic/setup.bash` | ROS Melodic (Python 2). The bridge imports `rospy`, `cv2`, `numpy`, `nav_msgs`, `geometry_msgs`, `sensor_msgs`. |
| catkin workspace | `/home/jackal/catkin_ws/devel/setup.bash` | Workspace that contains the camera launch package |
| Camera launch | `dual_d455.launch` from the `jackal_dual_d455` package (node `/dual_d455`) | Started automatically if the camera node is not running |
| Topics | `/odometry/filtered`, `/cmd_vel`, `/camera_{1,2}/color/image_raw`, `/camera_{1,2}/aligned_depth_to_color/image_raw` | Camera topics are set in `configs/jackal.yaml` |
| SSH account | user `jackal`; **password is not stored in this repository** | See Section 2.6 |

> The paths above are **paths on the robot**, so they are absolute on purpose. If your robot is set up differently,
> edit `ros_setup`, `workspace_setup`, `camera_launch_package`, `camera_launch_file`, `camera_node`, `ssh_user`
> and `ssh_password` in `JackalConnectionConfig` in `robot/jackal.py`.

---

## 2. Installation (fresh environment)

Clone the repository and run every command from its **root** (`ObjNav_Jackal/`).
```bash
git clone https://github.com/rllab-snu/ObjNav_Jackal.git
cd ObjNav_Jackal
```

### 2.1 Create the conda environment
```bash
conda create -n objnav_jackal python=3.12 -y
conda activate objnav_jackal
```

### 2.2 Install PyTorch (CUDA 12.8)
```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```
For a different CUDA version, pick the matching index URL on [pytorch.org](https://pytorch.org/get-started/locally/).

### 2.3 Install SAM 3
Clone the SAM 3 source into `third_party/sam3` inside the repository. The code uses this **relative path** by default.
```bash
git clone https://github.com/facebookresearch/sam3.git third_party/sam3
git -C third_party/sam3 checkout 96914d2      # commit used for validation
pip install "setuptools<81"                   # sam3 uses pkg_resources
pip install -e third_party/sam3
```

### 2.4 Remaining dependencies
```bash
pip install -r requirements.txt
```
SAM 3 requires `numpy<2`, so `requirements.txt` pins `numpy==1.26.4` and `opencv-python<4.10`.

### 2.5 SAM 3 weights (Hugging Face access)
The SAM 3 weights (`facebook/sam3`) are a gated model.
1. Request access to [facebook/sam3](https://huggingface.co/facebook/sam3) on Hugging Face and wait for approval.
2. Log in on the client:
   ```bash
   hf auth login        # or: export HF_TOKEN=<your token>
   ```
The weights are downloaded on first use into `~/.cache/huggingface/`. Download them once while you have internet
access, before going to the robot (the check in Section 3.2 does this).

### 2.6 SSH connection to the Jackal
The client connects with `paramiko`. The default host name is `jackalw`; change it with `--jackal-host`.

1. Add an alias to `~/.ssh/config`, or pass the IP address with `--jackal-host` instead.
   ```
   Host jackalw
       HostName 192.168.0.xxx      # Jackal IP
       User jackal
       # IdentityFile ~/.ssh/id_ed25519   # for key-based authentication
   ```
2. **Connect once by hand** to add the host key to `~/.ssh/known_hosts`. The code rejects unknown host keys
   (`RejectPolicy`), so the connection fails if you skip this step.
   ```bash
   ssh jackalw
   ```
3. **Authentication.** Key-based authentication is recommended: the client tries the `IdentityFile` from
   `~/.ssh/config`, the ssh-agent and your default keys.
   The SSH password is **intentionally left empty** in this repository:
   ```python
   # robot/jackal.py, JackalConnectionConfig
   ssh_password: Optional[str] = ""
   ```
   To use password authentication, **fill in the robot's password there before use**, and do not commit it.
   With an empty password only key-based authentication is attempted.

---

## 3. Configuration and checks

### 3.1 `configs/jackal.yaml`
Set this file up once per robot. The tag at the end of each line means:
`[required]` the value must be present in this file, `[default]` same as the code default,
`[robot override]` changed for the robot.

| Block | What to check |
|---|---|
| `cameras.main` | Camera used for localization and detection: topics, resolution, intrinsics (`fx/fy/cx/cy`), `depth_scale` (`0.001` for 16-bit mm), mounting height and tilt |
| `cameras.sub` | Higher camera used only for detection (optional). Delete the block if you do not use it. |
| `forward_offset_m`, `right_offset_m` | Distance from the robot's rotation center to the camera's optical center (m, forward/right positive). Measured on the Daejeon Jackal: `forward_offset_m: 0.15`, `right_offset_m: 0.0` for both cameras. **Re-measure for a different camera mount.** |
| `cameras.main.height_m`, `tilt_deg` | Must match the real mounting. Localization converts depth to world height with these values and compares only the same obstacle height band as the map (0.2–1.5 m above the floor). |
| `map.path` | Occupancy map used for localization and path planning (`occupancy_map/base_occ_map.dat.gz`) |
| `search.*` | Topology and prior score files used for search |
| `drive.*` | Speed and acceleration tuning |

Paths in the config (`map.path`, `search.prior_scores`, `sam3.bpe_path`, ...) are **relative to the repository root**,
so always run `run_robot` from the repository root.

### 3.2 Installation check (no robot needed)
Before connecting to the robot, check that the environment and data files are complete.
```bash
# (1) import every module, build the agent, and run a few decision steps on fake observations
python - <<'EOF'
import numpy as np
import robot.run_robot, robot.jackal, robot.render, robot.recording
from robot.config import RobotConfig
from robot.agent import RobotAgent
from robot.run_robot import configure_topology_search
cfg = RobotConfig.load("configs/jackal.yaml")
configure_topology_search(cfg, 1)
cfg.detection.detector = "none"
agent = RobotAgent(cfg); agent.reset(category="mug")
obs = {}
for cam in cfg.cameras:
    s = "" if cam.name == "main" else "_top"
    obs["rgb" + s] = np.zeros((cam.height, cam.width, 3), np.uint8)
    obs["depth" + s] = np.full((cam.height, cam.width, 1), 0.3, np.float32)
agent.set_odom_delta((0.0, 0.0, 0.0))
print("actions:", [agent.get_action(obs) for _ in range(5)])
EOF

# (2) load SAM 3 and run inference (downloads the weights on first run)
python - <<'EOF'
import numpy as np
from robot.sam3_detector import Sam3Detector
det = Sam3Detector("third_party/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz", device="cuda",
                   confidence_threshold=0.5)
out = det.segment(np.zeros((480, 640, 3), np.uint8), "mug")
print("SAM 3 OK, detections:", out["masks"].shape[0])
EOF
```

---

## 4. Running

Activate the conda environment and run from the repository root:
```bash
conda activate objnav_jackal
python -m robot.run_robot --goal <object> [--prior {1,2,3}] [options]
```

### 4.1 Options

| Option | Default | Description |
|---|---|---|
| `--goal` | none | Target object category (table in 4.2). Effectively required. |
| `--prior {1,2,3}` | none | Use a scenario prior topology (`topo_data/topo_graph/priorN_topology.pkl`). The robot first visits, **once**, the view where the target object was seen in that scenario, then switches to score search. Without it, the robot loads `base_topology.pkl` and starts score search directly. |
| `--jackal-host` | `jackalw` | SSH host name (alias or IP) |
| `--max_steps` | `500` | Maximum number of actions; exceeding it ends the run as a failure |
| `--render` | off | Live visualization window. Needs a display; press `Esc` to stop. |
| `--record [DIR]` | off | Save a run record. Without `DIR` it goes to `outputs/robot_runs/<time>_<goal>_<base|priorN>/`. |
| `--config` | `configs/jackal.yaml` | Config file |

### 4.2 Target objects (`--goal`)
The keys below are mapped to phrases that SAM 3 recognizes better (`utils/sam3_prompt.py`).
Any other word is passed to SAM 3 unchanged, so a natural-language phrase also works (e.g. `--goal "red cup"`);
it has no stored prior score, so the search order is less informed.

`bottle` `bowl` `clock` `coffee_machine` `comb` `dryer` `fkiller` `laptop` `lemon` `luggage`
`mug` `orange` `phone` `pringles` `remote_controller` `rubik_cube` `sponge` `spoon` `spray`
`table_tennis_paddle` `tape` `tennis_ball` `tissuebox` `toaster` `towel`

### 4.3 Examples
```bash
# find a phone in scenario prior 1 (live view and recording)
python -m robot.run_robot --goal phone --prior 1 --render --record

# find a mug with the base topology only, record without a window
python -m robot.run_robot --goal mug --record

# connect by IP without an SSH alias, record to a chosen directory
python -m robot.run_robot --goal bowl --jackal-host 192.168.0.xxx --record runs/bowl-01
```

### 4.4 What happens during a run
1. **early**: the pose is unknown, so the robot spins in place and moves safely on a local map until the particle filter converges.
   At least 180° of rotation is required; the localization budget is 80 steps.
2. **prior** (with `--prior`): the robot goes to the recorded target view and faces that direction.
3. **score**: among unvisited views it takes the top 3 by score, visits the nearest one, and runs SAM 3 detection.
4. **lastmile**: on the first valid detection the robot approaches the object and stops within 0.5 m.

If the object is seen before convergence, the robot may approach it directly on the odometry local map (`lastmile_early`).
Each step prints `mode`, `stage`, the action, convergence and the estimated pose to the console.

### 4.5 Termination and outputs
- `>>> STOP (goal reached)` means a normal arrival.
- Exceeding `--max_steps`, a planning failure (prints `failure_reason`) or an error ends the run as a failure.
- Stop with `Esc` (`--render`) or `Ctrl-C`; a stop command is sent to the robot on exit.
- Contents of the `--record` directory:
  - `metadata.json`: run configuration, result, termination reason
  - `steps.jsonl`: per-step action, estimated pose, odometry, search state
  - `run.log`: console log
  - `render.mp4`: visualization video (H.264)
  - `frames/`: per-step main/sub RGB, raw depth (uint16), SAM 3 masks

---

## 5. Building the occupancy map (optional)

The robot uses a single map, `occupancy_map/base_occ_map.dat.gz`, for localization (scan matching) and FMM path planning.
It is included, so **skip this section if you only run the robot.** Use it to change map settings or rebuild the map
with the same procedure.

### 5.1 How the map is built
The map is built in simulation. The **object-free base scene** of the Daejeon building shipped in `habitat_data/`
is loaded into Habitat, followed by two steps.

1. **Collection** (`collect_occupancy_map.py`)
   - The agent teleports to 400 points on the navmesh and captures depth while turning 12 × 30° at each point.
   - Depth is converted to world coordinates and accumulated into two 5 cm grids:
     - `obstacle_map`: geometry 0.2–1.5 m above the floor
     - `seen_map`: observed area
   - The obstacle height band is relative to the floor and equals the band localization uses for its scans.
2. **Manual edits** (`apply_occ_map_edits.py` + `occupancy_map/base_map_edits.json`)
   - **116 obstacle cells added**: the bed at the top right, a real obstacle that the scans only see sparsely.
   - **520 keep-out cells**: the entrance at the bottom center, so the robot never enters or routes through it.
     It is blocked for planning only (excluded from FMM paths, particle initialization and exploration goals);
     scan matching still treats it as floor.
   - Edits are stored as world `[x, z]` cell centers, so they apply at the same place on a newly collected map.

> The included `base_occ_map.dat.gz` was made with this procedure. Applying the edits to the collected map reproduces
> the included file **exactly**. Re-running the collection step today differs in 55 obstacle cells and 40 seen cells
> (about 0.1% of 37,370 cells) because of sampling differences; the map size, origin and keep-out area are the same.
> Repeated runs in the same environment always give the same result.

### 5.2 Habitat environment (separate from the robot environment)
habitat-sim 0.3.3 is distributed for Python 3.9, so create a **separate conda environment** from `objnav_jackal`.
An NVIDIA GPU with EGL is required; no display is needed (headless).
```bash
conda create -n objnav_jackal_sim python=3.9 cmake=3.14.0 -y
conda activate objnav_jackal_sim
conda install habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat -y

# habitat-lab is cloned into third_party/ inside the repository (git-ignored)
git clone --branch v0.3.3 --depth 1 https://github.com/facebookresearch/habitat-lab.git third_party/habitat-lab
pip install -e third_party/habitat-lab/habitat-lab

pip install -r requirements-sim.txt
```
Check the installation:
```bash
python -c "import habitat_sim, habitat; print(habitat_sim.__version__, habitat.__version__)"   # 0.3.3 0.3.3
```

### 5.3 Build the map
Run from the repository root, because the paths in `configs/daejeon_base.yaml` and `habitat_data/` are relative to it.
```bash
conda activate objnav_jackal_sim

# (1) collect: output_maps/occupancy_map/val/base_0_occ_map.dat.gz (+ base_0_occ.png), about 25 s on a GPU
python collect_occupancy_map.py --no_objects --num_points 400

# (2) apply the manual edits: robot map (+ a .png preview with the same name)
python apply_occ_map_edits.py \
    --src output_maps/occupancy_map/val/base_0_occ_map.dat.gz \
    --out occupancy_map/base_occ_map_new.dat.gz
```
To use the new map on the robot, point `map.path` in `configs/jackal.yaml` to it or overwrite the existing file.
In the PNG, white is unobserved, gray is free and black is obstacle (keep-out is not drawn).

Main options of `collect_occupancy_map.py`:

| Option | Default | Description |
|---|---|---|
| `--config` | `daejeon_base.yaml` | Habitat config in `configs/` |
| `--no_objects` | off | Removes scene objects and recomputes the navmesh. The shipped base scene has no objects, but always pass it so the navmesh matches the included map. |
| `--num_points` | `300` | Number of teleport points; the included map uses `400` |
| `--num_rotations` | `0` (= 360°/30° = 12) | Rotations per point |
| `--obstacle_height_min`, `--obstacle_height_max` | `0.2`, agent height (`1.5`) | Height band counted as obstacle (m above floor). Changing it breaks agreement with the localization scan band (`h_min=0.2`, `h_max=1.5` in `navigation/localizer.py`). |
| `--seed` | `42` | Sampling seed |
| `--vis_video` | off | Save a video of the collection |
| `--version`, `--save_dir` | `occupancy_map`, `output_maps/{version}/{split}` | Output location |

The agent and camera height in `configs/daejeon_base.yaml` is 1.5 m. This sets the top of the obstacle band and the
navmesh height to 1.5 m and reproduces the included map most closely. It is unrelated to the robot's camera height
(1.0 m), because obstacles are recorded by world height above the floor.

### 5.4 Shipped simulator assets (`habitat_data/`)
All are real files (no symlinks).
```
habitat_data/
  scene_datasets/daejeon_base/
    daejeon_base.scene_dataset_config.json   registers stages / lighting / scenes
    stages/daejeon_base.stage_config.json    building mesh and navmesh
    assets/daejeon.glb                       Daejeon building mesh (17 MB, embedded textures)
    daejeon_base.navmesh                     navmesh (agent radius 0.1 m, height 1.5 m)
    lighting/daejeon_base.lighting_config.json
    scenes/base.scene_instance.json          object-free base scene
  datasets/objnav/daejeon_base/v1/val/       1-episode dataset needed to create the Habitat environment
```
The episode is only used to create the Habitat environment; collection samples points on the navmesh regardless of
the episode's start.

---

## 6. Building the topology and search scores (optional)

Score search (step 3 in 4.4) uses two files:
- `topo_data/topo_graph/base_topology.pkl`: positions and room types of 6 nodes, each with 12 views at 30° steps.
- `topo_data/score/scores.json`: for each object, a 0–1 score per view (node × direction) of how likely the object is there.

Both are included, so **skip this section if you only run the robot.** Use it when the house layout or the node
positions change, or when you change how scores are computed.

### 6.1 Pipeline
```
(1) robot capture       robot/run_scan_capture.py      360° spin at each node, 12-direction RGB-D
        │              → topo_data/base/node{1..6}/                                   [shipped]
(2) build topology      build_topologies.py            image-carrying topology
        │              → topo_data/base_topology.pkl                       [NOT shipped, rebuild]
(3) annotation          qwen_priors/annotate_topo_meta.py   SAM 3 furniture → view.meta["furniture"]
        │              qwen_priors/annotate_captions.py    Qwen3-VL caption → view.meta["caption"]
(4) scoring             qwen_priors/gen_direction.py   --signal vlm / --signal caption
        │              → topo_data/score/{Qwen3-VL-8B_dir/, caption_llm_dir/, room_affinity.json}  [shipped]
(5) fusion              qwen_priors/ensemble_priors.py → topo_data/score/scores.json (+ renders/)  [shipped]
(6) runtime topology    strip_topology_images.py       → topo_data/topo_graph/base_topology.pkl  [shipped]
```

### 6.2 What is shipped and how to regenerate the rest

To keep the repository small, the **image-carrying topology `topo_data/base_topology.pkl` (≈ 57 MB) is not included.**
It can be rebuilt exactly from the shipped captures in about 3 minutes:

```bash
# rebuild topo_data/base_topology.pkl (environments: Section 6.3)
python build_topologies.py                                      # (2) captures → topology, a few seconds
python qwen_priors/annotate_topo_meta.py --priors base --no-objects   # (3) SAM 3 furniture, ~1 min
python qwen_priors/annotate_captions.py --priors base           # (3) Qwen3-VL captions, ~1 min
```

| Artifact | Shipped? | Needed by | How to get it |
|---|---|---|---|
| `topo_data/base/node{1..6}/` captures | yes (57 MB) | (2) | Shipped. Recapture on the robot with step (1) if the house changes. |
| `topo_data/base_topology.pkl` (images + annotations) | **no** | (3), (4), (6), `render_topology.py` | Rebuild with the three commands above |
| `topo_data/score/` intermediate signals and `scores.json` | yes | (5), robot | Shipped; regenerate with (4) and (5) |
| `topo_data/topo_graph/*_topology.pkl` (no images) | yes | robot, (5) | Shipped; regenerate base with (6) |
| `topo_data/score/renders/` heatmaps | no (git-ignored) | viewing only | Produced by (5) |

- Steps (2)–(4) and (6) need the image-carrying topology, so rebuild it first.
  Step (5) and its metrics read the shipped runtime topology when `topo_data/base_topology.pkl` is absent,
  so **re-fusing scores needs no rebuild**.
- The capture PNGs were losslessly recompressed to save space. Every image decodes to the same pixels as the
  original capture, so annotation and scoring give the same results.

Results of re-running each step from the shipped files:

| Step | Result |
|---|---|
| (2) + (3) rebuild | Node positions, room types, all 288 images (pixel-identical), 72 captions and the furniture detections match the topology the shipped scores were made from. |
| (4) scoring | VLM (1800), caption LLM (1800) and room affinity (150) scores all match the shipped files (greedy decoding). |
| (5) fusion | `scores.json` is byte-identical. |
| (6) runtime topology | Byte-identical to the shipped `topo_data/topo_graph/base_topology.pkl`. |

### 6.3 Environment
Add the Qwen packages to the robot environment `objnav_jackal` (Section 2). Run every command from the **repository root**.
```bash
conda activate objnav_jackal
pip install -r requirements-qwen.txt
export PYTORCH_ALLOC_CONF=expandable_segments:True     # avoids fragmentation when loading the 4-bit VLM
```
- The Qwen models are downloaded from Hugging Face on first use (no access approval needed):
  - `Qwen/Qwen3-VL-8B-Instruct`: about 17 GB
  - `Qwen/Qwen3-4B-Instruct-2507`: about 8 GB
- GPU memory: about 7 GB for the VLM (4-bit) and about 8 GB for the 4B LLM (bf16). They are never loaded together.
- All decoding is greedy (`do_sample=False`) with Qwen3 thinking disabled.

### 6.4 (1) Capture nodes on the robot — `robot/run_scan_capture.py`
Place the robot at each node position and run the script. The robot turns right 12 × 30° and saves main/sub RGB and
depth (uint16, mm) in each direction, while estimating its pose with the same particle filter as `run_robot`.
SAM 3 and driving are not used.
```bash
python -m robot.run_scan_capture --out topo_data/base/node1 [--render] [--jackal-host <host>]
#   ... repeat for node2 to node6
```
- **Start heading:** when the spin starts, the robot must face **map yaw 0** (up in the map image, world `-z`).
  The topology treats `dir_00` as yaw 0 and `dir_k` as `k × 30°` turned to the right.
- **Output:** 48 images `dir_<ii>_deg<ddd>_{main,sub}_{rgb,depth}.png` and `scan.json` (pose estimate per direction)
- **Node layout** is fixed in `build_topologies.py`: nodes 1–2 are `living_room`, 3–4 `kitchen`, 5–6 `bedroom`.
  Edit `ROOM_TYPE` to change the number of nodes or their room types.

### 6.5 (2) Build the topology — `build_topologies.py`
```bash
python build_topologies.py                    # topo_data/base/ → topo_data/base_topology.pkl
python render_topology.py                     # check figure: outputs/topology_base.png
```
- Node positions come from the last pose estimate `(x, z)` in `scan.json`.
- Nodes whose localization was wrong use hand-set coordinates in `FIXED_POS`. Nodes 5 and 6 are currently set to
  `(1.8, 1.8)` and `(0.2, 0.8)`; for example, node 5's scan estimate was `(-1.29, 6.42)`, outside the bedroom.
  Check the node positions in the `render_topology.py` figure and fix `FIXED_POS` if needed.
- The built file has no annotations yet; step (3) is required.

### 6.6 (3) Annotation
Both scripts **overwrite `topo_data/base_topology.pkl` in place**.
```bash
# SAM 3 furniture detection per view (sofa, kitchen countertop, bed, dressing table, ...)
python qwen_priors/annotate_topo_meta.py --priors base --no-objects
# one-sentence caption per view with Qwen3-VL-8B (4-bit), from the main+sub images only
python qwen_priors/annotate_captions.py --priors base
```
- `--no-objects`: the base scene has no target objects, so object detection is skipped and only furniture is detected.
- Furniture goes to `view.meta["furniture"]` per view and `node.meta["furniture"]["all"]` per node; captions to `view.meta["caption"]`.

### 6.7 (4) Scoring — `qwen_priors/gen_direction.py`
Scores 72 views (6 nodes × 12 directions) × 25 objects. Object names use the phrases in `utils/sam3_prompt.py`.
```bash
# (a) VLM image score: Qwen3-VL-8B-Instruct 4-bit; input = main+sub images + detected furniture + room type
python qwen_priors/gen_direction.py --signal vlm
#     → topo_data/score/Qwen3-VL-8B_dir/scores.json

# (b) caption LLM score + room affinity: Qwen3-4B-Instruct-2507 bf16, text only
python qwen_priors/gen_direction.py --signal caption
#     → topo_data/score/caption_llm_dir/scores.json
#     → topo_data/score/room_affinity.json   (key "node_id|object", judged from room type + all node furniture)
```
- Run time (RTX 4080 SUPER): `--signal vlm` about 12 min (about 10 s per view), `--signal caption` about 4 min (about 3 s per view)
- `--render` also draws per-object heatmaps of that signal alone.
- `--model` and `--load {4bit,bf16}` select another Qwen model or precision.
- Prompts: system prompts are in `qwen_priors/prompts.py` (`SHARP_VLM`, `SHARP_LLM`, `ROOM`, `CAPTION`), user templates in
  `qwen_priors/scorers.py`. The `SCORE: <0-1>` in the model output is parsed; a malformed answer counts as 0.
- The models only emit a few values (0, 0.3, 0.5, 0.8, 1.0, ...), so directions within a node often tie.

### 6.8 (5) Fusion — `qwen_priors/ensemble_priors.py`
```bash
python qwen_priors/ensemble_priors.py
#   → topo_data/score/scores.json  (+ 25 heatmaps topo_data/score/renders/<object>.png)
```
Score formula (`qwen_priors/common.py::combine_prior_scores`; the robot's live re-scoring uses the same function):
```
direction = max(VLM_image, 0.5 × caption_LLM)    # image score is the main signal, caption a half-strength backup
room_gate = 0.6 + 0.4 × room_affinity            # room affinity is a soft weight in 0.6–1.0
score     = direction × room_gate
```
- Room affinity never multiplies a view down to 0; even for an object in an unusual room, 60% of the view evidence remains.
- Scores are not normalized per object.
- The output is object-first: `{object: {node_id: {direction_k: score}}}`, where direction `k` is yaw `k × 30°`.
- A short metrics table is printed at the end (`qwen_priors/evals.py`). For example, `empty` is the mean score of views
  where no furniture was detected (wall, curtain, floor); lower is better.
- In the heatmaps, each node's 12 wedges show the per-direction scores (blue 0 → red 1, `dir_00` at the top).

### 6.9 (6) Runtime topology — `strip_topology_images.py`
The robot reads a lightweight topology without images (about 27 KB). Rebuild it if you redid (2) or (3).
```bash
python strip_topology_images.py        # topo_data/base_topology.pkl → topo_data/topo_graph/base_topology.pkl
```
Not needed if you only changed scores; the robot reads `scores.json` directly.

> The scenario topologies for `--prior 1|2|3` (`topo_data/topo_graph/prior{1,2,3}_topology.pkl`) are shipped only in
> the lightweight form. They were captured the same way with objects placed in each scenario, and SAM 3 object
> detections were added; the robot picks the prior view from those detections. Their raw captures are not included.

---

## 7. Path settings

There are no absolute client-side paths in this repository. Every path is relative to the repository root or can be
changed with an environment variable.

| Item | Default | How to change |
|---|---|---|
| SAM 3 source | `third_party/sam3` | Environment variable `SAM3_ROOT=<path>` (relative paths are taken from the repository root) |
| Python interpreter | the active environment | Set `SAM3_PYTHON=<python path>` to re-exec with that interpreter (when SAM 3 lives in a separate conda environment) |
| Map, topology, scores | `occupancy_map/…`, `topo_data/…` | `configs/jackal.yaml` |
| Run records | `outputs/robot_runs/…` | `--record <DIR>` |
| habitat-lab source (Section 5) | `third_party/habitat-lab` | Install location only; not referenced by the code |
| Map-building scene and dataset (Section 5) | `habitat_data/…` | `configs/daejeon_base.yaml` |
| Map collection output (Section 5) | `output_maps/occupancy_map/val/` | `--version`, `--save_dir` |
| Node captures (Section 6) | `topo_data/base/node{k}/` | `run_scan_capture.py --out` (default `outputs/scan-<time>/`) |
| Topology and score artifacts (Section 6) | `topo_data/base_topology.pkl`, `topo_data/score/` | `SCORE_DIR`, `FINAL_SCORES` in `qwen_priors/common.py` |
| ROS and workspace on the robot | `/opt/ros/melodic/…`, `/home/jackal/catkin_ws/…` | `JackalConnectionConfig` in `robot/jackal.py` (paths on the robot) |

Example: SAM 3 installed elsewhere, in a different conda environment
```bash
SAM3_ROOT=/opt/sam3 SAM3_PYTHON=~/miniconda3/envs/sam3/bin/python \
    python -m robot.run_robot --goal phone --prior 1
```

---

## 8. (Optional) Live view re-scoring: `topo_update: view`
Setting `search.topo_update: view` in `configs/jackal.yaml` makes Qwen3-VL-8B (4-bit) re-score each view the robot
arrives at from the live camera image. It is meant for scenes with moving objects; the default is `off`.
It needs extra packages and about 10 GB of GPU memory (including SAM 3).
```bash
pip install -r requirements-qwen.txt     # same packages as Section 6
```
If the model cannot be loaded, a warning is printed and the stored prior is used.

---

## 9. Troubleshooting

| Symptom | What to check |
|---|---|
| `Jackal SSH setup failed` / host key error | Section 2.6: connect once with `ssh jackalw` to register the host key; check the alias or IP |
| SSH authentication failure | The password in `robot/jackal.py` is empty by default. Set up key-based auth (Section 2.6) or fill in `ssh_password`. |
| `failed to launch dual D455 cameras` | The robot's `jackal_dual_d455` package, catkin workspace path, camera USB connections |
| Run stops with an odometry error | Check that the robot publishes `/odometry/filtered`; the run does not proceed without odometry |
| `SAM3 BPE vocabulary not found` | Check the clone in `third_party/sam3` (Section 2.3) or `SAM3_ROOT` |
| SAM 3 weights 401/403 | Check access approval for `facebook/sam3` and `hf auth login` (Section 2.5) |
| numpy 2.x errors | Re-pin with `pip install "numpy==1.26.4" "opencv-python<4.10"` |
| No `--render` window | Check the display (`$DISPLAY`); if `opencv-python-headless` is installed, remove it and install `opencv-python` |
| Estimated pose moves the wrong way | Driving forward should move the estimate forward, turning left should turn it left. If reversed, flip the sign of `d_lat` / `dyaw` in `odom_to_robot_delta` in `robot/run_robot.py`. |
| (Section 5) `habitat_sim` import or EGL/GPU error | Make sure `objnav_jackal_sim` is active and the NVIDIA driver supports EGL. Do not install habitat-sim in the robot environment (Python 3.12). |
| (Section 5) dataset or scene not found | Run from the repository root (`habitat_data/` paths are relative to it) |
| (Section 6) `topo_data/base_topology.pkl` not found | It is not shipped; rebuild it with the three commands in Section 6.2 |
| Robot sways left/right in front of the object or the goal jumps | Check that `cameras.*.forward_offset_m` and `right_offset_m` hold measured values |

---

## 10. Repository layout

```
configs/jackal.yaml          robot config (cameras, map, localization, search, detection, driving)
robot/
  run_robot.py               entry point
  agent.py                   RobotAgent (inherits the shared decision logic)
  config.py                  jackal.yaml loader
  robot_env.py               sensors → agent input format, actions → motion commands
  jackal.py                  SSH connection and low-level motion settings (ssh_password: fill in before use)
  jackal_bridge.py           ROS bridge that runs on the robot (sent over SSH at run time)
  sam3_detector.py           SAM 3 detector (runs in-process)
  render.py, recording.py    live visualization and run recording
agents/last_mile_fmm_agent.py  search, detection and last-mile decision logic
navigation/                  particle-filter localization, FMM planning, early_nav, collision shield
planning/topology.py         topology graph
qwen_priors/                 prior score fusion (at run time) and the scoring pipeline (Section 6)
utils/                       visualization panels, SAM 3 prompt mapping
occupancy_map/base_occ_map.dat.gz   map for localization and planning
topo_data/topo_graph/        base / prior1–3 runtime topologies
topo_data/score/scores.json  per-object view prior scores
requirements.txt             robot environment dependencies
LICENSE                      MIT license

# occupancy map building (Section 5, Habitat environment)
collect_occupancy_map.py     map collection by teleport + 360° scans
runners/collect_occupancy_map_runner.py
maps/occupancy_map.py        accumulates depth into obstacle / seen grids
utils/config_utils.py, utils/env_utils.py   Habitat config loader and environment creation
configs/daejeon_base.yaml    Habitat config for map building
habitat_data/                object-free Daejeon base scene assets + 1-episode dataset
apply_occ_map_edits.py       applies the manual edits (bed obstacle, entrance keep-out)
occupancy_map/base_map_edits.json   edited cells (world coordinates)
requirements-sim.txt         extra dependencies for the Habitat environment

# topology and search score building (Section 6, robot environment + requirements-qwen.txt)
robot/run_scan_capture.py    in-place 360° capture at a node + pose estimate
topo_data/base/node{1..6}/   raw captures (12 directions × main/sub × RGB/depth + scan.json)
build_topologies.py          captures → image-carrying topology (topo_data/base_topology.pkl, rebuilt, not shipped)
render_topology.py           draws topology nodes on the map
qwen_priors/annotate_topo_meta.py   SAM 3 furniture annotation
qwen_priors/annotate_captions.py    Qwen3-VL view caption annotation
qwen_priors/gen_direction.py        Qwen VLM/LLM scoring (view × object)
qwen_priors/ensemble_priors.py      fuses the three signals → scores.json (+ heatmaps)
qwen_priors/common.py, prompts.py, scorers.py, qwen_score.py, viz.py, evals.py   shared modules
topo_data/score/Qwen3-VL-8B_dir/, caption_llm_dir/, room_affinity.json   intermediate signals
strip_topology_images.py     builds the image-free runtime topology
requirements-qwen.txt        Qwen scoring dependencies
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The following third-party components are **not included** in this repository. They are installed or downloaded
during setup and remain under their own licenses:

| Component | Used for | License |
|---|---|---|
| [SAM 3](https://github.com/facebookresearch/sam3) (code and `facebook/sam3` weights) | object and furniture detection | [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE) |
| [habitat-sim](https://github.com/facebookresearch/habitat-sim), [habitat-lab](https://github.com/facebookresearch/habitat-lab) | occupancy map building (Section 5) | MIT |
| [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | view captions and prior scoring (Section 6) | Apache 2.0 |

---

## Citation

If you use this code, please cite this repository:

```bibtex
@misc{objnav_jackal,
  author       = {Kim, Minsoo},
  title        = {{ObjNav\_Jackal}: Object-Goal Navigation on a Real Clearpath Jackal},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/rllab-snu/ObjNav_Jackal}}
}
```

---

## Acknowledgement

This work was supported by the Institute of Information & communications Technology Planning & Evaluation(IITP) grant funded by the Korea government(MSIT) (No. RS-2024-00336738, Development of Complex Task Planning Technologies for Autonomous Agents)
