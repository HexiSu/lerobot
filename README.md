# LeRobot SO-ARM101 Pi0 Hierarchical Inference

This repository is a project fork built on top of [Hugging Face LeRobot](https://github.com/huggingface/lerobot). It keeps the original LeRobot training, dataset, robot-control, and policy infrastructure, and adds a practical long-horizon execution layer for a SO-ARM101 robot arm using a fine-tuned pi0 policy.

The main goal is to move beyond single-command VLA inference such as `Grab Orange` and support staged tasks such as:

```text
Grab Orange, when done, then Push Cup
```

The implementation follows a hierarchical control design:

```text
High-level task planning -> FSM task manager -> pi0/VLA low-level action policy
```

## Demo Videos

### Single-task pi0 inference: Grab Oranges

The pi0 policy was fully fine-tuned on an A6000 and deployed through asynchronous inference on an RTX 4090. The video below is shown at 3x speed.

<video src="assets/grab-oranges-3x.mp4" controls width="720"></video>

Fallback link: [assets/grab-oranges-3x.mp4](assets/grab-oranges-3x.mp4)

### Multi-task pi0 inference: Grab Orange, then Push Cup

The robot first executes `Grab Orange`. After the success detector observes that the orange has entered the configured cup ROI, the FSM switches to the next subtask and clears stale action chunks before executing `Push Cup`.

<video src="assets/Muti-task.mp4" controls width="720"></video>

Fallback link: [assets/Muti-task.mp4](assets/Muti-task.mp4)

The original single-task video is also kept at [assets/Grab Oranges.mp4](assets/Grab%20Oranges.mp4).

## What This Project Adds

Compared with upstream LeRobot, this project adds:

- **Hierarchical task execution for pi0**: long natural-language tasks can be split into ordered subtasks with a finite-state machine.
- **Rule-based and JSON task planners**: deterministic first-version planners for repeatable robotics experiments.
- **LLM planner extension point**: `task_planner=llm` is reserved as an explicit integration point. The pi0 checkpoint's PaliGemma/Gemma components are used for VLA action conditioning, not as a general chat-style planner.
- **Visual success detection**: an `orange_in_cup` detector based on HSV segmentation and a configurable cup ROI.
- **Async cloud inference workflow**: the robot client can run locally while the pi0 policy server runs on a remote GPU machine through an SSH tunnel.
- **Stale action protection during subtask switches**: action chunks are tagged with their source task, and the client drops delayed chunks from previous subtasks after an FSM transition.
- **Policy-server reuse**: the async server can reuse an already loaded policy when the same checkpoint is requested again, avoiding unnecessary reloads during repeated experiments.
- **Inference memory fix**: async policy inference runs under `torch.inference_mode()` to avoid retaining autograd graphs and exhausting GPU memory.

## Project Architecture

Key additions are concentrated in these areas:

- [src/lerobot/common/hierarchical_task.py](src/lerobot/common/hierarchical_task.py): task plans, rule-based/json planners, FSM task manager, pi0 language-info loader, and success detectors.
- [src/lerobot/async_inference/robot_client.py](src/lerobot/async_inference/robot_client.py): dynamic current-task selection, FSM transition handling, action-queue clearing, and stale action-chunk filtering.
- [src/lerobot/async_inference/policy_server.py](src/lerobot/async_inference/policy_server.py): task-change filtering protection, no-grad inference, policy reuse, and action task tagging.
- [src/lerobot/async_inference/configs.py](src/lerobot/async_inference/configs.py): CLI configuration for task planners and success detectors.
- [src/lerobot/scripts/lerobot_record.py](src/lerobot/scripts/lerobot_record.py): local `lerobot-record` support for the same hierarchical task workflow.

At runtime, the async flow is:

1. The local robot client captures SO-ARM101 observations and camera frames.
2. The task planner creates subtasks from the original task string.
3. The FSM exposes the current subtask text to pi0 as the `task` field.
4. The remote policy server predicts action chunks for the current subtask.
5. The client executes actions locally.
6. The success detector updates the FSM; on transition, the client clears queued actions and forces the next observation to be processed.

## Hardware and Training Setup

This project was tested with:

- Robot: SO-ARM101 follower arm.
- Cameras: one wrist/front camera and one fixed side camera.
- Policy: pi0, fully fine-tuned from `lerobot/pi0_base`.
- Training GPU: NVIDIA A6000.
- Async inference GPU: NVIDIA RTX 4090.
- Task examples: `Grab Orange`, `Grab Orange, when done, then Push Cup`.

## Installation

Start from a normal LeRobot development install. For pi0 and async inference, install the relevant extras:

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot
git clone https://github.com/HexiSu/lerobot.git
cd lerobot
pip install -e ".[pi,async]"
```

If you use `uv`, the upstream LeRobot workflow also works:

```bash
uv sync --locked --extra pi --extra async --extra test --extra dev
```

## Data Collection

Example SO-ARM101 data collection command:

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=suhexi_follower_arm \
    --robot.cameras='{ front: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 60, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM3 \
    --teleop.id=suhexi_leader_arm \
    --display_data=true \
    --dataset.repo_id=SuHexi/lerobot_suhexi_dataset_b \
    --dataset.num_episodes=7 \
    --dataset.single_task="Point At Snack" \
    --dataset.push_to_hub=false
```

Useful setup commands:

```bash
lerobot-find-cameras opencv
lerobot-find-port
sudo chmod 666 /dev/ttyACM*
hf auth login
wandb login
```

## Pi0 Fine-tuning

Example pi0 training command:

```bash
lerobot-train \
  --dataset.repo_id=SuHexi/lerobot_suhexi_dataset_b \
  --dataset.root=../datasets/lerobot_suhexi_dataset_b \
  --dataset.revision=v0.1.0 \
  --dataset.streaming=false \
  --policy.type=pi0 \
  --output_dir=./outputs/train/pi0 \
  --job_name=pi0 \
  --policy.pretrained_path=lerobot/pi0_base \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --steps=50000 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --wandb.project=Lerobot_suhexi \
  --batch_size=8 \
  --dataset.video_backend=pyav
```

For inference checkpoints, disable training-only options in the exported `config.json` if startup is slow or memory-constrained:

```json
{
  "compile_model": false,
  "gradient_checkpointing": false
}
```

## Async Inference

Run the policy server on the GPU machine:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
python -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080
```

Forward the server port to the local robot machine if the GPU server is remote:

```bash
ssh -N -L 8080:127.0.0.1:8080 <user>@<remote-host>
```

Run a single-task client locally:

```bash
python -m lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=suhexi_follower_arm \
    --robot.cameras='{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 60, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --task="Grab Orange" \
    --policy_type=pi0 \
    --pretrained_name_or_path=/root/autodl-tmp/outputs/train/pi0/checkpoints/last/pretrained_model \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average
```

Run a multi-stage client:

```bash
python -m lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=suhexi_follower_arm \
    --robot.cameras='{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 60, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --task="Grab Orange, when done, then Push Cup" \
    --task_planner=rule_based \
    --success_detector=orange_in_cup \
    --success_camera=side \
    --cup_roi="15,220,180,365" \
    --orange_hsv_lower="8,150,120" \
    --orange_hsv_upper="18,255,255" \
    --success_min_area=500 \
    --success_hold_s=1.0 \
    --policy_type=pi0 \
    --pretrained_name_or_path=/root/autodl-tmp/outputs/train/pi0/checkpoints/last/pretrained_model \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average
```

Notes:

- `task_planner=rule_based` splits connectors such as `when done, then`, `then`, and `然后`.
- `orange_in_cup` is a task-specific detector, not a general success detector.
- `cup_roi` must be calibrated for the side camera view.
- For headless servers, keep `success_debug_view=false`; GUI OpenCV windows may not be available.
- The model path is resolved on the policy-server machine, not on the local robot-client machine.

## Local Hierarchical Inference

The same task-planning options are also available through local `lerobot-record` inference:

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras='{ front: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 60, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 7, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
  --robot.id=suhexi_follower_arm \
  --display_data=false \
  --dataset.repo_id=SuHexi/eval_lerobot_suhexi_pi0_multistage \
  --dataset.single_task="Grab Orange, when done, then Push Cup" \
  --dataset.task_planner=rule_based \
  --dataset.success_detector=orange_in_cup \
  --dataset.success_camera=side \
  --dataset.cup_roi="15,220,180,365" \
  --policy.path=SuHexi/lerobot_suhexi_pi0 \
  --dataset.push_to_hub=false
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'lerobot'`

Install the repository in the active environment:

```bash
pip install -e ".[pi,async]"
```

Or run with:

```bash
PYTHONPATH=src python -m lerobot.async_inference.robot_client ...
```

### `CUDA out of memory` on the policy server

Stop stale server processes and restart with inference-only settings:

```bash
pkill -f lerobot.async_inference.policy_server
nvidia-smi
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080
```

Make sure the server code uses `torch.inference_mode()` for async policy prediction.

### Tokenizer download errors

If the server cannot reach Hugging Face, point `policy_preprocessor.json` at a local tokenizer snapshot:

```json
"tokenizer_name": "/path/to/models--google--paligemma-3b-pt-224/snapshots/<revision>"
```

### Robot serial port errors

```bash
lerobot-find-port
sudo chmod 666 /dev/ttyACM*
lsof /dev/ttyACM0
```

Kill old local clients if they still hold the port.

## Relationship to Upstream LeRobot

This repository remains based on Hugging Face LeRobot and keeps the upstream Apache-2.0 license and citation requirements. The additions here are project-specific extensions for SO-ARM101 pi0 fine-tuning, cloud async inference, and multi-stage task execution.

If you use the underlying LeRobot framework, please cite the upstream project:

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```
