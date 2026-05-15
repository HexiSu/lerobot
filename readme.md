# 摄像头
```bash
lerobot-find-cameras opencv
cheese -d /dev/video1
```

# 机械臂
```bash
lerobot-find-port
```

# 权限
```bash
sudo chmod 666 /dev/ttyACM*
```

# 登陆
```bash
hf auth login
wandb login
```

# 采集数据
```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=suhexi_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=suhexi_leader_arm \
    --display_data=true \
    --dataset.repo_id=SuHexi/lerobot_suhexi_dataset_a \
    --dataset.num_episodes=10 \
    --dataset.single_task="Grab Oranges" \
    --dataset.push_to_hub=false

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=suhexi_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 60, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM3 \
    --teleop.id=suhexi_leader_arm \
    --display_data=true \
    --dataset.repo_id=SuHexi/lerobot_suhexi_dataset_b \
    --dataset.num_episodes=7 \
    --dataset.single_task="Point At Snack" \
    --dataset.push_to_hub=false \
    --resume=true \
    --dataset.root=/home/suhexi/.cache/huggingface/lerobot/SuHexi/lerobot_suhexi_dataset_b
```

# 上传数据集
```bash
# 脚本
python upload_dataset.py

# 命令行
hf upload SuHexi/lerobot_suhexi_dataset_a /datasets/lerobot_suhexi_dataset_a/ --repo-type=dataset
```

# 回放
```bash
lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=suhexi_follower_arm \
    --dataset.repo_id=SuHexi/lerobot_suhexi_dataset_a \
    --dataset.episode=2
```

# 训练-ACT
```bash
lerobot-train \
  --dataset.repo_id=SuHexi/lerobot_suhexi_dataset_a \
  --dataset.root=../datasets/lerobot_suhexi_dataset_a \
  --dataset.streaming=false \
  --policy.type=act \
  --output_dir=./outputs/train/act/ \
  --job_name=act_orange \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=Lerobot_suhexi \
  --policy.push_to_hub=false \
  --steps=20000 \
  --batch_size=8 \
  --dataset.video_backend=pyav

# 续训
  lerobot-train \
  --config_path=./outputs/train/act/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

# 训练-smolVLA
```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --dataset.repo_id=SuHexi/lerobot_suhexi_dataset_a \
  --dataset.root=../datasets/lerobot_suhexi_dataset_a \
  --dataset.revision=v0.1.0 \
  --dataset.streaming=false \
  --output_dir=./outputs/train/smolvla \
  --job_name=smolvla_orange \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=Lerobot_suhexi \
  --policy.push_to_hub=false \
  --steps=40000 \
  --batch_size=8 \
  --dataset.video_backend=pyav

# 续训
  lerobot-train \
  --config_path=./outputs/train/smolvla/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

# 训练-pi0
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

# 续训
  lerobot-train \
  --config_path=./outputs/train/pi0/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

# 上传模型
```bash
# 脚本
python upload_model.py

# 命令行
huggingface-cli upload SuHexi/act \
  outputs/train/act/checkpoints/last/pretrained_model

# 上传某checkpoint
CKPT=010000
huggingface-cli upload SuHexi/act${CKPT} \
  outputs/train/act/checkpoints/${CKPT}/pretrained_model
```

# 同步推理
## 本地
```bash
lerobot-record  \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
  --robot.id=suhexi_follower_arm \
  --display_data=true \
  --dataset.repo_id=SuHexi/eval_lerobot_suhexi_dataset_a \
  --dataset.single_task="Grab Oranges" \
  --dataset.episode_time_s=1000 \
  --policy.path=./outputs/train/act/checkpoints/last/pretrained_model \
  --dataset.push_to_hub=false

lerobot-record  \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 60, fourcc: "MJPG"}}" \
  --robot.id=suhexi_follower_arm \
  --display_data=true \
  --dataset.repo_id=SuHexi/eval_lerobot_suhexi_dataset_a \
  --dataset.single_task="Grab Oranges" \
  --policy.path=./outputs/train/act/checkpoints/020000/pretrained_model \
  --dataset.push_to_hub=false

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras='{ camera1: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 60, fourcc: "MJPG"}, camera2: {type: opencv, index_or_path: 7, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
  --robot.id=suhexi_follower_arm \
  --display_data=false \
  --dataset.repo_id=SuHexi/eval_lerobot_suhexi_smolvla_twocameras \
  --dataset.single_task="Grab Oranges" \
  --policy.path=SuHexi/lerobot_suhexi_smolvla_twocameras \
  --dataset.push_to_hub=false

  --dataset.rename_map='{"observation.images.front": "observation.images.camera1","observation.images.side": "observation.images.camera2"}'

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras='{ camera1: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 60, fourcc: "MJPG"}, camera2: {type: opencv, index_or_path: 7, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
  --robot.id=suhexi_follower_arm \
  --display_data=false \
  --dataset.repo_id=SuHexi/eval_lerobot_suhexi_pi0 \
  --dataset.single_task="Grab Oranges" \
  --policy.path=SuHexi/lerobot_suhexi_pi0 \
  --dataset.push_to_hub=false

# 本地 pi0 多阶段推理：先抓橘子，检测到橘子进入杯子 ROI 后切换到下一子任务
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras='{ front: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 60, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 7, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
  --robot.id=suhexi_follower_arm \
  --display_data=false \
  --dataset.repo_id=SuHexi/eval_lerobot_suhexi_pi0_multistage \
  --dataset.single_task="Grab Orange, when done, then make a face" \
  --dataset.task_planner=rule_based \
  --dataset.success_detector=orange_in_cup \
  --dataset.success_camera=front \
  --dataset.cup_roi="260,180,420,360" \
  --dataset.success_hold_s=0.5 \
  --policy.path=SuHexi/lerobot_suhexi_pi0 \
  --dataset.push_to_hub=false

rm -rf /home/suhexi/.cache/huggingface/lerobot/SuHexi/eval_lerobot_suhexi_pi0

# ['auto', 'h264', 'h264_nvenc', 'h264_qsv', 'h264_vaapi', 'h264_videotoolbox', 'hevc', 'hevc_nvenc', 'hevc_videotoolbox', 'libsvtav1']
```

## hf
```bash
lerobot-record  \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
  --robot.id=suhexi_follower_arm \
  --display_data=true \
  --dataset.repo_id=SuHexi/eval_lerobot_suhexi_dataset_a \
  --dataset.single_task="Grab Oranges" \
  --policy.path=SuHexi/lerobot_suhexi_model_a
```

# 异步推理
```bash
# 启动服务器本地端口
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080

# SSH 绑定端口
ssh -N -L 8080:127.0.0.1:8080 -p 11733 root@connect.bjb2.seetacloud.com
ssh -N -L 8080:127.0.0.1:8080 -p 60019 featurize@workspace.featurize.cn

# 启动客户端-云端本地
python -m lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=suhexi_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 60, fourcc: "MJPG"}}" \
    --task="Grab Orange" \
    --policy_type=act \
    --pretrained_name_or_path=/root/autodl-tmp/lerobot/lerobot/outputs/train/act/checkpoints/last/pretrained_model \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average

python -m lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=suhexi_follower_arm \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 60, fourcc: "MJPG"}, camera2: {type: opencv, index_or_path: 7, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
    --task="Grab Orange" \
    --policy_type=smolvla \
    --pretrained_name_or_path=lerobot/outputs/train/smolvla_twocameras/checkpoints/last/pretrained_model \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.6 \
    --aggregate_fn_name=weighted_average

python -m lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=suhexi_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 60, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 7, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
    --task="Grab Orange" \
    --policy_type=pi0 \
    --pretrained_name_or_path=/home/featurize/lerobot/outputs/train/pi0/checkpoints/last/pretrained_model \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average

# pi0 多阶段任务：先抓橘子，检测到橘子进入杯子 ROI 后，自动切换到下一阶段
python -m lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=suhexi_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 60, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 7, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
    --task="Grab Orange, when done, then make a face" \
    --task_planner=rule_based \
    --success_detector=orange_in_cup \
    --success_camera=front \
    --cup_roi="260,180,420,360" \
    --success_hold_s=0.5 \
    --policy_type=pi0 \
    --pretrained_name_or_path=/home/featurize/lerobot/outputs/train/pi0/checkpoints/last/pretrained_model \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average

说明：`--task_planner=rule_based` 会把 `when done, then` / `then` / `然后` 等顺序连接词拆成多个子任务；每次 FSM 切换子任务时会清空上一阶段动作队列，并把新的子任务文本发送给 pi0。当前 pi0 checkpoint 的 PaliGemma/Gemma 组件用于 VLA 动作条件编码，不是通用聊天式任务规划 LLM；如需接入外部 LLM，可在 `task_planner=llm` 的接口上扩展。`orange_in_cup` 是基于 HSV + 杯子 ROI 的专用成功检测器，运行前需要根据摄像头画面调整 `--cup_roi`，必要时使用 `--success_debug_view=true` 调试。

# 启动客户端-hf
python -m lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8081 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=suhexi_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 60, fourcc: "MJPG"}}" \
    --task="Grab Orange" \
    --policy_type=act \
    --pretrained_name_or_path=SuHexi/lerobot_suhexi_model_a \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average
```