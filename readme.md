# 摄像头
```bash
lerobot-find-cameras opencv
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
    --robot.port=/dev/ttyACM2 \
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
  --dataset.repo_id=SuHexi/lerobot_suhexi_dataset_a \
  --dataset.root=../datasets/lerobot_suhexi_dataset_a \
  --dataset.revision=v0.1.0 \
  --dataset.streaming=false \
  --policy.type=smolvla \
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
  --robot.port=/dev/ttyACM2 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 60, fourcc: "MJPG"}}" \
  --robot.id=suhexi_follower_arm \
  --display_data=true \
  --dataset.repo_id=SuHexi/eval_lerobot_suhexi_dataset_a \
  --dataset.single_task="Grab Oranges" \
  --policy.path=./outputs/train/act/checkpoints/020000/pretrained_model \
  --dataset.push_to_hub=false

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.cameras='{"front": {"type": "opencv", "index_or_path": 3, "width": 640, "height": 480, "fps": 60, "fourcc": "MJPG"}}' \
  --robot.id=suhexi_follower_arm \
  --display_data=true \
  --dataset.repo_id=SuHexi/eval_lerobot_suhexi_dataset_a \
  --dataset.single_task="Grab Oranges" \
  --policy.path=TommyZihao/lerobot_zihao_model_a \
  --dataset.push_to_hub=false

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

# 启动客户端-云端本地
python -m lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM2 \
    --robot.id=suhexi_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 60, fourcc: "MJPG"}}" \
    --task="Grab Orange" \
    --policy_type=act \
    --pretrained_name_or_path=/root/autodl-tmp/lerobot/lerobot/outputs/train/act/checkpoints/last/pretrained_model \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average

# 启动客户端-hf
python -m lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8081 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM2 \
    --robot.id=suhexi_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 60, fourcc: "MJPG"}}" \
    --task="Grab Orange" \
    --policy_type=act \
    --pretrained_name_or_path=SuHexi/lerobot_suhexi_model_a \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average
```