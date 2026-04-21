from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="/home/suhexi/.cache/huggingface/lerobot/SuHexi/lerobot_suhexi_dataset_a",
    repo_id="SuHexi/lerobot_suhexi_dataset_a",
    repo_type="dataset"
)

api.create_tag("SuHexi/lerobot_suhexi_dataset_a", tag="v0.1.0", repo_type="dataset")