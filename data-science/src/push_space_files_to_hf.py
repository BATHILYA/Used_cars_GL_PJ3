import os
import argparse
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment_dir", type=str, required=True, help="Folder with Dockerfile/requirements/src")
    parser.add_argument("--space_repo_id", type=str, required=True, help="HF Space repo id: user/space")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN is not set.")

    api = HfApi(token=token)

    # Upload ONLY the Space repo content (Dockerfile/requirements/src/...)
    api.upload_folder(
        folder_path=args.deployment_dir,
        repo_id=args.space_repo_id,
        repo_type="space",
    )

    print("✅ Space files pushed to:", args.space_repo_id)

if __name__ == "__main__":
    main()
