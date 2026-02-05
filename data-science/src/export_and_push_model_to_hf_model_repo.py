import os
import argparse
import mlflow
import mlflow.sklearn
import joblib
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="MLflow model folder")
    parser.add_argument("--model_repo_id", type=str, required=True, help="HF model repo id: user/repo")
    parser.add_argument("--model_filename", type=str, default="best_price_model_v2.joblib")
    parser.add_argument("--revision_note", type=str, default="Publish new model version")
    parser.add_argument("--done_out", type=str, required=True)

    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN is not set.")

    api = HfApi(token=token)

    # Ensure model repo exists
    try:
        api.repo_info(repo_id=args.model_repo_id, repo_type="model")
    except RepositoryNotFoundError:
        create_repo(repo_id=args.model_repo_id, repo_type="model", private=False)

    # Load MLflow model and export to joblib
    model = mlflow.sklearn.load_model(f"file://{args.model_dir}")
    local_path = args.model_filename
    joblib.dump(model, local_path)

    # Upload model artifact
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=args.model_filename,
        repo_id=args.model_repo_id,
        repo_type="model",
        commit_message=args.revision_note,
    )

    # Write and upload a simple model card
    readme_path = "README.md"         
    card = f"""---
        tags:
        - regression
        - scikit-learn
        - mlflow
        ---

        # Used Cars Price Prediction Model

        This repository contains the exported model artifact: `{args.model_filename}`.
        """
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(card)
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=args.model_repo_id,
        repo_type="model",
        commit_message="Add/update model card",
    )

    os.makedirs(os.path.dirname(args.done_out), exist_ok=True)
    with open(args.done_out, "w", encoding="utf-8") as f:
        f.write("done")

    print("✅ Published model to HF model repo:", args.model_repo_id)

if __name__ == "__main__":
    main()
