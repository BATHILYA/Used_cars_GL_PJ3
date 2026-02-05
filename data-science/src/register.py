# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import os
import argparse
import logging
import mlflow
import pandas as pd
from pathlib import Path         
import json
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential


def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')  # Hint: Specify the type for model_name (str)
    parser.add_argument('--model_path', type=str, help='Model directory')  # Hint: Specify the type for model_path (str)
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")  # Hint: Specify the type for model_info_output_path (str)
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')                                                                                   
    return args
def get_credential():
    # Prefer AzureML job-managed auth when available (falls back cleanly)
    try:
        from azure.ai.ml.identity import AzureMLOnBehalfOfCredential  # type: ignore
        return AzureMLOnBehalfOfCredential()
    except Exception:
        # Avoid interactive creds inside jobs
        return DefaultAzureCredential(exclude_interactive_browser_credential=True)

def main(args):
    credential = get_credential()

    '''Loads the best-trained model from the sweep job and registers it'''
    print("Registering ", args.model_name)
    print("Model path:", args.model_path)                                     
    print("Registering the best trained used cars price prediction model")
    

    # Step 3: Register the logged model using its URI and model name, and retrieve its registered version.  
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.environ["AZUREML_ARM_SUBSCRIPTION"],
        resource_group_name=os.environ["AZUREML_ARM_RESOURCEGROUP"],
        workspace_name=os.environ["AZUREML_ARM_WORKSPACE_NAME"],
    )

    registered = ml_client.models.create_or_update(
        Model(
            name=args.model_name,
            path=args.model_path,     # this is your sweep output folder
            type="mlflow_model",
            description="Best model from sweep job",
        )
    )

    model_version = registered.version
    print("Registered model version:", model_version)

    # Step 4: Write model registration details, including model name and version, into a JSON file in the specified output path.  
    # Write model info
    print("Writing JSON")
    model_info = {"id": f"{args.model_name}:{model_version}"}
    output_path = args.model_info_output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)   
    with open(output_path, "w") as of:
        json.dump(model_info, of)    
        
if __name__ == "__main__":
    # Parse Arguments
    args = parse_args()   
    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Model info output path: {args.model_info_output_path}"
    ]
    for line in lines:
        print(line)   
    with mlflow.start_run(nested=True): # Starting the MLflow experiment run
        main(args)
    #mlflow.end_run() # End the MLflow run