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

def main(args):
    # Ensure MLflow is pointing to the workspace tracking server if provided by AML
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        
    # Register from local MLflow model directory
    model_uri = f"file://{args.model_path}"                                                   
    '''Loads the best-trained model from the sweep job and registers it'''
    print("Registering ", args.model_name)
    print("Model path:", args.model_path)                                     
    print("Registering the best trained used cars price prediction model")
    print("MLflow tracking uri:", mlflow.get_tracking_uri())

    # Step 3: Register the logged model using its URI and model name, and retrieve its registered version.  
    mv = mlflow.register_model(model_uri=model_uri, name=args.model_name)
    model_version = mv.version
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