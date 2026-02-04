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
    '''Loads the best-trained model from the sweep job and registers it'''

    print("Registering ", args.model_name)

    # Step 1: Load the model from the specified path using `mlflow.sklearn.load_model` for further processing.  
    # Load model
    model = mlflow.sklearn.load_model(f"file://{args.model_path}")
    
    # Step 2: Log the loaded model in MLflow with the specified model name for versioning and tracking.  
    # Log model using mlflow
    print("Registering the best trained used cars price prediction model")

    mlflow.sklearn.log_model(model, args.model_name)
    
    # Step 3: Register the logged model using its URI and model name, and retrieve its registered version.  
# Register logged model using mlflow
    run_id = mlflow.active_run().info.run_id
    model_uri = f'runs:/{run_id}/{args.model_name}'
    mlflow_model = mlflow.register_model(model_uri, args.model_name)
    model_version = mlflow_model.version
    
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