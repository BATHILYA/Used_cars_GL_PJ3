# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""

import os
import argparse
import logging
import mlflow
import pandas as pd  
from sklearn.preprocessing import LabelEncoder                                           
from sklearn.model_selection import train_test_split
from pathlib import Path


def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("prep")  # Create an ArgumentParser object
    parser.add_argument("--raw_data", type=str, help="Path to raw data")  # Specify the type for raw data (str)
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Test-train ratio")  # Specify the type (float) and default value (0.2) for test-train ratio
    parser.add_argument("--train_data", type=str, help="Path to train dataset")  # Specify the type for train data (str)
    parser.add_argument("--test_data", type=str, help="Path to test dataset")  # Specify the type for test data (str)
    args = parser.parse_args()

    return args

def main(args):  # Write the function name for the main data preparation logic
    '''Read, preprocess, split, and save datasets'''
    # Log arguments
    logging.info(f"Input data path: {args.raw_data}")
    logging.info(f"Test-train ratio: {args.test_train_ratio}")                                                         

    # Reading Data
    df = pd.read_csv(args.raw_data)

    # ------- WRITE YOUR CODE HERE -------
    # Step 1: Deterministic encoding for Segment to keep training/inference consistent.
    segment_map = {
        "luxury segment": 0,
        "non-luxury segment": 1,
    }
    unknown = set(df["Segment"].dropna().unique()) - set(segment_map.keys())
    if unknown:
        raise ValueError(f"Unknown Segment values found: {sorted(unknown)}")
    df["Segment"] = df["Segment"].map(segment_map).astype(int)
    mlflow.log_param("segment_encoding", str(segment_map))


    # Step 2: Split the dataset into training and testing sets using train_test_split with specified test size and random state.  
    # Split Data into train and test datasets
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)


    # Step 3: Save the training and testing datasets as CSV files in separate directories for easier access and organization.  
    # Save train and test data
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)
    train_df.to_csv(os.path.join(args.train_data, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data, "test.csv"), index=False)

    # Step 4: Log the number of rows in the training and testing datasets as metrics for tracking and evaluation.  
    # Log the metrics
    mlflow.log_metric('train size', train_df.shape[0])
    mlflow.log_metric('test size', test_df.shape[0])
if __name__ == "__main__":
    # Parse Arguments
    args = parse_args()  # Call the function to parse arguments

    lines = [
        f"Raw data path: {args.raw_data}",  # Print the raw_data path
        f"Train dataset output path: {args.train_data}",  # Print the train_data path
        f"Test dataset path: {args.test_data}",  # Print the test_data path
        f"Test-train ratio: {args.test_train_ratio}",  # Print the test_train_ratio
    ]
                      
    for line in lines:
        print(line)    
        
    with mlflow.start_run(nested=True):

        main(args)

    # mlflow.end_run()
