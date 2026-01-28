# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    
    # Step 1: Define arguments for train data, test data, model output, and RandomForest hyperparameters.  
    parser.add_argument("--train_data", type=str, help="Path to train data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='The number of trees in the forest')
    parser.add_argument('--max_depth', type=int, default=None,
                        help='The maximum depth of the tree.')
                        
    args = parser.parse_args()

    return args

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # -------- WRITE YOUR CODE HERE --------
    # Step 2: Read the train and test datasets from the provided paths using pandas.  
    train_df = pd.read_csv(Path(args.train_data)/"train.csv")
    test_df = pd.read_csv(Path(args.test_data)/"test.csv")

    # Step 3: Split the data into features (X) and target (y) for both train and test datasets.   
    y_train = train_df["price"].values  # 'price' is the target variable in this case study
    X_train = train_df.drop("price", axis=1).values
    y_test = test_df["price"].values  # 'price' is the target variable for testing
    X_test = test_df.drop("price", axis=1).values

    # Step 4: Initialize the RandomForest Regressor with specified hyperparameters, and train the model using the training data.  
    # Initialize and train a RandomForest Regressor
    rf_model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth,random_state=42)
    rf_model = rf_model.fit(X_train, y_train)

    # Step 5: Log model hyperparameters like 'n_estimators' and 'max_depth' for tracking purposes in MLflow.  
    mlflow.log_param("model", "rf_model")  # Provide the model name
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # Step 6: Predict target values on the test dataset using the trained model, and calculate the mean squared error.  
    rf_predictions = rf_model.predict(X_test)

    # Step 7: Log the R_squared metric in MLflow for model evaluation, and save the trained model to the specified output path.  
    # Compute and log R2 accuracy for test data
    r2 = r2_score(y_test, rf_predictions)
    print('R Squared of RandomForest Regressor on test set: {:.2f}'.format(r2))

    # Logging the R2 score as a metric
    mlflow.log_metric("R_squared", float(r2))  # Log the R²


    # Save the trained model
    mlflow.sklearn.save_model(rf_model, args.model_output)

if __name__ == "__main__":
    
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()

