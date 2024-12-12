import os
import glob
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder


from pathlib import Path


def file_processor():
    """
    Kludged together way to handle data
    """
    # Define the directory path
    directory_path = r"C:\git_repos\train-commerce-forecasting\data"

    # Get the list of all files in the directory
    files = os.listdir(directory_path)

    # Filter to grab only the files (not directories)
    files = [file for file in files if os.path.isfile(os.path.join(directory_path, file))]

    train=files[0:8]
    test=files[8:9]

    x_train_All_Traffic = []
    y_train_All_Traffic = []
    x_test_All_Traffic = []
    y_test_All_Traffic = []
    #x_validation_All_Traffic = []
    #y_validation_All_Traffic = []

    # Process the training files
    for file in train:
        year = int(file[12:16])
        # Read the CSV file
        one_year_data_All_Traffic = pd.read_csv(os.path.join(directory_path, file))
        
        # Drop rows with missing values
        one_year_data_All_Traffic.dropna(inplace=True)
        
        # Split into features and target
        x_All_Traffic = one_year_data_All_Traffic.drop(columns=['log_carloads','BEA_origin','BEA_dest'], inplace=False)
        y_All_Traffic = one_year_data_All_Traffic['log_carloads']
        
        # Append data to training lists
        x_train_All_Traffic.append(x_All_Traffic)
        y_train_All_Traffic.append(y_All_Traffic)

    # Process the testing file
    for file in test:
        year = int(file[12:16])
        # Read the CSV file
        one_year_data_test_All_Traffic = pd.read_csv(os.path.join(directory_path, file))
        
        # Drop rows with missing values
        one_year_data_test_All_Traffic.dropna(inplace=True)
        
        # Split into features and target
        x_test_All_Traffic1 = one_year_data_test_All_Traffic.drop(columns=['log_carloads','BEA_origin','BEA_dest'], inplace=False)
        y_test_All_Traffic1 = one_year_data_test_All_Traffic['log_carloads']
        
        # Append data to testing lists
        x_test_All_Traffic.append(x_test_All_Traffic1)
        y_test_All_Traffic.append(y_test_All_Traffic1)

    # Process the validation file
    #for file in validation:
    #    # Read the CSV file
    #    one_year_data_validation_All_Traffic = pd.read_csv(os.path.join(directory_path, file))
    #    
    #    # Drop rows with missing values
    #    one_year_data_validation_All_Traffic.dropna(inplace=True)
    #    
    #    # Split into features and target
    #    x_validation_All_Traffic1 = one_year_data_validation_All_Traffic.drop(columns=['log_carloads','BEA_origin','BEA_dest'], inplace=False)
    #    y_validation_All_Traffic1 = one_year_data_validation_All_Traffic['log_carloads']
    #    
    #    # Append data to validation lists
    #    x_validation_All_Traffic.append(x_validation_All_Traffic1)
    #    y_validation_All_Traffic.append(y_validation_All_Traffic1)

    # Concatenate all the data into DataFrames
    x_train_All_Traffic = pd.concat(x_train_All_Traffic)
    y_train_All_Traffic = pd.concat(y_train_All_Traffic)
    x_test_All_Traffic = pd.concat(x_test_All_Traffic)
    y_test_All_Traffic = pd.concat(y_test_All_Traffic)
    #x_validation_All_Traffic = pd.concat(x_validation_All_Traffic)
    #y_validation_All_Traffic = pd.concat(y_validation_All_Traffic)

    # Scaling the data
    scaler = StandardScaler()
    x_train_All_Traffic_scaled = scaler.fit_transform(x_train_All_Traffic)
    x_test_All_Traffic_scaled = scaler.transform(x_test_All_Traffic)
    #x_validation_All_Traffic_scaled = scaler.transform(x_validation_All_Traffic)

    return x_train_All_Traffic, y_train_All_Traffic, x_test_All_Traffic, y_test_All_Traffic

if __name__ == "__main__":
    pass