import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import re
import numpy as np
import matplotlib.pyplot as plt

# Paths to the single files in ground truth and predicted directories
ground_truth_file = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train/final_scheduling_annotations_allrows/30_results_output.csv'
predicted_file = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train/final_scheduling_annotations_allrows_astar/results_30_output.csv'
output_csv_file = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train/compa_paths/results_k_a.csv'

# Read the ground truth and predicted CSV files
ground_truth_df = pd.read_csv(ground_truth_file, delimiter=',', encoding='iso-8859-1')
predicted_df = pd.read_csv(predicted_file, delimiter=',', encoding='iso-8859-1')

# Ensure both dataframes have the same number of rows
min_rows = min(len(ground_truth_df), len(predicted_df))
ground_truth_df = ground_truth_df.iloc[:min_rows]
predicted_df = predicted_df.iloc[:min_rows]

# Extract the Total Length columns
ground_truth_total_length = ground_truth_df['Total Distance'].tolist()
predicted_total_length = predicted_df['Total Distance'].tolist()

# Calculate the range of Total Length in the ground truth file
total_length_range = max(ground_truth_total_length) - min(ground_truth_total_length)

# Calculate the normalized MSE for Total Length
total_length_mse = mean_squared_error(ground_truth_total_length, predicted_total_length)
normalized_total_length_mse = total_length_mse / total_length_range if total_length_range != 0 else total_length_mse

# Extract the indices for plotting
indices = np.arange(len(ground_truth_total_length))

# Plotting the results
plt.figure(figsize=(14, 7))

# Line plot for ground truth and predicted total lengths
plt.plot(indices, ground_truth_total_length, label='Ground Truth', color='blue', marker='o')
plt.plot(indices, predicted_total_length, label='Predicted', color='orange', marker='x')
plt.xlabel('Index')
plt.ylabel('Total Length')
plt.title('Total Lengths: Ground Truth vs Predicted')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Save results to a CSV file
results = [{
    'Ground Truth File': os.path.basename(ground_truth_file),
    'Predicted File': os.path.basename(predicted_file),
    'Normalized Total Length MSE': normalized_total_length_mse
}]
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_file, index=False)

print(f"Results saved to {output_csv_file}")

