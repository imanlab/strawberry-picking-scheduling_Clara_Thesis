import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# List of ground truth and predicted directories
ground_truth_directories = [
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train/final_scheduling_annotations_allrows', 
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/test/final_scheduling_annotations_annotations_allrows', 
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/val/final_scheduling_annotations_allrows', 
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/final_scheduling_annotations_allrows', 
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/test/final_scheduling_annotations_allrows', 
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/val/final_scheduling_annotations_allrows'
]
predicted_directories = [
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train/final_scheduling_annotations_allrows_astar', 
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/test/final_scheduling_annotations_annotations_allrows_astar', 
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/val/final_scheduling_annotations_allrows_astar',
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/final_scheduling_annotations_allrows_astar', 
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/test/final_scheduling_annotations_allrows_astar', 
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/val/final_scheduling_annotations_allrows_astar'
]

output_csv_file = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/compa_paths_k_a/normalized_error_results.csv'
base_output_plots_directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/compa_paths_k_a'

# Ensure the base output directory exists
os.makedirs(base_output_plots_directory, exist_ok=True)

results = []

# Lists to store values for combined plotting
all_indices = []
all_ground_truth_total_lengths = []
all_predicted_total_lengths = []
all_ground_truth_scheduling_orders = []
all_predicted_scheduling_orders = []

# Set a style
sns.set(style="whitegrid")

# Function to extract the number from the filename
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return match.group() if match else None

# Function to get the last two components of a path
def get_last_two_components(path):
    components = path.rstrip('/').split(os.sep)
    return '_'.join(components[-2:])

# Iterate over the ground truth and predicted directories
for gt_dir, pred_dir in zip(ground_truth_directories, predicted_directories):
    try:
        # Get a list of all CSV files in the current ground truth and predicted directories
        ground_truth_files = glob.glob(os.path.join(gt_dir, "*.csv"))
        predicted_files = glob.glob(os.path.join(pred_dir, "*.csv"))

        if not ground_truth_files or not predicted_files:
            print(f"No CSV files found in {gt_dir} or {pred_dir}")
            continue

        # Create a dictionary to map numbers to ground truth and predicted files
        ground_truth_map = {extract_number(f): f for f in ground_truth_files}
        predicted_map = {extract_number(f): f for f in predicted_files}

        # Create a subdirectory for the output plots
        output_plots_directory = os.path.join(base_output_plots_directory, get_last_two_components(gt_dir))
        os.makedirs(output_plots_directory, exist_ok=True)

        # Iterate over the ground truth files and find the corresponding predicted files
        for number, ground_truth_file in ground_truth_map.items():
            predicted_file = predicted_map.get(number)

            if predicted_file:
                try:
                    # Extract the name to keep track of results separately
                    ground_truth_file_name = os.path.basename(ground_truth_file)
                    predicted_file_name = os.path.basename(predicted_file)

                    print(f"Processing ground truth file: {ground_truth_file_name} with predicted file: {predicted_file_name}")

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

                    # Extract the Scheduling Order columns
                    ground_truth_scheduling_order = ground_truth_df['Scheduling Order'].tolist()
                    predicted_scheduling_order = predicted_df['Scheduling Order'].tolist()

                    # Compute the error for Total Length
                    total_length_errors = [(gt - pred) / pred for gt, pred in zip(ground_truth_total_length, predicted_total_length)]
                    mean_total_length_error = sum(total_length_errors) / len(total_length_errors)

                    # Compute the error for Scheduling Order
                    scheduling_order_errors = [(gt - pred) / pred for gt, pred in zip(ground_truth_scheduling_order, predicted_scheduling_order)]
                    mean_scheduling_order_error = sum(scheduling_order_errors) / len(scheduling_order_errors)

                    # Store the results
                    results.append({
                        'Ground Truth File': ground_truth_file_name,
                        'Predicted File': predicted_file_name,
                        'Mean Total Length Error': mean_total_length_error,
                        'Mean Scheduling Order Error': mean_scheduling_order_error
                    })

                    print(f"Mean Total Length Error for {ground_truth_file_name} and {predicted_file_name}: {mean_total_length_error}")
                    print(f"Mean Scheduling Order Error for {ground_truth_file_name} and {predicted_file_name}: {mean_scheduling_order_error}\n")

                    # Plotting the results for each pair
                    indices = np.arange(len(ground_truth_total_length))
                    all_indices.extend(indices + len(all_indices))  # Adjust indices for combined plot
                    all_ground_truth_total_lengths.extend(ground_truth_total_length)
                    all_predicted_total_lengths.extend(predicted_total_length)
                    all_ground_truth_scheduling_orders.extend(ground_truth_scheduling_order)
                    all_predicted_scheduling_orders.extend(predicted_scheduling_order)

                    plt.figure(figsize=(14, 14))

                    # Line plot for ground truth and predicted total lengths
                    plt.subplot(2, 1, 1)
                    plt.plot(indices, ground_truth_total_length, label='kmeans', color='blue', marker='o')
                    plt.plot(indices, predicted_total_length, label='A star', color='orange', marker='x')
                    plt.xlabel('Index')
                    plt.ylabel('Total Length')
                    plt.title(f'Total Lengths: Kmeans vs A star for {ground_truth_file_name}', fontsize=16)
                    plt.legend()
                    plt.grid(True)

                    # Line plot for ground truth and predicted scheduling orders
                    plt.subplot(2, 1, 2)
                    plt.plot(indices, ground_truth_scheduling_order, label='Ground Truth', color='blue', marker='o')
                    plt.plot(indices, predicted_scheduling_order, label='Predicted', color='orange', marker='x')
                    plt.xlabel('Index')
                    plt.ylabel('Scheduling Order')
                    plt.title(f'Scheduling Orders values: kmeans vs A star for {ground_truth_file_name}', fontsize=16)
                    plt.legend()
                    plt.grid(True)

                    plt.tight_layout()
                    plot_filename = os.path.join(output_plots_directory, f'{ground_truth_file_name}_vs_{predicted_file_name}.png')
                    plt.savefig(plot_filename)
                    plt.close()

                except Exception as e:
                    print(f"Error processing files {ground_truth_file_name} and {predicted_file_name}: {e}")
    except Exception as e:
        print(f"Error processing directory {gt_dir} and {pred_dir}: {e}")

# Save results to a new CSV file
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_file, index=False)

print(f"Results saved to {output_csv_file}")

# Combined plot for all files
plt.figure(figsize=(14, 14))

# Scatter plot for combined ground truth and predicted total lengths
plt.subplot(2, 1, 1)
plt.scatter(all_indices, all_ground_truth_total_lengths, label='Kmeans', color='blue', marker='o', alpha=0.7)
plt.scatter(all_indices, all_predicted_total_lengths, label='A star', color='orange', marker='x', alpha=0.7)
plt.xlabel('Index')
plt.ylabel('Total Length')
plt.title('Total Lengths: Kmeans vs A star (Combined)', fontsize=18)
plt.legend()
plt.grid(True)

# Scatter plot for combined ground truth and predicted scheduling orders
plt.subplot(2, 1, 2)
plt.scatter(all_indices, all_ground_truth_scheduling_orders, label='K means', color='blue', marker='o', alpha=0.7)
plt.scatter(all_indices, all_predicted_scheduling_orders, label='A star', color='orange', marker='x', alpha=0.7)
plt.xlabel('Index')
plt.ylabel('Scheduling Order')
plt.title('Scheduling Orders: Kmeans vs A star (Combined)', fontsize=18)
plt.legend()
plt.grid(True)

plt.tight_layout()
combined_plot_filename = os.path.join(base_output_plots_directory, 'combined_plot_kmeans_vs_astar.png')
plt.savefig(combined_plot_filename)
plt.close()

print(f"Combined plot saved to {combined_plot_filename}")

