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
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/test/final_scheduling_annotations_allrows', 
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/val/final_scheduling_annotations_allrows', 
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/final_scheduling_annotations_allrows', 
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/test/final_scheduling_annotations_allrows', 
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/val/final_scheduling_annotations_allrows'
]
​
predicted_directories = [
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train/final_scheduling_annotations_allrows_astar', 
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/test/final_scheduling_annotations_allrows_astar', 
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
combined_indices = []
combined_ground_truth_total_lengths = []
combined_predicted_total_lengths = []
combined_ground_truth_scheduling_orders = []
combined_predicted_scheduling_orders = []

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

def associate_files(gt_dir, pred_dir):
    # Get a list of all CSV files in the current ground truth and predicted directories
    ground_truth_files = glob.glob(os.path.join(gt_dir, "*.csv"))
    predicted_files = glob.glob(os.path.join(pred_dir, "*.csv"))

    if not ground_truth_files or not predicted_files:
        print(f"No CSV files found in {gt_dir} or {pred_dir}")
        return {}

    # Create a dictionary to map filenames to their full paths for ground truth and predicted files
    ground_truth_map = {os.path.basename(f): f for f in ground_truth_files}
    predicted_map = {os.path.basename(f): f for f in predicted_files}

    # Create a dictionary to store the associations
    associations = {}

    # Iterate over the ground truth files and find the corresponding predicted files by filename
    for filename in ground_truth_map:
        if filename in predicted_map:
            associations[filename] = (ground_truth_map[filename], predicted_map[filename])

    # Print the associations
    print(f"\nAssociations for directories:\nGround Truth: {gt_dir}\nPredicted: {pred_dir}")
    for filename, (gt_file, pred_file) in associations.items():
        print(f"{filename}:")
        print(f"  Ground Truth: {gt_file}")
        print(f"  Predicted: {pred_file}")

    return associations

# Iterate over the ground truth and predicted directories
for gt_dir, pred_dir in zip(ground_truth_directories, predicted_directories):
    try:
        # Get the file associations
        associations = associate_files(gt_dir, pred_dir)

        if not associations:
            continue

        # Create a subdirectory for the output plots
        output_plots_directory = os.path.join(base_output_plots_directory, get_last_two_components(gt_dir))
        os.makedirs(output_plots_directory, exist_ok=True)

        # Iterate over the associations and process the files
        for filename, (ground_truth_file, predicted_file) in associations.items():
            try:
                print(f"Processing ground truth file: {filename} with predicted file: {filename}")

                # Read only the first row of the ground truth and predicted CSV files for plotting
                ground_truth_first_row = pd.read_csv(ground_truth_file, delimiter=',', encoding='iso-8859-1', nrows=1)
                predicted_first_row = pd.read_csv(predicted_file, delimiter=',', encoding='iso-8859-1', nrows=1)

                # Read the entire ground truth and predicted CSV files for error computation
                ground_truth_df = pd.read_csv(ground_truth_file, delimiter=',', encoding='iso-8859-1')
                predicted_df = pd.read_csv(predicted_file, delimiter=',', encoding='iso-8859-1')

                # Ensure both dataframes have the same number of rows
                min_rows = min(len(ground_truth_df), len(predicted_df))
                ground_truth_df = ground_truth_df.iloc[:min_rows]
                predicted_df = predicted_df.iloc[:min_rows]

                # Extract the Total Length columns for the first row
                ground_truth_total_length_first = ground_truth_first_row['Total Distance'].tolist()
                predicted_total_length_first = predicted_first_row['Total Distance'].tolist()

                # Extract the Scheduling Order columns for the first row
                ground_truth_scheduling_order_first = ground_truth_first_row['Scheduling Order'].tolist()
                predicted_scheduling_order_first = predicted_first_row['Scheduling Order'].tolist()

                # Extract the Total Length columns for all rows
                ground_truth_total_length_all = ground_truth_df['Total Distance'].tolist()
                predicted_total_length_all = predicted_df['Total Distance'].tolist()

                # Extract the Scheduling Order columns for all rows
                ground_truth_scheduling_order_all = ground_truth_df['Scheduling Order'].tolist()
                predicted_scheduling_order_all = predicted_df['Scheduling Order'].tolist()

                # Compute the error for Total Length for the first row
                total_length_errors_first = [abs(gt - pred) / pred for gt, pred in zip(ground_truth_total_length_first, predicted_total_length_first)]
                mean_total_length_error_first = sum(total_length_errors_first) / len(total_length_errors_first)

                # Compute the error for Scheduling Order for the first row
                scheduling_order_errors_first = [abs(gt - pred) / pred for gt, pred in zip(ground_truth_scheduling_order_first, predicted_scheduling_order_first)]
                mean_scheduling_order_error_first = sum(scheduling_order_errors_first) / len(scheduling_order_errors_first)

                # Compute the error for Total Length for all rows
                total_length_errors_all = [abs(gt - pred) / pred for gt, pred in zip(ground_truth_total_length_all, predicted_total_length_all)]
                mean_total_length_error_all = sum(total_length_errors_all) / len(total_length_errors_all)

                # Compute the error for Scheduling Order for all rows
                scheduling_order_errors_all = [abs(gt - pred) / pred for gt, pred in zip(ground_truth_scheduling_order_all, predicted_scheduling_order_all)]
                mean_scheduling_order_error_all = sum(scheduling_order_errors_all) / len(scheduling_order_errors_all)

                # Store the results
                results.append({
                    'Ground Truth File': filename,
                    'Predicted File': filename,
                    'Mean Total Length Error (First Row)': mean_total_length_error_first,
                    'Mean Scheduling Order Error (First Row)': mean_scheduling_order_error_first,
                    'Mean Total Length Error (All Rows)': mean_total_length_error_all,
                    'Mean Scheduling Order Error (All Rows)': mean_scheduling_order_error_all
                })

                print(f"Mean Total Length Error (First Row) for {filename}: {mean_total_length_error_first}")
                print(f"Mean Scheduling Order Error (First Row) for {filename}: {mean_scheduling_order_error_first}")
                print(f"Mean Total Length Error (All Rows) for {filename}: {mean_total_length_error_all}")
                print(f"Mean Scheduling Order Error (All Rows) for {filename}: {mean_scheduling_order_error_all}\n")

                # Plotting the results for the first row
                indices = np.arange(len(ground_truth_total_length_first))
                combined_indices.extend(indices + len(combined_indices))  # Adjust indices for combined plot
                combined_ground_truth_total_lengths.extend(ground_truth_total_length_first)
                combined_predicted_total_lengths.extend(predicted_total_length_first)
                combined_ground_truth_scheduling_orders.extend(ground_truth_scheduling_order_first)
                combined_predicted_scheduling_orders.extend(predicted_scheduling_order_first)

                plt.figure(figsize=(14, 14))

                # Line plot for ground truth and predicted total lengths for the first row
                plt.subplot(2, 1, 1)
                plt.plot(indices, ground_truth_total_length_first, label='Kmeans', color='blue', marker='o')
                plt.plot(indices, predicted_total_length_first, label='A star', color='orange', marker='x')
                plt.xlabel('Index')
                plt.ylabel('Total Length')
                plt.title(f'Total Lengths: Kmeans VS A star for {filename}', fontsize=16)
                plt.legend()
                plt.grid(True)

                # Line plot for ground truth and predicted scheduling orders for the first row
                plt.subplot(2, 1, 2)
                plt.plot(indices, ground_truth_scheduling_order_first, label='K means', color='blue', marker='o')
                plt.plot(indices, predicted_scheduling_order_first, label='A star ', color='orange', marker='x')
                plt.xlabel('Index')
                plt.ylabel('Scheduling Order')
                plt.title(f'Scheduling Orders: K means VS A star  for {filename}', fontsize=16)
                plt.legend()
                plt.grid(True)

                plt.tight_layout()
                plot_filename = os.path.join(output_plots_directory, f'{filename}.png')
                plt.savefig(plot_filename)
                plt.close()

                # Plotting the results for all rows
                num_rows = min(10, len(ground_truth_total_length_all))  # Plotting for the first 10 rows
                indices = np.arange(num_rows)

                plt.figure(figsize=(14, 14))

                # Line plot for ground truth and predicted total lengths for all rows
                plt.subplot(2, 1, 1)
                plt.plot(indices, ground_truth_total_length_all[:num_rows], label='K means', color='blue', marker='o')
                plt.plot(indices, predicted_total_length_all[:num_rows], label='A star', color='orange', marker='x')
                plt.xlabel('Index')
                plt.ylabel('Total Length')
                plt.title(f'Total Lengths (First {num_rows} Rows): Kmeans VS A star for {filename}', fontsize=16)
                plt.legend()
                plt.grid(True)

                # Line plot for ground truth and predicted scheduling orders for all rows
                plt.subplot(2, 1, 2)
                plt.plot(indices, ground_truth_scheduling_order_all[:num_rows], label='Kmeans', color='blue', marker='o')
                plt.plot(indices, predicted_scheduling_order_all[:num_rows], label='A star', color='orange', marker='x')
                plt.xlabel('Index')
                plt.ylabel('Scheduling Order')
                plt.title(f'Scheduling Orders (First {num_rows} Rows): Kmeans VS A star for {filename}', fontsize=16)
                plt.legend()
                plt.grid(True)

                plt.tight_layout()
                plot_filename_all_rows = os.path.join(output_plots_directory, f'{filename}_all_rows.png')
                plt.savefig(plot_filename_all_rows)
                plt.close()

            except Exception as e:
                print(f"Error processing files {filename}: {e}")
    except Exception as e:
        print(f"Error processing directory {gt_dir} and {pred_dir}: {e}")

# Save results to a new CSV file
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_file, index=False)

print(f"Results saved to {output_csv_file}")

# Compute the average total lengths
average_ground_truth_total_length = np.mean(combined_ground_truth_total_lengths)
average_predicted_total_length = np.mean(combined_predicted_total_lengths)

# Combined plot for all files
plt.figure(figsize=(14, 14))

# Scatter plot for combined ground truth and predicted total lengths
plt.subplot(2, 1, 1)
plt.scatter(combined_indices, combined_ground_truth_total_lengths, label='Kmeans', color='blue', marker='o', alpha=0.7)
plt.scatter(combined_indices, combined_predicted_total_lengths, label='A star', color='orange', marker='x', alpha=0.7)
plt.axhline(y=average_ground_truth_total_length, color='blue', linestyle='--', linewidth=5, label='Avg Kmeans Total Length')
plt.axhline(y=average_predicted_total_length, color='orange', linestyle='--', linewidth=5, label='Avg A star Total Length')
plt.xlabel('Index')
plt.ylabel('Total Length')
plt.title('Total Lengths: Kmeans VS A star', fontsize=18)
plt.legend()
plt.grid(True)

# Compute the average scheduling orders
average_ground_truth_scheduling_order = np.mean(combined_ground_truth_scheduling_orders)
average_predicted_scheduling_order = np.mean(combined_predicted_scheduling_orders)

# Scatter plot for combined ground truth and predicted scheduling orders
plt.subplot(2, 1, 2)
plt.scatter(combined_indices, combined_ground_truth_scheduling_orders, label='Kmeans', color='blue', marker='o', alpha=0.7)
plt.scatter(combined_indices, combined_predicted_scheduling_orders, label='A star', color='orange', marker='x', alpha=0.7)
plt.axhline(y=average_ground_truth_scheduling_order, color='blue', linestyle='--', linewidth=5, label='Avg Kmeans Scheduling Order')
plt.axhline(y=average_predicted_scheduling_order, color='orange', linestyle='--', linewidth=5, label='Avg A star Scheduling Order')
plt.xlabel('Index')
plt.ylabel('Scheduling Order')
plt.title('Scheduling Orders: Kmeans VS A star' , fontsize=18)
plt.legend()
plt.grid(True)

plt.tight_layout()
combined_plot_filename = os.path.join(base_output_plots_directory, 'combined_plot_k_a.png')
plt.savefig(combined_plot_filename)
plt.close()

print(f"Combined plot saved to {combined_plot_filename}")

