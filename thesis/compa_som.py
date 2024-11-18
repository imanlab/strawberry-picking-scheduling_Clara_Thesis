import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# List of ground truth and predicted directories
predicted_directories = [
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/final_scheduling_som'
]
ground_truth_directories = [
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/final_scheduling_annotations_allrows_astar'
]
output_csv_file = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/compa_paths_som_a/normalized_error_results.csv'
base_output_plots_directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/compa_paths_som_a'

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

                # Read the ground truth and predicted CSV files
                ground_truth_df = pd.read_csv(ground_truth_file, delimiter=',', encoding='iso-8859-1')
                predicted_df = pd.read_csv(predicted_file, delimiter=',', encoding='iso-8859-1')

                # Ensure both dataframes have the same number of rows
                min_rows = min(len(ground_truth_df), len(predicted_df))
                ground_truth_df = ground_truth_df.iloc[:min_rows]
                predicted_df = predicted_df.iloc[:min_rows]

                # Extract the Total Length columns
                ground_truth_total_length = ground_truth_df['Scheduling Order'].tolist()
                predicted_total_length = predicted_df['Total Length'].tolist()
                
                

               
                # Compute the error for Total Length
                total_length_errors = [abs(gt - pred) / pred for gt, pred in zip(ground_truth_total_length, predicted_total_length)]
                mean_total_length_error = sum(total_length_errors) / len(total_length_errors)

            
                # Store the results
                results.append({
                    'Ground Truth File': filename,
                    'Predicted File': filename,
                    'Mean Total Length Error': mean_total_length_error,
                    
                })

                print(f"Mean Total Length Error for {filename}: {mean_total_length_error}")
               

                # Plotting the results for each pair
                indices = np.arange(len(ground_truth_total_length))
                combined_indices.extend(indices + len(combined_indices))  # Adjust indices for combined plot
                combined_ground_truth_total_lengths.extend(ground_truth_total_length)
                combined_predicted_total_lengths.extend(predicted_total_length)
                
                plt.figure(figsize=(14, 14))

                # Line plot for ground truth and predicted total lengths
                plt.subplot(2, 1, 1)
                plt.plot(indices, ground_truth_total_length, label='A star', color='blue', marker='o')
                plt.plot(indices, predicted_total_length, label='Som', color='orange', marker='x')
                plt.xlabel('Index')
                plt.ylabel('Total Length')
                plt.title(f'Total Lengths: Kmeans vs A star  for {filename}', fontsize=16)
                plt.legend()
                plt.grid(True)

                plt.tight_layout()
                plot_filename = os.path.join(output_plots_directory, f'{filename}.png')
                plt.savefig(plot_filename)
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
plt.scatter(combined_indices, combined_ground_truth_total_lengths, label='A star', color='blue', marker='o', alpha=0.7)
plt.scatter(combined_indices, combined_predicted_total_lengths, label='Som', color='orange', marker='x', alpha=0.7)
plt.axhline(y=average_ground_truth_total_length, color='blue', linestyle='--', linewidth=5, label='Avg A star Total Length')
plt.axhline(y=average_predicted_total_length, color='orange', linestyle='--', linewidth=5, label='Avg Som Total Length')
plt.xlabel('Index')
plt.ylabel('Total Length')
plt.title('Total Lengths: A star VS Som ', fontsize=18)
plt.legend()
plt.grid(True)


plt.tight_layout()
combined_plot_filename = os.path.join(base_output_plots_directory, 'combined_plot_astar_Som.png')
plt.savefig(combined_plot_filename)
plt.close()

print(f"Combined plot saved to {combined_plot_filename}")


