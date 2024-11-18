import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# List of ground truth and predicted directories
ground_truth_directories= [
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train/final_scheduling_annotations_allrows_astar', 
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/test/final_scheduling_annotations_allrows_astar', 
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/val/final_scheduling_annotations_allrows_astar', 
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/final_scheduling_annotations_allrows_astar', 
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/test/final_scheduling_annotations_allrows_astar', 
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/val/final_scheduling_annotations_allrows_astar'
]

 
predicted_directories = [
     '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train/coordinates_for_images/output_02', 
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/test/coordinates_for_images/output_02', 
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/val/coordinates_and_hop/output_02',
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/coordinates_for_images/output_02', 
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/test/coordinates_for_images/output_02', 
    '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/val/coordinates_for_images/output_02'

]
output_csv_file = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/compa_a_som_cnn/normalized_error_results.csv'
base_output_plots_directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/compa_a_som_cnn'

# Ensure the base output directory exists
os.makedirs(base_output_plots_directory, exist_ok=True)

results = []

# Lists to store values for combined plotting
combined_indices = []
combined_predicted_total_lengths = []
combined_ground_truth_scheduling_orders = []

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
        #breakpoint()
        

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
                # Read only the first row of the ground truth and predicted CSV files for plotting
                ground_truth_first_row = pd.read_csv(ground_truth_file, delimiter=',', encoding='iso-8859-1', nrows=1)
                predicted_first_row = pd.read_csv(predicted_file, delimiter=',', encoding='iso-8859-1', nrows=1)

                # Extract the Scheduling Order columns for the first row
                ground_truth_scheduling_order_first = ground_truth_first_row['Scheduling Order'].tolist()
                predicted_total_length_first = predicted_first_row['Total Length'].tolist()

                # Compute the error for Total Length compared to Scheduling Order for the first row
                total_length_errors_first = [abs(gt - pred) / gt for gt, pred in              zip(ground_truth_scheduling_order_first, predicted_total_length_first)]
                mean_total_length_error_first = sum(total_length_errors_first) / len(total_length_errors_first)

                # Store the results for the first row
                results.append({
                  'Ground Truth File': filename,
                  'Predicted File': filename,
                  'Mean Total Length Error': mean_total_length_error_first,
                })

                print(f"Mean Total Length Error for {filename}: {mean_total_length_error_first}\n")

                # Plotting the results for the first row
                indices = np.arange(len(ground_truth_scheduling_order_first))
                combined_indices.extend(indices + len(combined_indices))  # Adjust indices for combined plot
                combined_ground_truth_scheduling_orders.extend(ground_truth_scheduling_order_first)
                         
                combined_predicted_total_lengths.extend(predicted_total_length_first)



                plt.figure(figsize=(14, 14))

                #Line plot for ground truth scheduling orders and predicted total lengths for the first row
                plt.plot(indices, ground_truth_scheduling_order_first, label='Scheduling Order A star (Ground Truth)', color='blue', marker='o')
                plt.plot(indices, predicted_total_length_first, label='Total Length (Predicted)', color='orange', marker='x')
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.title(f'Scheduling Order A star  (Ground Truth) VS Total Length (Predicted) for {filename}', fontsize=16)
                plt.legend()
                plt.grid(True)

                plt.tight_layout()
                plot_filename_first_row = os.path.join(output_plots_directory, f'{filename}_first_row.png')
                plt.savefig(plot_filename_first_row)
                plt.close()

            except Exception as e:
                print(f"Error processing files {filename}: {e}")
    except Exception as e:
        print(f"Error processing directory {gt_dir} and {pred_dir}: {e}")

# Save results to a new CSV file
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_file, index=False)

print(f"Results saved to {output_csv_file}")

# Compute the average scheduling orders and total lengths
average_ground_truth_scheduling_order = np.mean(combined_ground_truth_scheduling_orders)
average_predicted_total_length = np.mean(combined_predicted_total_lengths)

# Combined plot for all files
plt.figure(figsize=(14, 14))

# Scatter plot for combined ground truth scheduling orders and predicted total lengths
plt.scatter(combined_indices, combined_ground_truth_scheduling_orders, label='Scheduling Order A star (Ground Truth)', color='blue', marker='o', alpha=0.7)
plt.scatter(combined_indices, combined_predicted_total_lengths, label='Total Length (Predicted)', color='orange', marker='x', alpha=0.7)
plt.axhline(y=average_ground_truth_scheduling_order, color='blue', linestyle='--', linewidth=5, label='Avg Scheduling Order A star(Ground Truth)')
plt.axhline(y=average_predicted_total_length, color='orange', linestyle='--', linewidth=5, label='Avg Total Length (Predicted)')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Scheduling Order A star (Ground Truth) VS Total Length (Predicted)', fontsize=18)
plt.legend()
plt.grid(True)

plt.tight_layout()
combined_plot_filename = os.path.join(base_output_plots_directory, 'combined_plot_a_som.png')
plt.savefig(combined_plot_filename)
plt.close()

print(f"Combined plot saved to {combined_plot_filename}")


