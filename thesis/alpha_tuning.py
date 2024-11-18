import os
import glob
import pandas as pd

def find_min_total_length(folder_path):
    results = []

    # Iterate over each subdirectory in the given folder
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)

        if os.path.isdir(subdir_path):
            min_total_length = float('inf')
            corresponding_alpha = None

            # Iterate over each CSV file in the subdirectory
            for csv_file in glob.glob(os.path.join(subdir_path, "*.csv")):
                df = pd.read_csv(csv_file)

                # Filter out rows with alpha=0.0
                df = df[df['Alpha'] != 0.0]

                # Find the minimum total length and corresponding alpha value in the current file
                if not df.empty:
                    min_length_row = df.loc[df['Total Length'].idxmin()]
                    if min_length_row['Total Length'] < min_total_length:
                        min_total_length = min_length_row['Total Length']
                        corresponding_alpha = min_length_row['Alpha']

            if corresponding_alpha is not None:
                results.append({
                    'Folder': subdir,
                    'Min Total Length': min_total_length,
                    'Corresponding Alpha': corresponding_alpha
                })

    return results

def save_results_to_csv(results, output_csv):
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

# Input folder path and output CSV file path
input_folder_path = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/coordinates_and_hop/new_alpha_tuning/output'  # Update with the actual path
output_csv_file = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/coordinates_and_hop/new_alpha_tuning/output/output.csv'  # Update with the actual path

# Find minimum total length and corresponding alpha
results = find_min_total_length(input_folder_path)

# Save results to a CSV file
save_results_to_csv(results, output_csv_file)

print(f"Results saved to {output_csv_file}")

