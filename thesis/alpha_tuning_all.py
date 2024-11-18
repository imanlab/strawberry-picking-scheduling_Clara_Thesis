import os
import glob
import pandas as pd
import numpy as np

def find_best_alpha(alpha_folder, scheduling_order_folder, output_csv):
    results = []

    # Iterate over each subdirectory in the given folder
    for subdir in os.listdir(alpha_folder):
        subdir_path = os.path.join(alpha_folder, subdir)
        print(subdir_path)

        if os.path.isdir(subdir_path):
            min_mre = float('inf')
            best_alpha = None
            print('ciao')

            # Iterate over each CSV file in the subdirectory
            for csv_file in glob.glob(os.path.join(subdir_path, "*.csv")):
                try:
                    df_alpha = pd.read_csv(csv_file)
                    
                    # Skip computation if alpha is 0.0
                    alpha_value = df_alpha['Alpha'][0]
                    if alpha_value == 0.0:
                        print(f"Skipping computation for alpha = {alpha_value}")
                        continue
                    
                    # Extract file number from filename
                    file_num = os.path.splitext(os.path.basename(csv_file))[0].split('_')[-2]
                    print(file_num)
                    scheduling_order_file = os.path.join(scheduling_order_folder, f"{file_num}_results_output.csv")
                    print(scheduling_order_file)

                    if os.path.exists(scheduling_order_file):
                        df_scheduling = pd.read_csv(scheduling_order_file)
                        common_length = min(len(df_alpha), len(df_scheduling))
                        total_length_alpha = df_alpha['Total Length'][:common_length]
                        print(total_length_alpha)
                        scheduling_order = df_scheduling['Scheduling Order'][:common_length]
                        print(scheduling_order)

                        # Compute MRE
                        mre = np.mean(np.abs((total_length_alpha - scheduling_order) / scheduling_order))
                        print(mre)
                        if mre < min_mre:
                            min_mre = mre
                            best_alpha = alpha_value  # Assuming alpha is the same for all rows in the file
                except Exception as e:
                    print(f"Error processing file {csv_file}: {e}")

            if best_alpha is not None:
                results.append({
                    'Folder': subdir,
                    'Best Alpha': best_alpha,
                    'Min MRE': min_mre
                })

    # Save results to a CSV file
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

scheduling_order_folder_path= '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/final_scheduling_annotations_allrows_astar'  # Extracted folder path
alpha_folder_path = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/coordinates_and_hop/new_alpha_tuning/output'  # Path to the folder with reordered files
output_csv_file_path = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/coordinates_and_hop/new_alpha_tuning/best_alpha_results_a.csv'  # Output CSV file path

# Find the best alpha with the smallest MRE
find_best_alpha(alpha_folder_path, scheduling_order_folder_path, output_csv_file_path)

