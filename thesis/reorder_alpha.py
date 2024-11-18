import os
import glob
import pandas as pd

def reorder_rows_by_total_length(folder_path):
    # Iterate over each subdirectory in the given folder
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)

        if os.path.isdir(subdir_path):
            # Iterate over each CSV file in the subdirectory
            for csv_file in glob.glob(os.path.join(subdir_path, "*.csv")):
                try:
                    df = pd.read_csv(csv_file)

                    # Sort the DataFrame by 'Total Length' in increasing order
                    df_sorted = df.sort_values(by='Total Length')

                    # Save the sorted DataFrame back to the CSV file
                    df_sorted.to_csv(csv_file, index=False)
                    print(f"Processed and sorted file: {csv_file}")
                except Exception as e:
                    print(f"Error processing file {csv_file}: {e}")

# Input folder path
input_folder_path = '//home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/coordinates_and_hop/new_alpha_tuning/output'  # Update with the actual path

# Reorder rows by increasing total length for each CSV file
reorder_rows_by_total_length(input_folder_path)

print("Reordering completed.")

