import pandas as pd
import os

# Define the folder path where your original CSV files are located
original_folder_path = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/test/optimal_fin_astar'
# Define the folder path where you want to save the modified CSV files
new_folder_path = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/test/final_scheduling_annotations_allrows_astar'

# Create the new folder if it doesn't exist
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

# Loop through each file in the original folder
for filename in os.listdir(original_folder_path):
    if filename.endswith('.csv'):
        # Construct the full file path for the original file
        original_file_path = os.path.join(original_folder_path, filename)
        
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(original_file_path)
        
        # Calculate the new column
        # Assuming 'Total Distance' and 'kendall_tau' are the column names for distance and kendall_tau values
        df['Scheduling Order'] = df['Total Distance'] + (df['kendall_tau'] * 100)
        
        # Sort the DataFrame by the new column in increasing order
        df = df.sort_values(by='Scheduling Order')
        
        # Construct the new file path in the new folder
        new_file_path = os.path.join(new_folder_path, filename)
        
        # Save the modified DataFrame to the new CSV file
        df.to_csv(new_file_path, index=False)

print("All CSV files have been processed and saved in the new folder.")
