import os
from plyfile import PlyData
import pandas as pd

# Define the directory containing the PLY files
ply_dir = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train_ply' # Replace with your directory

# Define the directory where the CSV files will be saved
z_coordinate_dir = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/z_extraction' # Replace with your directory

# Iterate over each file in the directory
for filename in os.listdir(ply_dir):
 if filename.endswith('.ply'): # Only process .ply files
     print(f"Processing {filename}") # Print the current file being processed
     # Load the PLY file
     plydata = PlyData.read(os.path.join(ply_dir, filename))

     # Get the 'vertex' element
     vertices = plydata['vertex']

     # Calculate the depth of points
     depth = vertices['z']

     # Convert the depth to DataFrame
     df = pd.DataFrame({'Depth': depth})

     # Save the depth to a CSV file
     csv_filename = os.path.join(z_coordinate_dir, os.path.splitext(filename)[0] + '.csv') # Remove .ply extension and add .csv
     df.to_csv(csv_filename, index=False)
     print(f"Saved depth to {csv_filename}") # Print the CSV file saved

