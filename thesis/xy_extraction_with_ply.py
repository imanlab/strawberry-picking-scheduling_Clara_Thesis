import os
import json
import numpy as np
import pandas as pd
import tempfile
import open3d as o3d
from plyfile import PlyData
import csv
import ast

# Define the directory containing the PLY files
ply_dir = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train_ply' # Replace with your directory

# Define the path to the JSON file
json_file = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/segmentation2_data/train.json' # Replace with your file

# Define the directory where the CSV files will be saved
xy_coordinate_dir = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train/xy_seg' # Replace with your directory

z_coordinate_dir = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train/z_extraction' # Replace with your directory

# Define the directory where the .ply files will be saved
ply_seg_dir = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train/ply_seg' # Replace with your directory

def extract_coordinates_from_json(json_file):
    with open(json_file, 'r') as f:
        json_data = json.load(f) # Load the JSON file

        coords_dict = {}

        for key, image_data in json_data.items(): # Iterate over the items in the dictionary
            filename = key.split(".")[0] # Remove the size part from the filename
            regions = image_data['regions']

            for i, region in enumerate(regions):
                coords = region['shape_attributes']

                if 'all_points_x' in coords and 'all_points_y' in coords:
                    x_coords = coords['all_points_x']
                    y_coords = coords['all_points_y']

                    # Save the coordinates to a CSV file
                    csv_filename = os.path.join(xy_coordinate_dir, f'{filename}_{i}.csv') # Add the index to the filename
                    df = pd.DataFrame({'X': x_coords, 'Y': y_coords})
                    df.to_csv(csv_filename, index=False)

                    if filename not in coords_dict:
                        coords_dict[filename] = []
                    coords_dict[filename].append((x_coords, y_coords))

    return coords_dict


    
def main():
    coords_dict = extract_coordinates_from_json(json_file)


if __name__ == "__main__":
    main()

