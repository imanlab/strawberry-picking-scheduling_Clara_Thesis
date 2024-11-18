import os
import numpy as np
import json
import cv2

# Load the JSON file
json_file = "/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/segmentation_v2/train.json" # Replace with your .json file path
with open(json_file, 'r') as f:
 data = json.load(f)

# Get the list of all .npy files in the directory
npy_folder = "/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train_npy" # Replace with your .npy folder path
npy_files = [f for f in os.listdir(npy_folder) if f.endswith('.npy')]


# Create a new folder for the average values
csv_folder = "/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/z_poligons" # Replace with your avg folder path
os.makedirs(csv_folder, exist_ok=True)

# Prepare an empty list to store all points with their coordinates
all_points = []

# Iterate over each .npy file
for npy_file in npy_files:
    img_name = os.path.splitext(npy_file)[0].replace('_rdepth', '') + '_rgb.png'

    if img_name in data:
        polygons = data[img_name]['regions']
        matrix = np.load(os.path.join(npy_folder, npy_file))

        for i, polygon in enumerate(polygons):
            try:
                # Extract X and Y coordinates from the JSON data
                x_coords = np.array(polygon['shape_attributes']['all_points_x'])+2
                y_coords = np.array(polygon['shape_attributes']['all_points_y'])+2

                # Adjust indices to be within the valid range
                y_coords = np.clip(y_coords, 0, matrix.shape[0] - 1)
                x_coords = np.clip(x_coords, 0, matrix.shape[1] - 1)

                # Use adjusted X and Y coordinates to index into the matrix (Z coordinates)
                z_coordinates = matrix[y_coords, x_coords]

                # Prepare each point with its coordinates
                points_with_coordinates = np.column_stack((x_coords, y_coords, z_coordinates))

                # Save the points with coordinates to a CSV file
                np.savetxt(os.path.join(csv_folder, f'strawberry_{npy_file}_{i}.csv'), points_with_coordinates, delimiter=',', header='X,Y,Z', comments='')
            except KeyError:
                print(f"KeyError occurred at iteration {i} for image {img_name}")
    else:
        print(f"No data found for {img_name}")

