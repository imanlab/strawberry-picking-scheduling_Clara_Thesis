import os
import numpy as np
import json
import cv2

# Load the JSON file
json_file = "/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/segmentation_v2/train.json" # Replace with your .json file path
with open(json_file, 'r') as f:
 data = json.load(f)

# Get the list of all .npy files in the directory
npy_folder = "/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train_npy/" # Replace with your .npy folder path
npy_files = [f for f in os.listdir(npy_folder) if f.endswith('.npy')]
print(npy_files)

# Create a new folder for the new .npy files
new_folder = "/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/npy_seg" # Replace with your new .npy folder path
os.makedirs(new_folder, exist_ok=True)

# Iterate over each .npy file
for npy_file in npy_files:
 # Remove the '.npy' and '_rdepth' extensions from the .npy file name to get the image name
 img_name = os.path.splitext(npy_file)[0].replace('_rdepth', '') +'_rgb.png'


 # Check if the image name exists in the JSON data
 if img_name in data:
     # Extract the coordinates of the polygons from the JSON data
     polygons = data[img_name]['regions']

     # Load the .npy file
     matrix = np.load(os.path.join(npy_folder, npy_file))

     # Iterate over each polygon
     for i, polygon in enumerate(polygons):
       try:
         # Extract the row and column coordinates from the polygon
         rows = np.array(polygon['shape_attributes']['all_points_y'])
         cols = np.array(polygon['shape_attributes']['all_points_x'])

         # Create a mask for the polygon
         mask = np.zeros_like(matrix)
         points = np.stack([cols, rows], axis=-1)
         cv2.fillPoly(mask, [points], 1)

         # Crop the matrix data based on the row and column coordinates
         cropped_matrix = matrix * mask

         # Save the cropped data to a new .npy file in the new folder
         np.save(os.path.join(new_folder, f'npy_cropped_{npy_file}_{i}.npy'), cropped_matrix)
       except KeyError:
          print(f"KeyError occurred at iteration {i} for image {img_name}")
         
         
 else:
     print(f"No data found for {img_name}")

