import pandas as pd
import cv2
import json
import os
import ast
import numpy as np

# Load CSV file
data = pd.read_csv('/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/segmentation_v2/val.csv')

# Specify directory to save images
directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/segmentation2_data/single_val_seg_annotation'
if not os.path.exists(directory):
   os.makedirs(directory)

# Check if required columns exist
required_columns = ['region_shape_attributes', 'region_attributes', 'filename']
if not all(column in data.columns for column in required_columns):
   raise KeyError("One or more required columns are missing in the CSV file")

# Get list of all image files in the directory
image_files = os.listdir('/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/val')

# Define a function to safely parse a string into a dictionary
def safe_literal_eval(s):
  try:
      if isinstance(s, str):
          s = s.replace("'", '"')
          return ast.literal_eval(s)
      elif isinstance(s, dict):
          return s
      else:
          return None
  except ValueError:
      return None

# Initialize a counter for each new image
counter =  0

# Iterate over the images
for image_file in image_files:
    # Reset the counter for each new image
    counter =  0

    # Construct full path to image file
    image_path = os.path.join('/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/val', image_file)

    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"Image file {image_file} does not exist")
        continue

    # Load image
    image = cv2.imread(image_path)

    # Check if image was loaded successfully
    if image is None:
        print(f"Failed to load image {image_file}")
        continue

    # Convert image to RGB if it's not
    if len(image.shape) ==  2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Get image dimensions
    image_height, image_width, _ = image.shape

    # Convert 'region_shape_attributes' and 'region_attributes' columns into dictionaries
    data['region_shape_attributes'] = data['region_shape_attributes'].apply(safe_literal_eval)
    data['region_attributes'] = data['region_attributes'].apply(safe_literal_eval)

    # Get the data corresponding to the current image file
    image_data = data[data['filename'] == image_file]

    # Iterate over the polygons for the current image
    for _, row in image_data.iterrows():
        # Get region_shape_attributes directly as it's already a dictionary
        region_shape_attributes = row['region_shape_attributes']

        if region_shape_attributes is not None and 'name' in region_shape_attributes and region_shape_attributes['name'] == 'polygon':
            points = np.column_stack((region_shape_attributes['all_points_x'], region_shape_attributes['all_points_y']))
            points = np.int32(points)

            # Check if polygon is within image dimensions
            if not all(0 <= point[0] < image_width for point in points) or not all(0 <= point[1] < image_height for point in points):
                print(f"Polygon {region_shape_attributes} is outside image dimensions")
                continue

            # Get the correct order of points
            hull = cv2.convexHull(points)

            # Extract picture
            mask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.fillPoly(mask, [hull],  255)
            picture = cv2.bitwise_and(image, image, mask=mask)

            # Check if picture data is valid
            if picture is None:
                print(f"No picture data for polygon {row}")
                continue

            # Save picture to file
            file_path = os.path.join(directory, f'picture_{image_file}_{counter}.png')
            print(f"Saving picture to {file_path}")
            success = cv2.imwrite(file_path, picture)

            # Check if picture was saved successfully
            if not success:
                print(f"Failed to save picture to {file_path}")

            # Get annotations directly as it's already a dictionary
            annotations = row['region_attributes']

            if annotations is not None:
                # Save annotations to file
                annotation_file_path = os.path.join(directory, f'annotations_{image_file}_{counter}.json')
                with open(annotation_file_path, 'w') as file:
                    json.dump(annotations, file)

            # Increment the counter after saving each annotation
            counter +=  1

