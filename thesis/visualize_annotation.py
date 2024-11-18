import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

# Load the annotations from the CSV file
file_path = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/segmentation_v2/train.csv'  # Replace with your file path
annotations = pd.read_csv(file_path)

# Function to draw polygons on an image
def draw_polygons(image_path, annotations):
    # Open the image
    image = Image.open(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    for index, row in annotations.iterrows():
        region_shape_attributes = json.loads(row['region_shape_attributes'])
        region_attributes = json.loads(row['region_attributes'])
        
        all_points_x = region_shape_attributes['all_points_x']
        all_points_y = region_shape_attributes['all_points_y']
        
        polygon = list(zip(all_points_x, all_points_y))
        bright_yellow = '#FFFF00'  # Bright highlighter yellow
        polygon_patch = patches.Polygon(polygon, closed=True, edgecolor=bright_yellow, facecolor='none', linewidth=2)
        plt.gca().add_patch(polygon_patch)
        
        # Annotate the polygon with its attributes
        centroid_x = sum(all_points_x) / len(all_points_x)
        centroid_y = sum(all_points_y) / len(all_points_y)
        annotation_text = f" {region_attributes['Hardness of Picking']}"
        plt.text(centroid_x, centroid_y, annotation_text, color='white', fontsize=23, ha='center')

    plt.axis('off')
    plt.show()

# Extract unique image paths
unique_images = annotations['filename'].unique()

# Assuming the images are stored locally in 'images/' directory
image_base_path = "/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train"  # Replace with the actual directory path

# Display annotations for each unique image
for image_name in unique_images:
    image_path = os.path.join(image_base_path, image_name)
    if os.path.exists(image_path):
        image_annotations = annotations[annotations['filename'] == image_name]
        draw_polygons(image_path, image_annotations)
    else:
        print(f"Image {image_name} not found in the directory {image_base_path}")
'  # Replace with your file path'
annotations = pd.read_csv(file_path)


# Extract unique image paths
unique_images = annotations['filename'].unique()

# Assuming the images are stored locally in 'images/' directory
image_base_path = "path/to/images/"  # Replace with the actual directory path

# Display annotations for each unique image
for image_name in unique_images:
    image_path = os.path.join(image_base_path, image_name)
    if os.path.exists(image_path):
        image_annotations = annotations[annotations['filename'] == image_name]
        draw_polygons(image_path, image_annotations)
    else:
        print(f"Image {image_name} not found in the directory {image_base_path}")

