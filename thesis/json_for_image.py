import os
import json


with open('/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/segmentation_v2/train.json') as f:
    data = json.load(f)

annotation_directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/segmentation_v2/json_for_image'

image_directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train'

# Check if the directory exists, if not, create it
if not os.path.exists(annotation_directory):
    os.makedirs(annotation_directory)

# Iterate over the items in the data, extract the image filename and its annotations, and save them to separate files
for image_filename, image_data in data.items():
    # Extract the annotations
    annotations = image_data['regions']

    # Save the annotations to a separate JSON file
    annotation_filename = f"{image_filename}_annotations.json"
    annotation_path = os.path.join(annotation_directory, annotation_filename)

    with open(annotation_path, 'w') as f:
        json.dump(annotations, f)


