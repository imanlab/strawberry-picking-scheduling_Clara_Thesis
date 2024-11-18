import json
import os

# Specify the path to the input JSON file
input_json_path = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/segmentation_v2/test_with_scheduling_with_z_with_centers.json'

# Specify the directory to save the individual JSON files
output_directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/segmentation_v2/single_test_scheduling_data_annotations_with_coordinates'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Load the JSON data
with open(input_json_path, 'r') as file:
    data = json.load(file)

# Iterate over each image in the data
for image_name, image_data in data.items():
    # Iterate over each strawberry in the image
    for i, strawberry in enumerate(image_data['regions']):
        # Create a filename for the strawberry JSON file
        strawberry_filename = f"{image_name.split('.')[0]}_strawberry_{i}.json"
        strawberry_filepath = os.path.join(output_directory, strawberry_filename)
        
        # Write the strawberry data to a new JSON file
        with open(strawberry_filepath, 'w') as file:
            json.dump(strawberry, file, indent=4)

print("Strawberry JSON files have been created.")

