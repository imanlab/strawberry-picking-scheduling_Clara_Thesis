import os
import json
import pandas as pd
import glob

# Directory containing the JSON files
json_base_directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/segmentation_v2/single_test_scheduling_data_annotations_with_coordinates'  # Change to your JSON files directory

# Output directory for the CSV files
output_csv_directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/test/coordinates_for_hop'  # Change to your desired output directory

# Ensure the output directory exists
os.makedirs(output_csv_directory, exist_ok=True)

# Get a list of all JSON files in the directory
json_files = glob.glob(os.path.join(json_base_directory, "*.json"))

# Dictionary to store data per image
data_per_image = {}

# Process each JSON file
for json_file in json_files:
    # Extract image number and strawberry number from the filename
    base_filename = os.path.basename(json_file)
    parts = base_filename.split('_')
    image_number = parts[4]
    strawberry_number = int(parts[-1].split('.')[0])  # Extract strawberry number
    print(strawberry_number)
    # Read the JSON file
    with open(json_file, 'r') as file:
        json_data = json.load(file)
        
    # Check if the required attributes exist in the JSON data
    if "shape_attributes" in json_data and "center_x" in json_data["shape_attributes"] and "center_y" in json_data["shape_attributes"] and "center_z" in json_data["shape_attributes"]:
        center_x = json_data["shape_attributes"]["center_x"]
        center_y = json_data["shape_attributes"]["center_y"]
        center_z = json_data["shape_attributes"]["center_z"]
    else:
        center_x = center_y = center_z = ""

    # Extract the hardness of picking value
    if "region_attributes" in json_data and "Hardness of Picking" in json_data["region_attributes"]:
        hardness_of_picking = json_data["region_attributes"]["Hardness of Picking"]
    else:
        hardness_of_picking = ""
    
    # Prepare the data for this image
    if image_number not in data_per_image:
        data_per_image[image_number] = {}
    data_per_image[image_number][strawberry_number] = [center_x, center_y, center_z, hardness_of_picking]

# Save the results to CSV files per image
for image_number, data in data_per_image.items():
    # Sort the data by strawberry number
    sorted_data = [data[key] for key in sorted(data.keys())]
    
    # Create a DataFrame for the current image
    df = pd.DataFrame(sorted_data, columns=['X', 'Y', 'Z', 'Hardness of Picking'])
    
    # Save the DataFrame to a CSV file
    output_csv_file = os.path.join(output_csv_directory, f'image_{image_number}_data.csv')
    df.to_csv(output_csv_file, index=False)

print("Processing complete. CSV files have been generated.")
