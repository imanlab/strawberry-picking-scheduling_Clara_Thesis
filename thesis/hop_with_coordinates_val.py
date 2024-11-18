import os
import pandas as pd
import json

# Define paths for the directories containing the CSV and JSON files and the output directory
csv_dir = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/val/coordinates_for_images'
json_dir = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/segmentation_v2/single_val_seg_annotation'
output_dir = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/val/coordinates_and_hop'

# Function to read JSON file and extract hardness of picking
def get_hardness_from_json(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data.get('Hardness of Picking', None)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)


# Process each CSV file
for csv_filename in os.listdir(csv_dir):
    if csv_filename.endswith('.csv'):
        csv_file_path = os.path.join(csv_dir, csv_filename)
        csv_data = pd.read_csv(csv_file_path, header=None)

        # Add the Hardness of Picking to each row
        hardness_values = []
        base_filename = os.path.splitext(csv_filename)[0]
        for index, row in csv_data.iterrows():
            json_filename = f"annotations_strawberry_riseholme_lincoln_tbd_{base_filename}_rgb.png_{index}.json"
            json_file_path = os.path.join(json_dir, json_filename)
            if os.path.exists(json_file_path):
                hardness_value = get_hardness_from_json(json_file_path)
                hardness_values.append(hardness_value)
            else:
                hardness_values.append(None)

        # Append the hardness values as a new column without a header
        csv_data[len(csv_data.columns)] = hardness_values

        # Save the updated CSV to the output directory without a header
        output_csv_path = os.path.join(output_dir, csv_filename)
        csv_data.to_csv(output_csv_path, index=False, header=False)

print(f"Processed files have been saved to {output_dir}")

