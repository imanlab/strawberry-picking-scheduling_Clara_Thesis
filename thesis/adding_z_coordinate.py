import os
import json
import csv

# Load the JSON file
with open('/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/segmentation2_data/train_with_scheduling.json', 'r') as file:
    data = json.load(file)
    #print(data)

# Directory containing the CSV files
csv_directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train/z_poligons'

# Process each CSV file
for filename in os.listdir(csv_directory):
    if filename.endswith('.csv'):
        # Construct the full file path
        file_path = os.path.join(csv_directory, filename)

        # Extract image number and strawberry number from the filename
        base_filename, _ = os.path.splitext(filename)
        print(base_filename)
        parts = base_filename.split('_')
        image_number = parts[-3] # Assuming the image number is the second to last part of the filename
        strawberry_number = int(parts[-1].split('.')[0]) # Assuming the strawberry number is the last part before the extension
        
        print(strawberry_number)
        
        # Construct the JSON key for the image
        json_key = f"strawberry_riseholme_lincoln_tbd_{image_number}_rgb.png"


        # Load the CSV file and skip the header row
        with open(os.path.join(csv_directory, filename), 'r') as csvfile:
            reader = csv.reader(csvfile, skipinitialspace=True, delimiter=',') 
            next(reader) # Skip the header row
            z_coordinates = []
            for row in reader:
                if len(row) >= 3: 
                    try:
                        # Assuming each row contains X, Y, Z coordinates
                        z_coordinates.append(int(float(row[2])))# Append Z coordinate to the list
                    except ValueError:
                        print(f"Error parsing row: {row}")

            # Update the Z coordinates for the corresponding strawberry annotation
            if json_key in data and strawberry_number < len(data[json_key]['regions']):
                data[json_key]['regions'][strawberry_number]['shape_attributes']['all_points_z'] = z_coordinates


# Save the updated JSON file
output_json_path = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/train_with_scheduling_with_z.json'  # Replace with your desired output JSON file path

with open(output_json_path, 'w') as file:
    json.dump(data, file, indent=4)

