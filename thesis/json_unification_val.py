import json
import glob
import os

#Add Scheduling Order annotations to the json file

# Load the original JSON file
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

original_data = load_json_file('/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/segmentation_v2/val.json')

def add_labels_to_original(original_data, folder_path):

    # Find all JSON files in the folder
    json_files = glob.glob(os.path.join(folder_path, '*.json'))
    
    for file in json_files:
    
        # Extract the strawberry number and image number from the filename
        # Example: annotated_annotations_strawberry_riseholme_lincoln_tbd_0_rgb.png_1.json
        # This assumes a specific naming convention for the additional JSON files
        base_name = os.path.basename(file)
        print(f'base_name: {base_name}')
        strawberry_number = base_name.split('_')[-1].split('.')[0]
        print(f'strawberry_number: {strawberry_number}')
        image_number = base_name.split('_')[-3].split('.')[0]
        print(f'image_number: {image_number}')
        
        # Load the additional JSON file
        additional_data = load_json_file(file)
        print(f'additional_data: {additional_data}')
        
        # Assuming the additional JSON contains the new labels
        new_label_value = additional_data
        
        # Find the corresponding image in the original data
        image_key = f'strawberry_riseholme_lincoln_tbd_{image_number}_rgb.png'
        if image_key in original_data:
            # Find the corresponding strawberry region
            strawberry_region = original_data[image_key]['regions'][int(strawberry_number)]
            # Check if "scheduling_order" is in additional_data and add it to the region_attributes
            if 'Scheduling Order' in additional_data:
                strawberry_region['region_attributes']['Scheduling Order'] = additional_data['Scheduling Order']
            print(additional_data['Scheduling Order'])
    return original_data
    
# Folder path containing additional JSON files
folder_path = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/single_val_scheduling_annotations'

# Add labels to the original data
updated_data = add_labels_to_original(original_data, folder_path)

def save_json_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Save the updated data to a new JSON file
save_json_file(updated_data, '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/segmentation_v2/val_with_scheduling.json')
