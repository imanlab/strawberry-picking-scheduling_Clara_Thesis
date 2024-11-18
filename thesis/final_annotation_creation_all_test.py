import pandas as pd
import json
import os
import glob
import re


def extract_index_from_filename(file_path):
    try:
        # Split the file path by slashes and take the last element to get the filename
        filename = file_path.split('/')[-1]
        # Now extract the number as before
        index_str = filename.split('_')[0]
        print(index_str)
        return int(index_str)
    except ValueError:
        return None
        
def extract_index_from_json(filename):
    # Extract the index from the filename
    try:
        index_str = filename.split('_')[-1].split('.')[0]
        return int(index_str)
    except ValueError:
        return None
        
def extract_first_number_from_filename(filename):
    try:
        # Splitting on '_' and then removing '.json'
        parts = filename[:-5].split('_')
        first_number_str = parts[-3]  # Taking the second-to-last part before removing '.json'
        print(first_number_str)
        return int(first_number_str)
    except (ValueError, IndexError):
        return None


    
def associate_strawberries(csv_path, json_folder):
    try:
        index_csv = extract_index_from_filename(csv_path)  # Extract index from CSV file path
        df = pd.read_csv(csv_path)
        print(df)
        print(f'index_csv: {index_csv}')
        #breakpoint()
        if 'Path' not in df.columns:
            raise ValueError("Column 'Path' not found in the CSV file.")

        # Check if the 'Path' column contains a list
        path_str = df['Path'].iloc[0]
        print(f'path: {path_str}')
        if isinstance(path_str, list):
            path = path_str
            scheduling_order = 0  # Directly assign 0 if 'Path' is a list
            print('ciao')
        else:
            # Convert path_str to a list of integers if it's not already a list
            path_str = str(path_str)  # Ensure it's a string before splitting
            path = [int(x) for x in path_str.split(', ')]
            scheduling_order = None  # Initialize scheduling_order
            print(f'path: {path_str}')

        # Define the destination folder for the annotated JSON files
        destination_folder = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/segmentation2_data/single_test_scheduling_data_annotations'

        condition_met = False  # Flag to check if the condition has been met for any JSON file

        for json_file_name in os.listdir(json_folder):
            if json_file_name.endswith('.json'):
                print(f'index_csv: {index_csv}')
                print(f'csv path: {csv_path}')
                print(f'path: {path_str}')
                json_file_path = os.path.join(json_folder, json_file_name)
                print(f'json_file_path: {json_file_path}')
                index_json =extract_index_from_json(json_file_name)  # Extract index from JSON file name
                print(f'index: {index_json}')

                # Extract the first number from the JSON file name
                first_number_json = extract_first_number_from_filename(json_file_name)

                if first_number_json is None or first_number_json != index_csv:
                    print(f"Warning: The first number in the JSON file name ({first_number_json}) does not match the index from the CSV file ({index_csv}). Skipping.")
                else:
                    condition_met = True

                    with open(json_file_path, 'r') as json_file:
                        original_data = json.load(json_file)

                    if index_json in path:
                         sciao = path.index(index_json)
                         original_data['Scheduling Order'] = f'{sciao}'
                         print(f'index path: {sciao}')
                    else:
                         print(f"Warning: Index {index_json} is not in the 'Path' list. Skipping.")
                         continue  # Skip this iteration if index is not in the path list


                    # Modify the output_json_path construction to use the destination folder
                    output_json_path = os.path.join(destination_folder, f'annotated_{json_file_name}')
                    with open(output_json_path, 'w') as output_file:
                        json.dump(original_data, output_file, indent=2)

                    print(f"Annotation saved to {output_json_path}")

        if not condition_met:
            print(f"Warning: No JSON file in {json_folder} corresponds to the CSV file {csv_path}. Skipping.")

    except AttributeError as e:
        print(f"Error processing file {csv_path}: {e}")
  
# Example usage
json_folder_path = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/segmentation2_data/single_test_seg2_data_annotation'
csv_folder_path = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/test/final_scheduling_annotations'

# Find all CSV files in the folder
csv_files = glob.glob(os.path.join(csv_folder_path, "*.csv"))

# Iterate over each CSV file
for csv_file in csv_files:
    try:
        associate_strawberries(csv_file, json_folder_path)
    except AttributeError as e:
        print(f"Error processing file {csv_file}: {e}")
        break


