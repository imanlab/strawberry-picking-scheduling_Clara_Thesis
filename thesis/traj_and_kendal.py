import os
import pandas as pd
import json
from scipy.stats import kendalltau
import glob
import numpy as np

# Directory containing the CSV files
input_csv_directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train/optimal_path_astar'

# Directory containing the JSON files
json_base_directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/segmentation2_data/single_train_seg2_data_annotation'

# Output directory for the CSV files
output_csv_directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train/optimal_fin_ga'

# Get a list of all CSV files in the directory
input_csv_files = glob.glob(os.path.join(input_csv_directory, "*.csv"))

def process_row(row, image_number):
    #nodes = row['Path']
    nodes = row['Nodes'] #for ga
     
    hardness_of_picking_values = []

    # Check if nodes isa single integer and convert it to a list if so
    if isinstance(nodes, np.int64):
        nodes = [nodes]
        print('CIAO')
        
    elif isinstance(nodes, str):  # If nodes is a string, remove spaces and commas
    
        nodes = nodes.replace(",", "")
        nodes = list(map(int, nodes.split(' ')))

    json_pattern = f'annotations_strawberry_riseholme_lincoln_tbd_{image_number}_rgb.png_{{}}.json'

    for node in nodes:
        print(f'node: {node}')
        print(f'nodes: {nodes}')
        json_filename = json_pattern.format(node)
        print(json_filename)
        json_file_path = os.path.join(json_base_directory, json_filename)

        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as file:
                annotation_data = json.load(file)
            hardness_of_picking = int(annotation_data["Hardness of Picking"])
            print(hardness_of_picking)
            hardness_of_picking_values.append(hardness_of_picking)
        else:
            continue

    return hardness_of_picking_values

def compute_difference_in_indices(input_arr):
    # Step  1: Sort the input array to get the sorted array
    sorted_arr = sorted(input_arr)
    print(input_arr)
    print(sorted_arr)
    # Step  2: Find the indices of the elements in the sorted array
    sorted_indices = {element: index for index, element in enumerate(sorted_arr)}
    print(sorted_indices)
    # Step  3: Find the indices of the elements in the input array
    indices = {element: index for index, element in enumerate(input_arr)}
    print(indices)
    # Step  3: Calculate the difference in indices for each element in the input array
    differences = [abs(indices[element] - sorted_indices[element]) for element in input_arr]
   
    total = sum(differences)/ len(input_arr)
    return total
    
# Iterate over each CSV file
for input_csv_file_path in input_csv_files:
    # Extract the image number from the CSV filename
    image_number = os.path.splitext(os.path.basename(input_csv_file_path))[0].split('_')[0]
    #image_number = os.path.splitext(os.path.basename(input_csv_file_path))[0].split('_')[1] #for astar
    print(image_number)

    # Read the CSV file using pandas
    df = pd.read_csv(input_csv_file_path, delimiter=',', encoding='iso-8859-1')
    # Add the 'Hardness_of_picking' column by applying the process_row function to each row
    df['Hardness_of_picking'] = df.apply(lambda row: process_row(row, image_number), axis=1)
    
    # Calculate the absolute value of the Kendall Tau distance for each row
    df['kendall_tau'] = df['Hardness_of_picking'].apply(lambda x: compute_difference_in_indices(x))[:]

    
    # Define the output CSV file path
    output_csv_file_name = os.path.splitext(os.path.basename(input_csv_file_path))[0] + '_output.csv'
    output_csv_file_path = os.path.join(output_csv_directory, output_csv_file_name)

    # Save the updated DataFrame to the output CSV file
    df.to_csv(output_csv_file_path, index=False)


