import os
import pandas as pd
import json
from scipy.stats import kendalltau

# Define the CSV file path and the directory containing the JSON files
csv_file_path = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/shortest_tours.csv'
json_directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/segmentation_v2/single_json'

# Read the CSV file using pandas
df = pd.read_csv(csv_file_path, delimiter=',', encoding='iso-8859-1')

# Function to process a single tour
def process_tour(nodes):
    hardness_of_picking_values = []
    for node in nodes:
        json_filename = f'annotations_strawberry_riseholme_lincoln_tbd_0_rgb.png_{node}.json'
        json_file_path = os.path.join(json_directory, json_filename)
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as file:
                annotation_data = json.load(file)
            hardness_of_picking = int(annotation_data["Hardness of Picking"])
            hardness_of_picking_values.append(hardness_of_picking)
        else:
            print(f"JSON file not found for node {node}")
    return hardness_of_picking_values

# Apply the function to each tour and compute the Kendall Tau distance
df['Hardness_of_picking'] = df['Nodes'].apply(process_tour)
df['kendall_tau'] = df['Hardness_of_picking'].apply(lambda x: kendalltau(x, sorted(x))[0])

# Save the updated DataFrame to a new CSV file
df.to_csv('updated_strawberry_hardness_of_picking_result.csv', index=False)

