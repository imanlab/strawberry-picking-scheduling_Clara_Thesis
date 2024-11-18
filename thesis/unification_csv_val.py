import csv
import os
from collections import defaultdict

#Take the csv files with the xy centers and add the z coordinate (check if it's true):it's not

def combine_csvs(directory, output_directory):
    # Get a list of all CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
   
    
    # Group the files by picture_number
    grouped_files = defaultdict(list)
    for csv_file in csv_files:
        picture_number = int(csv_file.split('_')[5])
        print(picture_number)
        grouped_files[picture_number].append(csv_file)

    # Process each group
    for picture_number, files in grouped_files.items():
        # Determine the maximum polygon_index from the file names
        max_polygon_index = max(int(file.split('_')[-1].split('.')[0]) for file in files) + 1
        
        # Initialize the rows list with None values up to the maximum polygon_index
        rows = [None] * (max_polygon_index + 1)

        # Loop over each CSV file in the group
        for csv_file in files:
            file_path = os.path.join(directory, csv_file)
            with open(file_path, 'r') as file:
                # Use csv.reader to read the rows from the file
                reader = csv.reader(file)
                for row in reader:
                    # Calculate the polygon_index from the file name
                    polygon_index = int(csv_file.split('_')[-1].split('.')[0]) + 1
                    # Check if the corresponding element in the rows list is None
                    if rows[polygon_index] is None:
                        # If it is, replace it with an empty list
                        rows[polygon_index] = []
                    # Replace the corresponding element in the rows list
                    rows[polygon_index] = row

        # Open the output file in write mode
        output_file = os.path.join(output_directory, f'{picture_number}.csv')
        with open(output_file, 'w', newline='') as file:
            # Use csv.writer to write the rows to the file
            writer = csv.writer(file)
            # Filter out None values before writing to the file
            writer.writerows(row for row in rows if row is not None)

directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/val/xy_centers'
output_directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/val/coordinates_for_images'
combine_csvs(directory, output_directory)

