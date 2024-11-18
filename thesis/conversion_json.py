import csv
import json

def csv_to_json(csvFilePath, jsonFilePath):
    # Create a dictionary to hold the dictionaries
    data = {}

    # Open a CSV reader
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.reader(csvf)
        headers = next(csvReader) # Get the headers from the first row

        for i, rows in enumerate(csvReader, start=1):
            if len(rows) != len(headers):
                print(f"Skipping row {i}: Unexpected number of columns")
                continue

            try:
                # Convert each row into a dictionary
                row_dict = dict(zip(headers, rows))

                # Parse the JSON strings into Python data structures
                row_dict['region_shape_attributes'] = json.loads(row_dict['region_shape_attributes'])
                row_dict['region_attributes'] = json.loads(row_dict['region_attributes'])

                # Use the filename as the key and the row as the value
                filename = row_dict['filename']
                del row_dict['filename']
                data[filename] = row_dict
            except json.JSONDecodeError as e:
                print(f"Error at row {i}: {e}")
                continue

    # Open a JSON writer and use the json.dumps() function to dump data
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, ensure_ascii=False))


# Decide the two file paths according to your computer system
csvFilePath = r'/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/segmentation2_data/train.csv'
jsonFilePath = r'/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/segmentation2_data/train.json'

# Call the csv_to_json function
csv_to_json(csvFilePath, jsonFilePath)

