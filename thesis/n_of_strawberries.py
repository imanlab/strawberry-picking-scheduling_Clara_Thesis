import os
import pandas as pd
from collections import defaultdict

def count_polygons_in_folder(folder):
   # Initialize a dictionary to store the count of polygons per image
   polygon_counts = {}
   polygon_counts = defaultdict(int)

   # List all files in the folder
   files = os.listdir(folder)

   # Iterate over the files
   for filename in files:
       
       # Skip if the file is not a text file
       if not filename.endswith('.txt'):
           continue

       # Extract the image name from the filename
       img_name = '_'.join(filename.split('_')[:-1])
       #print(img_name)


       # Store the count in the dictionary
       polygon_counts[img_name] += 1

   return dict(polygon_counts)
   

def save_to_csv(polygon_counts, filename):
  # Convert the dictionary to a DataFrame
  df = pd.DataFrame(list(polygon_counts.items()), columns=['Image Name', 'Number of Polygons'])
  
  # Save the DataFrame to a CSV file
  df.to_csv(filename, index=False)

# Use the function
polygon_counts = count_polygons_in_folder('/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train/z_extraction')
save_to_csv(polygon_counts, '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/output.csv')

