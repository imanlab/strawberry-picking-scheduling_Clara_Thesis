import pandas as pd
import glob
import os

def add_z_coordinates():
   file_path = "/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/val/xy_centers"
   files = glob.glob(file_path + "/*rgb.png_*csv")
   print(files)
   for file in files:
       # Read the CSV file
       df = pd.read_csv(file)
       #print(file)

       # Get the base name of the file
       base_name = os.path.splitext(os.path.basename(file))[0]
       base_name = base_name.replace('centroid_', '')
       base_name = base_name.replace('_rgb.png_', '_rdepth.npy_')
       print(base_name)

       # Get the index from the file name
       #index = base_name.split('_')[-1]

       # Construct the z-coordinate file path
       z_file = f"/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/val/z_extraction/avg_{base_name}.txt"
       print(z_file)

       # Try to read the z-coordinate file
       try:
          with open(z_file, 'r') as f:
              z = float(f.read())
              print(z)
              df.insert(2, z, 'z')

              # Write the updated DataFrame back to the CSV file
              df.to_csv(file, index=False)
       except FileNotFoundError:
          df.insert(2, None, 'z')
          df.to_csv(file, index=False)

if __name__ == '__main__':
  add_z_coordinates()

