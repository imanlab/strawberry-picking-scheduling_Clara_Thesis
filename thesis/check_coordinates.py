from plyfile import PlyData
import pandas as pd
import os

def print_and_save_coordinates(ply_file, output_dir):
  plydata = PlyData.read(ply_file)
  x_coords = plydata['vertex']['x']
  y_coords = plydata['vertex']['y']
  
  # Create a DataFrame from the coordinates
  df = pd.DataFrame({'X': x_coords, 'Y': y_coords})
  
   # Save the DataFrame to a CSV file
  df.to_csv(output_dir, index=False)
  
  print(f"Saved coordinates to {output_dir}")

print_and_save_coordinates('/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train_ply/strawberry_riseholme_lincoln_tbd_0_pc.ply', '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/check_coordinates.csv')


