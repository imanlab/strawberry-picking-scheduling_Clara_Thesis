import os
import json
import numpy as np
import pandas as pd
import tempfile
import open3d as o3d
from plyfile import PlyData

# Define the directory containing the PLY files
ply_dir = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train_ply' # Replace with your directory

# Define the path to the JSON file
json_file = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/segmentation_v2/train.json' # Replace with your file

# Define the directory where the CSV files will be saved
xy_coordinate_dir = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/xy_seg' # Replace with your directory

z_coordinate_dir = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/z_extraction' # Replace with your directory

# Define the directory where the .ply files will be saved
ply_seg_dir = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/ply_seg' # Replace with your directory

'''
def extract_coordinates_from_ply(ply_file, polygon_index):
 plydata = PlyData.read(ply_file)
 z_coords = plydata['vertex']['z']
 df = pd.DataFrame({'Z': z_coords})
 csv_filename = os.path.join(z_coordinate_dir, f'{os.path.splitext(os.path.basename(ply_file))[0]}_{polygon_index}.csv')
 df.to_csv(csv_filename, index=False)
 print(f"Saved Z coordinates to {csv_filename}")
 return z_coords
'''

def filter_points_from_ply(ply_file, coords, scale_factor):
 point_cloud = o3d.io.read_point_cloud(ply_file)
 x_coords, y_coords = coords
 x_coords *= scale_factor
 y_coords *= scale_factor
 z_coords = extract_coordinates_from_ply(ply_file)
 coords_3d = np.stack((x_coords, y_coords, z_coords), axis=-1)
 polygon_volume = {"class": "SelectionPolygonVolume","version": "1.1","points": coords_3d}
 with tempfile.NamedTemporaryFile(suffix=".json") as temp:
   json.dump(polygon_volume, temp)
   temp.flush()
   vol = o3d.visualization.read_selection_polygon_volume(temp.name)
 ply_filename = os.path.join(ply_seg_dir, os.path.splitext(ply_file)[0] + '_cropped.ply')
 o3d.io.write_point_cloud(ply_filename, cropped_point_cloud, compressed=False, write_ascii=True)
 return cropped_point_cloud

def main():
 scale_factor = 0.0002645833 # Define scale_factor here
 for filename in os.listdir(xy_coordinate_dir):
    if filename.endswith('.csv'):
        print(f"Processing {filename}")
        filename_without_extension = os.path.splitext(filename)[0]
        ply_file = os.path.join(ply_dir, f'{filename_without_extension}.ply')
        with open(os.path.join(xy_coordinate_dir, filename), 'r') as f:
            data = pd.read_csv(f)
        x_coords = data['X'].values.astype(np.float64)
        y_coords = data['Y'].values.astype(np.float64)
        x_coords *= scale_factor
        y_coords *= scale_factor
        polygon_index = int(filename_without_extension.split('_')[-1])
        coords_3d = np.stack((x_coords, y_coords, extract_coordinates_from_ply(ply_file, polygon_index)), axis=-1)
        cropped_point_cloud = filter_points_from_ply(ply_file, coords_3d, scale_factor)
        print(cropped_point_cloud)
        ply_filename = os.path.join(ply_seg_dir, os.path.splitext(filename)[0] + '_cropped.ply')
        o3d.io.write_point_cloud(ply_filename, cropped_point_cloud)
        json_filename = os.path.join(json_dir, f'{filename_without_extension}.json')
        with open(json_filename, 'w') as f:
            json.dump(coords_3d, f)


if __name__ == "__main__":
  main()






      
       

