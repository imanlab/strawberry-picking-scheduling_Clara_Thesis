import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

def compute_2d_polygon_centroid(vertices):
 centroid = [0, 0]
 signed_area = 0.0
 x0 = 0.0 
 y0 = 0.0 
 x1 = 0.0 
 y1 = 0.0 
 a = 0.0 

 for i in range(len(vertices)-1):
     x0 = vertices[i][0]
     y0 = vertices[i][1]
     x1 = vertices[i+1][0]
     y1 = vertices[i+1][1]
     a = x0*y1 - x1*y0
     signed_area += a
     centroid[0] += (x0 + x1)*a
     centroid[1] += (y0 + y1)*a

 x0 = vertices[-1][0]
 y0 = vertices[-1][1]
 x1 = vertices[0][0]
 y1 = vertices[0][1]
 a = x0*y1 - x1*y0
 signed_area += a
 centroid[0] += (x0 + x1)*a
 centroid[1] += (y0 + y1)*a

 signed_area *= 0.5
 centroid[0] /= (6.0*signed_area)
 centroid[1] /= (6.0*signed_area)

 return centroid

with open('/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/segmentation2_data/val.json', 'r') as f:
 annotations = json.load(f)

image_dir = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/val'

centroid_dir = os.path.join(image_dir, '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/val/xy_centers')

if not os.path.isdir(centroid_dir):
 os.mkdir(centroid_dir)

for filename in os.listdir(image_dir):
    if not filename.endswith('.png'):
        continue

    image = cv2.imread(os.path.join(image_dir, filename))

    # Check if the filename exists in the annotations dictionary
    if filename in annotations:
        if 'regions' in annotations[filename]:
            annotation_index = 0

            for annotation in annotations[filename]['regions']:
                if 'shape_attributes' in annotation and 'all_points_x' in annotation['shape_attributes']:
                    all_points_x = annotation['shape_attributes']['all_points_x']
                    all_points_y = annotation['shape_attributes']['all_points_y']
                    points = np.array([all_points_x, all_points_y]).T

                    coordinates = zip(all_points_x, all_points_y)
                    centroid = compute_2d_polygon_centroid(list(coordinates))

                    output_filename = f"centroid_{filename}_{annotation_index}.csv"
                    output_filepath = os.path.join(centroid_dir, output_filename)

                    with open(output_filepath, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(centroid)

                    annotation_index += 1
                else:
                    print(f"Error in file {filename}, annotation index {annotation_index} caused a KeyError.")
    else:
        print(f"No annotations found for {filename}")
