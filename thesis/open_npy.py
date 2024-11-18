import numpy as np

# Step 1: Load the .npy file
data = np.load('/home/clara/Thesis/strawberry-pp-w-r-dataset-master/1/1/001/strawberry_dyson_lincoln_tbd__001_1_rdepth.npy')

# Step 2: Save the numpy array as a CSV file
np.savetxt('output7.csv', data, delimiter=',')
print(data.shape)


'''
# Step 1: Load the .npy file
data = np.load('/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train_npy/strawberry_riseholme_lincoln_tbd_30_rdepth.npy')

# Step 2: Save the numpy array as a CSV file
np.savetxt('output.csv', data, delimiter=',')
print(data.shape)
'''
