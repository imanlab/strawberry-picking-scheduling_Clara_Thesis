'''
from scipy.stats import kendalltau

x = [0, 0, 0, 0]


print(sorted(x))

print(abs(kendalltau(x, sorted(x))[0]))


def find_min_diff(arr):
    # Step  1: Sort the input array
    sorted_arr = sorted(arr)
    print(sorted_arr)
    
    # Step  2: Sort a copy of the input array
    copy_arr = arr
    print(copy_arr)
    # Step  3: Compare the sorted arrays
    min_diff = float('inf')
    for i in range(len(sorted_arr)):
        diff = abs(sorted_arr[i] - copy_arr[i])
        print(f'sorted_arr[i]: {sorted_arr[i]}')
        print(f'copy_arr[i]: {copy_arr[i]}')
        print(diff)
        min_diff = min(min_diff, diff)
    
    return min_diff

# Example usage
input_arr = [1, 0, 2, 0, 3, 0, 1]
print(find_min_diff(input_arr))  # Output will depend on the specific differences

import numpy as np
def normalised_kendall_tau_distance(values1, values2): # (The normalised distance is between 0 and 1), the distance between equals is 0.
    """Compute the Kendall tau distance."""
    n = len(values1)
    assert len(values2) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    print(i)
    print(j)
    a = np.argsort(values1)
    b = np.argsort(values2)
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    return ndisordered / (n * (n - 1))
    
arr = [1, 0, 2, 0, 3, 0, 1]
arr_sorted = sorted(arr)    
print(normalised_kendall_tau_distance(arr, arr_sorted))

'''

def compute_difference_in_indices(input_arr):
    # Step 1: Sort the input array to get the sorted array
    sorted_arr = sorted(input_arr)
    print(input_arr)
    print(sorted_arr)
    
    # Step 2: Find the indices of the elements in the sorted array
    sorted_indices = {element: index for index, element in enumerate(sorted_arr)}
    print(sorted_indices)
    
    # Step 3: Find the indices of the elements in the input array
    indices = {element: index for index, element in enumerate(input_arr)}
    print(indices)
    
    # Step 4: Calculate the difference in indices for each element in the input array
    differences = [abs(indices[element] - sorted_indices[element]) for element in input_arr]
   
    total = sum(differences) / len(input_arr)
    return total

    
input_arr = [1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 2, 0, 0]





print(compute_difference_in_indices(input_arr))
