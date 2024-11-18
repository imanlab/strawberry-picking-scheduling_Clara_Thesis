import csv
import matplotlib.pyplot as plt

def count_non_increasing(sequence):
    non_increasing_count = 0
    prev_value = sequence[0]

    for value in sequence[1:]:
        if value < prev_value:
            non_increasing_count += 1
        prev_value = value

    return non_increasing_count

# Read CSV file and extract the last column
file_path = "/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/grading_trajectories/38_results_output.csv"  # Update with your CSV file path

with open(file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header row
    sequence_column_index = -1  # Assuming the last column contains the float values
    sequences = []

    for row in reader:
        sequence_str = row[sequence_column_index].strip()
        print(sequence_str)
        if sequence_str:
            sequence = list(map(float, sequence_str.split(',')))
            sequences.append(sequence)


plt.plot(sequences)

plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Input Sequences")
#splt.legend()
plt.show()

# Count non-increasing sequences for each sequence
non_increasing_count = count_non_increasing(sequences)
print(f"Number of times values are not in increasing order:", non_increasing_count)


