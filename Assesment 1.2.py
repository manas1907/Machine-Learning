import numpy as np

# 5.1: Create NumPy array 'arr' with values from 1 to 10
arr = np.arange(1, 11)

# 5.2: Create another NumPy array 'arr2' with values from 11 to 20
arr2 = np.arange(11, 21)

# 5.3: Perform operations and print the results
addition_result = arr + arr2
subtraction_result = arr - arr2
multiplication_result = arr * arr2
division_result = arr / arr2

print("Array 'arr':", arr)
print("Array 'arr2':", arr2)

print("Addition Result:", addition_result)
print("Subtraction Result:", subtraction_result)
print("Multiplication Result:", multiplication_result)
print("Division Result:", division_result)

# 6.1: Reshape 'arr' into a 2x5 matrix [rows x columns]
arr_2x5 = arr.reshape(2, 5)

# 6.2: Transpose the matrix obtained in the previous step [5x2]
transposed_matrix = arr_2x5.T

# 6.3: Flatten the transposed matrix into a 1D array
flattened_array = transposed_matrix.flatten()

# 6.4: Stack 'arr' and 'arr2' vertically
stacked_result = np.vstack((arr, arr2))

# Print the results
print("\nReshaped 'arr' (2x5 matrix):")
print(arr_2x5)

print("\nTransposed Matrix:")
print(transposed_matrix)

print("\nFlattened Array:")
print(flattened_array)

print("\nVertically Stacked Result:")
print(stacked_result)

# 7.1: Calculate the mean, median, and standard deviation of 'arr'
mean_value = np.mean(arr)
median_value = np.median(arr)
std_deviation_value = np.std(arr)

# 7.2: Find the maximum and minimum values in 'arr'
max_value = np.max(arr)
min_value = np.min(arr)

# 7.3: Normalize 'arr' (subtract the mean and divide by the standard deviation)
normalized_arr = (arr - mean_value) / std_deviation_value

# Print the results
print("\nMean of 'arr':", mean_value)
print("Median of 'arr':", median_value)
print("Standard Deviation of 'arr':", std_deviation_value)
print("Maximum value in 'arr':", max_value)
print("Minimum value in 'arr':", min_value)

print("\nNormalized 'arr':", normalized_arr)

# 8.1: Create a boolean array 'bool_arr' for elements in 'arr' greater than 5
bool_arr = arr > 5

# 8.2: Use 'bool_arr' to extract the elements from 'arr' that are greater than 5
filtered_elements = arr[bool_arr]

# Print the results
print("\nBoolean array for elements greater than 5:")
print(bool_arr)
print("\nElements in 'arr' greater than 5:")
print(filtered_elements)

# 9.1: Generate a 3x3 matrix with random values between 0 and 1
random_matrix = np.random.rand(3, 3)

# 9.2: Create an array of 10 random integers between 1 and 100
random_integers = np.random.randint(1, 101, 10)

# 9.3: Shuffle the elements of 'arr' randomly
np.random.shuffle(arr)

# Print the results
print("\nRandom 3x3 Matrix:")
print(random_matrix)

print("\nArray of 10 Random Integers between 1 and 100:")
print(random_integers)

print("\nShuffled 'arr':", arr)

# 10.1: Apply the square root function to all elements in 'arr'
sqrt_arr = np.sqrt(arr)

# 10.2: Use the exponential function to calculate ex for each element in 'arr'
exp_arr = np.exp(arr)

# Print the results
print("\nSquare root of 'arr':", sqrt_arr)
print("\nExponential function applied to 'arr':", exp_arr)

# 11.1: Create a 3x3 matrix 'mat_a' with random values
mat_a = np.random.rand(3, 3)

# 11.2: Create a 3x1 matrix 'vec_b' with random values
vec_b = np.random.rand(3, 1)

# 11.3: Multiply 'mat_a' and 'vec_b' using the dot product
result = np.dot(mat_a, vec_b)

# Print the results
print("\nMatrix 'mat_a':")
print(mat_a)

print("\nMatrix 'vec_b':")
print(vec_b)

print("\nResult of mat_a * vec_b (dot product):")
print(result)

#12.1 Create a 2D array 'matrix' with values from 1 to 9
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

#12.2 Calculate the mean of each row
row_means = np.mean(matrix, axis=1, keepdims=True)

# Subtract the mean of each row from each element in that row
normalized_matrix = matrix - row_means

print("\nOriginal Matrix:")
print(matrix)

print("\nMatrix after subtracting row means:")
print(normalized_matrix)
