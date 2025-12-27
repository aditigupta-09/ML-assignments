import numpy as np

# (a) Reverse the NumPy array
arr = np.array([1, 2, 3, 6, 4, 5])
reversed_arr = arr[::-1]
print(reversed_arr)

# (b) Flatten the NumPy array
arr1 = np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]])
flattened_arr = arr1.flatten()
print(flattened_arr)

# (c) Compare the following numpy arrays
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[1, 2], [3, 4]])
comparison_result = (arr1 == arr2).all()
print(comparison_result)

# (d) Find the most frequent value and their index(s)
x = np.array([1, 2, 3, 4, 5, 1, 2, 1, 1, 1])
y = np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3])

def find_most_frequent(array):
    values, counts = np.unique(array, return_counts=True)
    most_frequent_value = values[np.argmax(counts)]
    indices = np.where(array == most_frequent_value)
    return most_frequent_value, indices

most_freq_x, indices_x = find_most_frequent(x)
print(most_freq_x, indices_x)

most_freq_y, indices_y = find_most_frequent(y)
print(most_freq_y, indices_y)

# (e) For the array gfg
gfg = np.matrix('4 1 9; 12 3 1; 4 5 6')
print(np.sum(gfg))
print(np.sum(gfg, axis=1))
print(np.sum(gfg, axis=0))

# (f) For the matrix: n_array
n_array = np.array([[55, 25], [30, 44]])

print(np.trace(n_array))
eigen_values, eigen_vectors = np.linalg.eig(n_array)
print(eigen_values)
print(eigen_vectors)
print(np.linalg.inv(n_array))
print(np.linalg.det(n_array))

# (g) Multiply the following matrices and also find covariance
# i.
P1 = np.array([[1], [2], [3]])
q1 = np.array([[4, 5, 6, 7]])
multiplication_i = P1 @ q1
covariance_i = np.cov(P1.flatten(), q1.flatten())
print(multiplication_i)
print(covariance_i)

# ii.
P2 = np.array([[1, 2], [3, 4], [5]])
q2 = np.array([[4, 5, 1], [6, 7, 2]])

P2_adjusted = np.array([[1, 2, 3], [4, 5, 6]])
q2_T = q2.T
multiplication_ii_P_qT = P2_adjusted @ q2_T

P2_flat = P2_adjusted.flatten()
q2_flat = q2.flatten()
covariance_ii = np.cov(P2_flat, q2_flat)
print(multiplication_ii_P_qT)
print(covariance_ii)

# (h) Find inner, outer and cartesian product
x = np.array([[2, 3, 4], [3, 2, 9]])
y = np.array([[1, 5, 0], [5, 10, 3]])


inner_product = np.inner(x, y)
print(inner_product)


outer_product = np.outer(x, y)
print(outer_product)


cartesian_product = np.kron(x, y)
print(cartesian_product)
#ques-2
import numpy as np

arr = np.array([[1, -2, 3], [-4, 5, -6]])
print("array")
print(arr)
print("absolute")
print(np.abs(arr))
print("percentiles flattened 25,50,75")
print(np.percentile(arr, [25, 50, 75]))
print("percentiles each column 25,50,75")
print(np.percentile(arr, [25, 50, 75], axis=0))
print("percentiles each row 25,50,75")
print(np.percentile(arr, [25, 50, 75], axis=1))
print("mean median std flattened")
print(np.mean(arr), np.median(arr), np.std(arr))
print("mean median std each column")
print(np.mean(arr, axis=0), np.median(arr, axis=0), np.std(arr, axis=0))
print("mean median std each row")
print(np.mean(arr, axis=1), np.median(arr, axis=1), np.std(arr, axis=1))

a = np.array([-1.8, -1.6, -0.5, 0.5, 1.6, 1.8, 3.0])
print("array a")
print(a)
print("floor")
print(np.floor(a))
print("ceil")
print(np.ceil(a))
print("trunc")
print(np.trunc(a))
print("round to nearest integer")
print(np.round(a))
#Ques-3
import numpy as np

arr = np.array([10, 52, 62, 16, 16, 54, 453])
print("sorted array")
print(np.sort(arr))
print("indices of sorted array")
print(np.argsort(arr))
print("4 smallest")
print(np.sort(arr)[:4])
print("5 largest")
print(np.sort(arr)[-5:])

arr2 = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0])
print("integer elements only")
print(arr2[arr2.astype(int) == arr2])
print("float elements only")
print(arr2[arr2.astype(int) != arr2])

