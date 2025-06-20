import numpy as np

a = np.array([1,2,3])
print("array a: ", a)

x = np.atleast_2d(a)
print("array x:", x)

arr = np.array([1, 2, 3, 4], ndmin=5)

print(arr)
print('shape of array :', arr.shape)