import numpy as np

a = np.array([[1,2,3], [4,5,6]])
print("array a: ", a)
print("shape of a:", a.shape)

x = np.atleast_2d(a)
print("array x:", x)
print("shape of x: ", x.shape)

arr = np.array([1, 2, 3, 4], ndmin=5)

print(arr)
print('shape of array :', arr.shape)

print(np.eye(3))