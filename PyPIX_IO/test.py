import numpy as np

A = np.array([[0, 1], [2, 3], [4, 5],
              [6, 7], [8, 9], [10, 11]])

print(A.shape)

b = np.zeros((10, 6, 8, 2))

b[0, :, 0, :] = A

print(b[0])

