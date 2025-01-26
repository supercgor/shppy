import numpy as np

a = np.array([[1,2,3], [4,5,6], [7,8,9]])
rs = np.ones((10, 3))

std_out = np.dot(rs, np.linalg.inv(a))

out = np.empty_like(rs, order = "F")


out = np.dot(np.linalg.inv(a).T, rs.T).T

print(np.allclose(std_out, out))