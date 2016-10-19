import numpy as np
import numpy.linalg as la

# Get the ellipse stretch and rotation in order 
# to represent this 2 by 2 SPD matrix as an ellipse
def getEllipseParameters(matrix):
   eigenvalues, eigenvectors = la.eig(matrix)
   #Pick which eigenvectors which axis will be transformed to
   eigenvector_x = eigenvectors[:, 0]

   theta = getRotation(eigenvector_x)
   stretch_x = la.norm(eigenvalues[0])
   stretch_y = la.norm(eigenvalues[1])

   return [stretch_x, stretch_y, theta]

# How much is this vector rotated counter clockwise
# from the positive x-axis
def getRotation(vec):
	x_axis = np.array([1, 0])
	rotate = np.arccos((x_axis.dot(vec))/(la.norm(x_axis)*la.norm(vec)))
	if vec[1] < 0:
		rotate = 2*np.pi - rotate
	return rotate
#END

if __name__ == "__main__":
   matrix = np.array([[1, 0], [0, 2]])
   print getEllipseParameters(matrix)


