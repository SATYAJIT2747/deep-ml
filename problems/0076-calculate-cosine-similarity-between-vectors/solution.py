import numpy as np

def cosine_similarity(v1, v2):
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	# Implement your code here
	v1_mag = np.linalg.norm(v1)
	v2_mag = np.linalg.norm(v2)
	if(v1_mag == 0 or v2_mag == 0):
		return -1
	elif (v1.shape != v2.shape):
		return  -1
	prod = v1@v2
	return float(prod/(v1_mag*v2_mag))
