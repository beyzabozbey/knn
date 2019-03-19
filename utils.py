import numpy as np

###########################################################
# DO NOT EDIT ANYTHING ABOVE
###########################################################

def l2_distance(point1, point2):
	#########################################################################
	# Implement your l2 distance below
	#########################################################################
	distance = np.linalg.norm(point1 - point2)

	return distance

def l1_distance(point1, point2):
	#########################################################################
	# Implement your l1 distance below
	#########################################################################
	dis_array = np.absolute(point1 - point2)
	distance = 0

	for i in range(len(dis_array)):
		distance += dis_array[i]

	return distance

def linf_distance(point1, point2):
	#########################################################################
	# Implement your l_inf distance below
	#########################################################################
	dis_array = np.absolute(point1 - point2)
	distance = dis_array[0]

	for i in range(len(dis_array)):
		if distance < dis_array[i]:
			distance = dis_array[i]
	
	return distance

def inner_distance(point1, point2):
	#########################################################################
	# Implement your inner product distance below
	#########################################################################
	magnitude1 = np.dot(point1, point1) ** 0.5
	magnitude2 = np.dot(point2, point2) ** 0.5

	return -(np.dot(point1, point2) / (magnitude1 * magnitude2))
