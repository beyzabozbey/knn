import numpy as np

###########################################################
# DO NOT EDIT ANYTHING ABOVE
###########################################################

class KNN:
	def __init__(self, k, dist_func):
		'''
		k: number of nearest neighbors considered
		dist_func: type of distance function
		'''
		self.k = k
		self.get_dist = dist_func

	def train(self, feat, label):
		'''
		feat: N x M
		label: N
		'''
		self.trn_feat = np.array(feat)
		self.trn_label = np.array(label)

	def predict(self, feat):
		'''
		-- Input
		feat: N' x M
		-- Ouput
		label: N' 
		'''
		#########################################################################
		# Implement your data splitting below
		#########################################################################

		label = []
		for i in range(len(feat)):
			dist_list = []
			for j in range(len(self.trn_feat)):
				dist_list.append(self.get_dist(np.array(feat[i]), np.array(self.trn_feat[j])))

			distances, dist_labels = (list(t) for t in zip(*sorted(zip(dist_list, self.trn_label.tolist()))))
			dist_labels = dist_labels[0:self.k]
			label.append(max(set(dist_labels), key=dist_labels.count))

		return label

