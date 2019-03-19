import argparse
import json
import numpy as np
from sklearn.metrics import accuracy_score

import utils
from classifier import KNN

###########################################################
# DO NOT EDIT ANYTHING ABOVE
###########################################################


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('training_data_file', type=str, default='')
	parser.add_argument('test_data_file', type=str, default='')
	parser.add_argument('--num_fold', type=int, default=10)
	parser.add_argument('--random_seed', type=int, default=42)

	return parser.parse_args()

def prepare_data(training_data_file, test_data_file, random_seed, num_fold):
	'''
	-- Inputs
	training_data_file: path to training data
	test_data_file:		path to test data
	random_seed:		random seed for np.random.shuffle
	num_fold:			number of fold
	-- Outputs
	features_folds:		list of features per fold, len(features_fold)=num_fold
	labels_folds:		list of labels per fold, len(labels_fold)=num_fold
	features_test:		features for test, N x M
	labels_test:		labels for test, N
	'''

	#########################################################################
	# Implement your data splitting below
	#########################################################################

	training_file = open(training_data_file, "r")
	training_data = json.load(training_file)
	labels = training_data["labels"]
	features = training_data["features"]
	training_file.close()

	test_file = open(test_data_file, "r")
	test_data = json.load(test_file)
	features_test = test_data["features"]
	labels_test = test_data["labels"]
	test_file.close()

	# normalization

	means = []
	for i in range(len(features[0])):
		sum = 0
		for j in range(len(labels)):
			sum += features[j][i]
		mean = sum / float(len(labels))
		means.append(mean)
		for j in range(len(labels)):
			features[j][i] -= mean

	for i in range(len(features_test[0])):
		for j in range(len(labels_test)):
			features_test[j][i] -= means[i]
	
	# shuffling

	labels = np.array(labels)
	np.random.seed(random_seed)
	np.random.shuffle(labels)
	features = np.array(features)
	np.random.seed(random_seed)
	np.random.shuffle(features)

	# splitting

	features_folds = []
	labels_folds = []
	for i in range(num_fold):
		features_folds.append([])
		labels_folds.append([])

	for i in range(len(labels)):
		features_folds[i % num_fold].append(features[i].tolist())
		labels_folds[i % num_fold].append(labels[i])

	return features_folds, labels_folds, features_test, labels_test



if __name__ == '__main__':
	# settings
	args = parse_args()
	ks = [1,2,5]
	dist_funcs = [utils.l1_distance,
					utils.l2_distance,
					utils.linf_distance,
					utils.inner_distance]

	# data
	features_folds, labels_folds, \
			features_test, labels_test = prepare_data(args.training_data_file,
														args.test_data_file,
														args.random_seed,
														args.num_fold)
						

	# knn
	cv_accuracies = np.zeros((len(ks), len(dist_funcs)))
	test_accuracies = np.zeros((len(ks), len(dist_funcs)))
	for idx_k, k in enumerate(ks):
		for idx_func, dist_func in enumerate(dist_funcs):
			knn = KNN(k=k, dist_func=dist_func)
			
			# cross-validation
			accuracy_folds = np.zeros(args.num_fold)
			for idx_fold in range(args.num_fold):
				# data
				features_train = np.concatenate([x for i,x in enumerate(features_folds) if i!=idx_fold])
				labels_train = np.concatenate([x for i,x in enumerate(labels_folds) if i!=idx_fold])
				features_val = features_folds[idx_fold]
				labels_val = labels_folds[idx_fold]
				
				# train-val
				knn.train(features_train, labels_train)
				labels_pred_val = knn.predict(features_val)

				# eval
				accuracy_folds[idx_fold] = accuracy_score(labels_val, labels_pred_val)

			cv_accuracies[idx_k, idx_func] = np.mean(accuracy_folds)

			# train-test
			features_train = np.concatenate(features_folds)
			labels_train = np.concatenate(labels_folds)
			knn.train(features_train, labels_train)

			labels_pred_test = knn.predict(features_test)	
			test_accuracies[idx_k, idx_func] = accuracy_score(labels_test, labels_pred_test)
			
	# save
	json.dump(cv_accuracies.tolist(), open('cv_accuracies.json', 'w'))
	json.dump(test_accuracies.tolist(), open('test_accuracies.json', 'w'))

