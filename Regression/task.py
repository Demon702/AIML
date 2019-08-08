import numpy as np
from utils import *
import matplotlib.pyplot as plt
def preprocess(X, Y):
	''' TASK 0
	X = input feature matrix [N X D] 
	Y = output values [N X 1]
	Convert data X, Y obtained from read_data() to a usable format by gradient descent function
	Return the processed X, Y that can be directly passed to grad_descent function
	NOTE: X has first column denote index of data point. Ignore that column 
	and add constant 1 instead (for bias part of feature set)
	'''
	print(Y.shape)
	print(X.shape)
	counter = 0
	Xnew = np.array([[1]*X.shape[0]])
	X = np.array(X)
	counter = 0
	for i in range(X.shape[1]-1):
		# print(i)
		# print("element = ",X[0][i+1])
		if isinstance(X[0][i+1], int) or isinstance(X[0][i+1], float):
			feature = X[:,i+1]
			mean = np.mean(feature)
			std = np.std(feature)
			# print(mean,std)
			counter+=1
			newfeature = ((feature - mean)/std).astype(float);
			Xnew = np.concatenate((Xnew,[newfeature]),axis = 0)
		else:
			labels = []
			
			for j in range(X.shape[0]):
				# print(j)
				if X[j][i+1] not in labels:
					# print(X[j][i+1])
					counter+=1
					labels.append(X[j][i+1])
			# print("///////////////////////////////////////////")
			# print(labels)
			newfeature = one_hot_encode(X[:,i+1], labels).astype(float)
			# print(Xnew.shape)
			# print(newfeature.shape)
			Xnew = np.concatenate((Xnew,newfeature.transpose()),axis = 0)
			# print("Xnew shape", Xnew.shape)
	
 
	# print("counter = ",counter)	
	print("Xnew shape", Xnew.shape)
	return (Xnew.transpose(), Y.astype(float))
	

def grad_ridge(W, X, Y, _lambda):
	'''  TASK 2
	W = weight vector [D X 1]
	X = input feature matrix [N X D]
	Y = output values [N X 1]
	_lambda = scalar parameter lambda
	Return the gradient of ridge objective function (||Y - X W||^2  + lambda*||w||^2 )
	'''
	return -np.matmul(X.transpose(), (Y - X@W)) + _lambda* W
	pass

def ridge_grad_descent(X, Y, _lambda, max_iter=30000, lr=0.00001, epsilon = 1e-4):
	''' TASK 2
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	lr 			= learning rate
	epsilon 	= gradient norm below which we can say that the algorithm has converged 
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	NOTE: You may precompure some values to make computation faster
	'''
	print("X shape", X.shape)
	X = np.array(X, dtype = np.double)
	Y = np.array(Y, dtype = np.double)
	W = np.zeros([X.shape[1] , Y.shape[1]], dtype = np.double)
	for _ in range(max_iter):
		gradient = grad_ridge(W,X,Y,_lambda)
		if np.linalg.norm(gradient) < epsilon:
			break;
		W = W - lr* gradient
	return W

def k_fold_cross_validation(X, Y, k, lambdas, algo):
	''' TASK 3
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	k 			= number of splits to perform while doing kfold cross validation
	lambdas 	= list of scalar parameter lambda
	algo 		= one of {coord_grad_descent, ridge_grad_descent}
	Return a list of average SSE values (on validation set) across various datasets obtained from k equal splits in X, Y 
	on each of the lambdas given 
	'''
	length = len(X)
	print("length = ", length)
	split_length = int(length/k)
	print("split_length = ", split_length)
	# splitdata = np.empty((0,0,X.shape[1]), np.double)
	sselist = []
	for lambda0 in lambdas:
		sse = 0
		for i in range(k):
			kth_train_label = np.array([])
			kth_train_label = Y[: i*split_length]
			if i == 0:
				kth_train_dataset = np.empty((0,X.shape[1]), np.double)
			else:
				kth_train_dataset = np.array(X[: i*split_length,:])
			# print(kth_dataset)
			# print("=============================")
			# print(X[(i + 1)*split_length :,:])
			# print("train dataset")
			kth_train_dataset = np.concatenate((kth_train_dataset, X[(i + 1)*split_length :,:]), axis =0)
			kth_train_label = np.append(kth_train_label, Y[(i + 1)*split_length :,:], axis =0)
			# print(kth_train_dataset)
			kth_validation_dataset  = X[i*split_length: (i+1)*split_length, :]
			kth_validation_label = Y[i*split_length: (i+1)*split_length]
			# print("test_set")
			# print(kth_validation_dataset)
			# print("label_set")
			# print(kth_train_label)
			trained_weight = algo(kth_train_dataset , kth_train_label, lambda0)
			Ypredicted = kth_validation_dataset@trained_weight
			sse += np.linalg.norm(Ypredicted - kth_validation_label)**2
		sselist.append(sse/k)



		
	return sselist

def coord_grad_descent(X, Y, _lambda, max_iter=1000):
	''' TASK 4
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	'''
	X = np.array(X, dtype = np.float)
	Y = np.array(Y, dtype = np.float)
	W = np.array(np.random.rand(X.shape[1],1), np.float)
	# print(X.shape)
	# print(Y.shape)
	# print(W.shape)
	# print(Y.flatten().shape)
	print("lambda = ", _lambda)
	# first_term  = (2* Y.transpose() @X).flatten()
	# print("first_term shape", first_term.shape)
	# print(first_term)
	# print("second_term shape", second_term.shape)
	for _ in range(1000):

		difference = []
		for i in range(W.shape[0]):

			# for j in range(W.shape[1]):
			Xelem =  X[:,i]
			W[i] = 0
			# print(W)
			# second_term = 2*X.transpose()@X
			# norm_gradient = _lambda * np.sign(W[i])
			# # print("norm_gradient = ",norm_gradient)
			intermediate = Y - np.matmul(X,W)
			# print("first_term = ", first_term[i])
			# print("second_term = ", second_term[i])
			# print(_)
			# print("intermediate")
			# print(np.dot(intermediate,Xelem))
			# print("Xelem norm")
			# # print(np.dot(Xelem,Xelem))
			# sum = 0
			# for j in range(len(W)):
			# 	if j!=i:
			# 		sum+= W[j]*second_term[j,i]

			# # print("Xelem",i)
			# # print(Xelem)
			# if second_term[i,i] != 0:
			# 	updatedw = (sum + first_term[i])/second_term[i,i]
			# else:
			# 	updatedw = 0
			if np.dot(Xelem,Xelem)>0:
			
				updatedw = (np.matmul(intermediate.transpose(),Xelem) -1.0*_lambda) /np.dot(Xelem,Xelem)

			# print(np.dot(intermediate.flatten(),Xelem))
				if updatedw <0:
					updatedw =  (np.matmul(intermediate.transpose(),Xelem) + 1.0*_lambda) /(np.dot(Xelem,Xelem))
					if updatedw > 0:
						updatedw = 0
			else:
				updatedw = 0
			# print(updatedw)
			# print(i,j ,abs(updatedw - W[i,j]))
			difference.append(abs(updatedw - W[i]))
			W[i] = updatedw
			# print(W)
		# print("SSe = " , max(difference))
		# print("SSE actual = ", np.linalg.norm(Y.flatten() - X@W))
		# if max(difference) <= 1e-4:
		# 	break
	print([W])
	return W

if __name__ == "__main__":
	# Do your testing for Kfold Cross Validation in by experimenting with the code below 
	X, Y = read_data("./dataset/train.csv")
	X, Y = preprocess(X, Y)
	trainX, trainY, testX, testY = separate_data(X, Y)
	
	lambdas = [10000,50000,100000,150000,200000,250000,300000,350000] # Assign a suitable list Task 5 need best SSE on test data so tune lambda accordingly
	scores = k_fold_cross_validation(trainX, trainY, 6, lambdas, coord_grad_descent)
	plot_kfold(lambdas, scores)