import numpy as np

class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		# Stores the outgoing summation of weights * feautres 
		self.data = None

		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))	
		self.biases = np.random.normal(0,0.1, (1, out_nodes))
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary 

	def forwardpass(self, X):
		# print('Forward FC ',self.weights.shape)
		# Input
		# activations : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layers
		
		# n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_nodes]
		# OUTPUT activation matrix		:[n X self.out_nodes]

		###############################################
		# TASK 1 - YOUR CODE HERE
		# print("Fully Connected Forward",self.weights.shape)
		# print(X.shape)
		self.data = np.matmul(X,self.weights) + self.biases
		return sigmoid(self.data)
		# raise NotImplementedError
		###############################################
		
	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		noofsamples = activation_prev.shape[0]
		# print("Backward Fullyconnected")
		del_Error_del_normal_curr = delta * derivative_sigmoid(self.data)
		deltaweight = (1/noofsamples)*np.matmul(del_Error_del_normal_curr.transpose(),activation_prev)
		deltabias = (1/noofsamples)* sum(del_Error_del_normal_curr,0)
		old_weight = self.weights
		self.weights = self.weights - lr * deltaweight.transpose()
		self.biases = self.biases - lr* deltabias
		return np.matmul(old_weight,del_Error_del_normal_curr.transpose()).transpose()
		# raise NotImplementedError
		###############################################

class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, numfilters, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for convolution layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer
		# numfilters  - number of feature maps (denoting output depth)
		# stride	  - stride to used during convolution forward pass
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = numfilters
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		# Stores the outgoing summation of weights * feautres 
		self.data = None
		
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		self.biases = np.random.normal(0,0.1,self.out_depth)
		

	def forwardpass(self, X):
		# print('Forward CN ',self.weights.shape)
		# print(self.in_depth)
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		# n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE
		# print(X.shape)
		# print(self.stride)
		# raise NotImplementedError
		# out = np.zeros([out_row,out_col])
		# print("Backward CNN")
		out = np.zeros([X.shape[0],self.out_depth, self.out_row, self.out_col])
		for image in range(X.shape[0]):
			for filter in range(self.out_depth):
				filtermatrix = self.weights[filter,:,:,:]
				intermediate = np.zeros([self.in_depth,self.out_row,self.out_col])
				for indepth in range(self.in_depth):
					for i in range(self.out_row):
						for j in range(self.out_col):
							# print(image, filter, indepth,i,j)
							a = i*self.stride
							b = j*self.stride
							intermediate[indepth,i,j] = sum(sum(X[image,indepth,a:a + self.filter_row,b:b+self.filter_col]*filtermatrix[indepth,:,:]))
				out[image, filter,: ,:] = sum(intermediate,0) + self.biases[filter]
		self.data = out	
		# X = np.array(X)
		# Y = np.array(Y)
		# [filter_row,filter_col] = Y.shape
		# [in_row,in_col] = X.shape
		# # print(filter_row,filter_col)
		# for i in range(out_row):
		# 	for j in range(out_col):
		# 		a = i*stride
		# 		b = j*stride
		# 		out[a,b] = sum(sum(X[a:a + filter_row,b:b+filter_col]*Y))
		# 		# out[a,b]=sum(sum(X[i:i+ filter_row-1,j:j+filter_col-1]*Y))
		return sigmoid(out)
		###############################################

	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size
		deltaweight = np.zeros(self.weights.shape)
		del_Error_del_normal_curr = delta * derivative_sigmoid(self.data)

		###############################################
		# TASK 2 - YOUR CODE HERE
		for outdepth in range(self.out_depth):
			for i in range(self.out_row):
				for j in range(self.out_col):
					# print(image, filter, indepth,i,j)
					a = i*self.stride
					b = j*self.stride
					for image in range(n):
						deltaweight[outdepth,:,:,:]+= (1/n)*activation_prev[image,:,a:a +self.filter_row,b:b + self.filter_col]*del_Error_del_normal_curr[image,outdepth,i,j] 
		deltabias = np.zeros(self.biases.shape)
		for outdepth in range(self.out_depth):
			for image in range(n):
				deltabias[outdepth]+= sum(sum(del_Error_del_normal_curr[image,outdepth,:,:]))
		deltabias = deltabias/n
		self.weights-= lr * deltaweight
		self.biases-= lr* deltabias
		out = np.zeros(activation_prev.shape)
		for image in range(n):
			for outdepth in range(self.out_depth):
				for i in range(self.out_row):
					for j in range(self.out_col):
						a = i*self.stride
						b = j*self.stride
						out[image,:,a:a +self.filter_row,b:b + self.filter_col]+=del_Error_del_normal_curr[image,outdepth,i,j]*self.weights[outdepth,:,:,:]

		return out
		# raise NotImplementedError
		###############################################
	
class AvgPoolingLayer:
	def __init__(self, in_channels, filter_size, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for max_pooling layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer

		# NOTE: Here we assume filter_size = stride
		# And we will ensure self.filter_size[0] = self.filter_size[1]
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = self.in_depth
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

	def forwardpass(self, X):
		
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		# n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE
		# print("Pooling Layer",X.shape)
		out =  np.zeros([X.shape[0],X.shape[1],self.out_row,self.out_col])
		for image in range(X.shape[0]):
			for filter in range(self.out_depth):
				for i in range(self.out_row):
					for j in range(self.out_col):
						a = i*self.stride
						b = j*self.stride
						out[image, filter,i,j] = np.mean(X[image, filter, a: a + self.filter_row, b : b + self.filter_col])




		return out 
		# raise NotImplementedError
		###############################################


	def backwardpass(self, alpha, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# activations_curr : Activations of current layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		out = np.zeros(activation_prev.shape)
		n = activation_prev.shape[0] # batch size
		for i in range(self.out_row):
			for j in range(self.out_col):
				a = i*self.stride
				b = j*self.stride
				for image in range(activation_prev.shape[0]):
					for filter in range(activation_prev.shape[1]):
						out[image,filter,a:a +self.filter_row,b:b + self.filter_col]+=delta[image,filter,i,j]*np.full([self.filter_row,self.filter_col],(1/(self.filter_row*self.filter_col)))

		return out
		###############################################
		# TASK 2 - YOUR CODE HERE
		# raise NotImplementedErro

		###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        # print("Flatten Layer",X.shape)
        # print("Flatten Layer",self.in_batch, self.r * self.c * self.k)
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)


# Helper Function for the activation and its derivative
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))
