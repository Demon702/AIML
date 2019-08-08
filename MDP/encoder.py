import sys
import numpy as np
import copy
if __name__ == "__main__":
	filename = sys.argv[1]
	if len(sys.argv) == 3:
		p = float(sys.argv[2])
	else:
		p = 1.0
	with open(filename) as f:
		a = np.array(f.readlines())
		array = np.array([b.split(' ') for b in a]).astype(int)
	# print(array)
	counterarray = np.zeros(array.shape, dtype = int)
	counter = 0
	endstate = []
	for i in range(array.shape[0]):
		for j in range(array.shape[1]):
			element = array[i][j]
			if element == 0:
				counterarray[i,j] = counter
				counter+=1
			elif element == 2:
				counterarray[i,j] = startstate = counter
				counter+=1
			elif element == 3:
				counterarray[i,j] = counter	
				endstate.append(counter)
				counter+=1
	numstates = counter
	print("numStates" , numstates)
	print("numActions" , 4)
	print("start" , startstate)
	print("end" , )
	for i in endstate:
		print(i ,)
	print()
	# print(counterarray)
	# print(startstate)
	# print(endstate)
	numstates = counter
	dictionary = {}
	for i in range(array.shape[0]):
		for j in range(array.shape[1]):
			if array[i][j] == 2 or array[i][j] == 0:
				left = max(j-1,0)
				right = min(j+1,array.shape[1] - 1)
				upper = max(i-1,0)
				lower = min(i+1, array.shape[0] - 1)
				# possible_moves = np.array([array[i,left], array[i,right] , array[upper,j] , array[lower, j]])
				# possible_moves = (possible_moves != 1)
				# denominator = sum(possible_moves)
				# actual_moves = [counterarray[i,j], counterarray[i,j], counterarray[i,j], counterarray[i,j]]
				# for k in range(4):
				# 	if possible_moves[k]:
				# 		if k == 0:
				# 			actual_moves[k] = counterarray[i,left]
				# 		elif k == 1:
				# 			actual_moves[k] = counterarray[i, right]
				# 		elif k == 2:
				# 			actual_moves[k] = counterarray[upper, j]
				# 		elif k == 3:
				# 			actual_moves[k] = counterarray[lower,j]
				# for action1 in range(4):
				# 	if possible_moves[action1]:
				# 		denominator = sum(possible_moves)
				# 		actual_prob = p + (1 - p)/denominator
				# 		residual_prob = (1 - p)/denominator
				# 		for action2 in range(4):
				# 			if possible_moves[action2]:
				# 				if (action2 == action1):
				# 					print "transition" ,counterarray[i,j],action1, actual_moves[action2],-1,actual_prob
				# 				else:
				# 					print "transition" ,counterarray[i,j],action1, actual_moves[action2],-1,residual_prob
				# 	else:
				# 		print "transition" ,counterarray[i,j],action1, counterarray[i,j],-1000,1.0

				if array[i, left]!= 1:
					print("transition" ,counterarray[i,j],0, counterarray[i,left],-1,1)
				else:
					print("transition" ,counterarray[i,j],0, counterarray[i,j],-100000,1)

				if array[i, right]!= 1:
					print("transition" ,counterarray[i,j],1, counterarray[i,right],-1,1)				
				else:
					print("transition" ,counterarray[i,j],1, counterarray[i,j],-100000,1)

				if array[upper, j]!= 1:
					print("transition" ,counterarray[i,j],2, counterarray[upper,j],-1,1)				
				else:
					print("transition" ,counterarray[i,j],2, counterarray[i,j],-100000,1)

				if array[lower, j]!= 1:
					print("transition" ,counterarray[i,j],3, counterarray[lower,j],-1,1)				
				else:
					print("transition" ,counterarray[i,j],3, counterarray[i,j],-100000,1)

	# print(dictionary)
		
	print("discount" , "", 1.0)