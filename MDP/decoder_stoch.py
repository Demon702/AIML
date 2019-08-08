import sys
import numpy as np
import copy
import random
if __name__ == "__main__":
	random.seed(5)
	filename = sys.argv[1]
	solution_file = sys.argv[2]
	if len(sys.argv) == 4:
		p = float(sys.argv[3])
	else:
		p = 1.0
	# print(p)
	with open(filename) as f:
		a = np.array(f.readlines())
		array = np.array([b.split(' ') for b in a]).astype(int)
	# print(array)
	counterarray = np.zeros(array.shape, dtype = int)
	counter = 0
	for i in range(array.shape[0]-1):
		for j in range(array.shape[1]):
			element = array[i][j]
			if element == 0:
				counter+=1
				counterarray[i,j] = counter
			elif element == 2:
				counter+=1
				counterarray[i,j] = startstate = counter
			elif element == 3:
				counter+=1
				counterarray[i,j] = endstate = counter
	# print(counterarray)
	# print(startstate)
	# print(endstate)
	[x, y] = np.where(counterarray == startstate)
	x = x[0]
	y = y[0]
	count = 0
	# print "x = ", x, "y = ", y
	with open(solution_file) as f:
		lines = np.array(f.readlines())
		# print(lines)
		line_to_read = startstate - 1
		while line_to_read!= endstate - 1:
			line = lines[line_to_read]
			splitarray = line.split(' ')
			# print(splitarray)
			action = int(splitarray[1][:-1])
			# print(action)
			left = max(y-1,0)
			right = min(y+1,array.shape[1] - 1)
			upper = max(x-1,0)
			lower = min(x+1, array.shape[0] - 1)
			possible_moves = np.array([array[x,left], array[x,right] , array[upper,y] , array[lower, y]])
			possible_moves = (possible_moves != 1)
			denominator = sum(possible_moves)
			random_number = random.random()
			actual_prob = p + (1 - p)/denominator
			residual_prob = (1 - p)/denominator
			feasible_moves = np.array([0,1,2,3])[possible_moves==1]
			# print("feasible_moves = ", feasible_moves)
			if random_number > p:
				for i in range(denominator):
					if i == denominator - 1:
						action = feasible_moves[denominator - 1]
						break
					if random_number > p + i* residual_prob and random_number <= p + (i + 1)* residual_prob:
						action = feasible_moves[i]
						break
			# print(action)
			if action == 0:
				print("W",end = " ") 
				y-=1
				line_to_read = counterarray[x,y] - 1
			elif action == 1:
				print("E",end = " ") 
				y+=1
				line_to_read = counterarray[x,y] - 1
			elif action == 2:
				print("N",end = " ")
				x-=1
				line_to_read = counterarray[x,y] - 1
			elif action == 3:
				print("S",end = " ")
				x+=1
				line_to_read = counterarray[x,y] - 1
			# print(line_to_read + 1)
			count+= 1 
	# print count, 