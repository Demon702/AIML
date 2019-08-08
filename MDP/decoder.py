import sys
import numpy as np
import copy
import random
if __name__ == "__main__":
	filename = sys.argv[1]
	solution_file = sys.argv[2]
	with open(filename) as f:
		a = np.array(f.readlines())
		array = np.array([b.split(' ') for b in a]).astype(int)
	# print(array)
	counterarray = np.zeros(array.shape, dtype = int)
	counter = 0
	for i in range(array.shape[0] - 1):
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
			
			if action == 0:
				print("W" ,end = " ")
				y-=1
				line_to_read = counterarray[x,y] - 1
			elif action == 1:
				print("E" ,end = " ")
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
			# print("nextline = ", line_to_read + 1)
			count+= 1 
				

