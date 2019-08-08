import sys
import numpy as np
import copy
if __name__ == "__main__":
	filename = sys.argv[1]
	with open(filename) as f:
		firstline = f.readline()
		numstate = int(firstline.split(' ')[1])
		secondline = f.readline()
		numaction = int(secondline.split(' ')[1])
		thirdline = f.readline()
		startstate = int(thirdline.split(' ')[1])
		fourthline = f.readline()
		endstate = [int(x) for x in fourthline.split(' ')[1:] ]
		dictionary = {}
		line = f.readline()
		all_list = []
		
		for i in range(numstate):
			all_list.append([])	

		for i in range(numstate):
			for j in range(numaction):
					all_list[i].append([])
		while(line):
			splitarray = line.split(' ')
			if splitarray[0] == "discount":
				break
			state1 = int(splitarray[1])
			action = int(splitarray[2])
			state2 = int(splitarray[3])
			reward = float(splitarray[4])
			probability = float(splitarray[5])
			# if state1> endstate[counter] and endstate!= [-1]:
			# 	all_list[state1 - 1][action].append([state2, reward , probability])
			# else:
			all_list[state1][action].append([state2, reward , probability])
			line = f.readline()
			# print(state1, action, state2,reward, probability )
		discount = float(line.split(' ')[2])
		# print("startstate = ", startstate)
		# print("endstate = ", endstate)
		# print("discount = ", discount)
	all_list = np.array(all_list)
	# print(all_list.shape)
	# print(all_list)
	# print(dictionary.keys())

	list_for_work = np.empty([0,2])
	for i in range(numstate):
		if i in endstate:
			list_for_work = np.concatenate((list_for_work,[[0,-1]]))
			continue
		list_for_work = np.concatenate((list_for_work,[[0.0,0]]))

	# all_list = []
	# for state1 in range(numstate):
	# 	statelist = []
	# 	for action in range(numaction):
	# 		actionlist = []
	# 		for state2 in range(numstate):
	# 			if (state1, action, state2) in dictionary.keys():
	# 				# print(np.append([state2], dictionary[state1, action , state2]))
	# 				actionlist.append([state2, dictionary[state1, action , state2][0],dictionary[state1, action , state2][1]])
	# 		statelist.append(actionlist)
	# 	all_list.append(statelist)
	# print(all_list)
	# print("list_for_work",list_for_work)
	iterations = 0
	# is_action = np.zeros([numstate, numaction, numstate])
	# for state1 in range(numstate):
	# 	for action in range(numaction):
	# 		for state2 in range(numstate):
	# 			if (state1,action,state2) in dictionary.keys():
	# 				print "Reached here"
	# 				is_action[state1,action, state2] = 1
	while(1):
		# print(iterations)
		# print("//////////////////////////////////////////////////////////////////")
		iterations+=1
		oldlist = copy.deepcopy(list_for_work)
		for state1 in range(numstate):
			if state1 in endstate:
				# endstatecrossed = 1
				continue
			# valuelist = np.empty([0,2])
			actionlist = []
			for action in range(numaction):
				# print(state1, action)
				value = 0.0
				for prob in all_list[state1, action]:
					
					# print(prob)
					state2 = int(prob[0]) 
						# print(state1,action,state2)
						# print("first ",[action,list_for_work[state2,0]])
						# print("seond",dictionary[state1,action,state2])
						# print(np.append(list_for_work[state2,0],dictionary[state1,action,state2]))
					# actionlist = np.concatenate((actionlist,[[oldlist[state2][0],prob[1], prob[2]]]))
					value+= prob[2]*(prob[1] + discount* oldlist[state2][0])
				actionlist.append(value)
				# print("actionlist = ", actionlist)
				# value = list(map(lambda x: x[2]*(x[1] + discount * x[0]) , actionlist))
				# valuelist = np.concatenate((valuelist,[[action,sum(value)]]), axis = 0)
			# print("valuelist = ", valuelist)
			# highest = np.where(valuelist[:,1] == max(valuelist[:,1]))[0]
			# print(highest)
			# 
			list_for_work[state1,0] = max(actionlist)
			list_for_work[state1,1] =  actionlist.index(max(actionlist))
		# print(list_for_work)
		if max(abs(oldlist[:,0] - list_for_work[:,0])) <= 1e-16:
			break
		# print(max(abs(oldlist[:,0] - list_for_work[:,0])))

	for i in list_for_work:
		print(i[0], int(i[1]))
	print("iterations", iterations)
	
	# print(dictionary)