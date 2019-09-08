#!/usr/bin/env python
import os.path
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors
import time

#filename = "/Users/mobby/git/ex_data/logfile_test_1000.txt"


def extract_data(filename):

	L_X = []
	L_Y = []
	L_angles = []
 
	if not os.path.isfile(filename):
		print ("File does not exist.")

	else:
	# Open the file as f.
	# The function readlines() reads the file.
		with open(filename) as f:
			content = f.read().splitlines()
	 
	# Show the file contents line by line.
	# We added the comma to print single newlines and not double newlines.
	# This is because the lines contain the newline character '\n'.

		for line in content:

			split_line = line.split()

			print(split_line)

			if len(split_line) > 2:

				L_X.append([float(split_line[7])]) 
				L_Y.append([float(split_line[8])])
				L_angles.append([float(split_line[4]), float(split_line[5]), float(split_line[6])])
				#L_fitness.append(float(split_line[3]))
				print("pos : ",L_X[-1]," ",L_Y[-1])
				target = [float(split_line[-2]), float(split_line[-1])]

	print( "target position: ", target)

	print(len(L_X)," timesteps extracted from", filename)

	return(L_X, L_Y, L_angles, target)


def forward_model(theta1, theta2, theta3):

	#print(theta1)
	#print(type(theta1))

	T_12 = np.array([[np.cos(theta1), -np.sin(theta1)],[np.sin(theta1), np.cos(theta1)]])
	T_23 = np.array([[np.cos(theta2), -np.sin(theta2)],[np.sin(theta2), np.cos(theta2)]])

	return(T_12,T_23)


def get_position(theta1,theta2,theta3) :

	l1 = 1/3 #arm's first part size
	l2 = 1/3
	l3 = 1/3

	T_12, T_23 = forward_model(theta1, theta2, theta3)
	T_13 = T_12@T_23

	v_1 = np.array([[l1*np.cos(theta1)],[l1*np.sin(theta1)]])
	v_2 = T_12 @ np.array([[l2*np.cos(theta2)],[l2*np.sin(theta2)]])
	v_3 = T_13 @ np.array([[l3*np.cos(theta3)],[l2*np.sin(theta3)]])

	return (v_1 + v_2 + v_3, v_1, v_2, v_3)

def get_positions(angles) :

	L_pos = []
	L_vec = []

	for angle in angles :

		pos, v_1, v_2, v_3 = get_position(angle[0], angle[1], angle[2])
		L_pos.append(pos)
		L_vec.append([v_1,v_2,v_3])

	# plt.figure()
	# plt.scatter()

	return(L_pos, L_vec)

def plot_arm(pos, vecs, axe):

	[v_1, v_2, v_3] = vecs

	axe.scatter(pos[0], pos[1])

	# print(vectors[0][0])
	# print(vectors[0][1])
	# print(vectors[0][2])

	axe.plot([0,v_1[0]],[0,v_1[1]])
	axe.plot([v_1[0],v_1[0]+v_2[0]],[v_1[1],v_1[1]+v_2[1]])
	axe.plot([v_1[0]+v_2[0], v_1[0]+v_2[0]+v_3[0]],[v_1[1]+v_2[1],v_1[1]+v_2[1]+v_3[1]])

	#plt.show()

def init_figure(xmin,xmax,ymin,ymax):
	fig = figure(0)
	ax = fig.add_subplot(111, aspect='equal')
	ax.xmin=xmin
	ax.xmax=xmax
	ax.ymin=ymin
	ax.ymax=ymax
	clear(ax)
	return ax

def clear(ax):
	pause(0.001)
	cla()
	ax.set_xlim(ax.xmin,ax.xmax)
	ax.set_ylim(ax.ymin,ax.ymax)

if __name__ == '__main__':

	### PLOT RANDOM BEHAVIOURS
	# for i in range(0,5):

	# 	filename = "/Users/mobby/git/ex_data/test_1000it_10_sec2/logfile_test" + str(i) + ".txt"
		
	# 	L_X, L_Y, target = extract_data(filename)

	# 	plt.figure()
		
	# 	print("Target :", target)
	# 	plt.scatter(L_X[0],L_Y[0],c='black')

	# 	plt.scatter(L_X[1:],L_Y[1:],c='r')

	# 	plt.scatter(target[0],target[1],c='b')


	### PLOT BEST FIT
	#extension = input("Which file would you want to visualize ? (it should be in ex_data directory) ")

	#filename = "/Users/mobby/git/" + extension

	#filename = "/Users/mobby/git/ex_data/2019-08-20_13_59_03_1913/final_model_35.txt"
	#filename = "/Users/mobby/git/ex_data/2019-08-20_13_59_03_1913/final_model_712.txt"
	filename = "/Users/mobby/git/ex_data/2019-08-20_13_59_03_1913/final_model_392.txt"
		
	L_X, L_Y, L_angles, target = extract_data(filename)

	L_pos, L_vec = get_positions(L_angles)
	plt.ion()
	fig = plt.figure(0)
	ax = fig.add_subplot(111, aspect='equal', projection='3d')
	ax.xmin=-0.4
	ax.xmax=0.6
	ax.ymin=-0.4
	ax.ymax=0.6
	plt.pause(0.001)


	for i in range (len(L_pos)):

		plt.cla()
		ax.set_xlim(ax.xmin,ax.xmax)
		ax.set_ylim(ax.ymin,ax.ymax)
		#print("pos")

		pos = L_pos[i]
		[v_1, v_2, v_3] = L_vec[i]

		ax.plot([0,v_1[0]],[0,v_1[1]],0.2, c = 'grey')
		ax.plot([v_1[0],v_1[0]+v_2[0]],[v_1[1],v_1[1]+v_2[1]],0.2, c='grey')
		ax.plot([v_1[0]+v_2[0], v_1[0]+v_2[0]+v_3[0]],[v_1[1]+v_2[1],v_1[1]+v_2[1]+v_3[1]],0.2, c='grey')


		#additional points
		ax.scatter(target[0], target[1], 0.2, c = 'green', label = 'targ')
		# ax.scatter(0.1,0.5, 0.2, c='black', label = 'obs')
		# ax.scatter(0,0.4, 0.2, c = 'black', label = 'obs')
		#ax.scatter(-0.2, 0.3, 0.2, c='black', label = 'obs')
		ax.scatter(L_pos[0][0], L_pos[0][1], 0.2, c = 'blue', label = 'start')
		ax.scatter(L_pos[i][0], L_pos[i][1], 0.2, c = 'grey')
		ax.scatter(0, 0, 0.2, c = 'grey', label ='robot joint 0')
		ax.scatter(0, 0, 0.3, c='white')
		ax.scatter(0,0,0.1,c='white')
		plt.show()
		plt.pause(0.001)



	ax.scatter(L_X, L_Y, 0.2, c='r', marker='x')
	ax.legend()

	plt.pause(3)

	plt.show()