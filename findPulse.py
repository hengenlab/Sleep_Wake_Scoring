import os
import numpy as np
import matplotlib.pyplot as plt
import neuraltoolkit.ntk_ecube as ntk

def findPulse(dirb, df):
	'''
	finds the binary file that contains the sync pulse for the camera
	dirb: digital binary directory
	df: the first file in that directory '''
	t,dr = ntk.load_digital_binary(df)

	max_pos=np.where(dr==1)
	zpos = np.where(dr==0)
	first_on = max_pos[0][0]
	next_off = zpos[0][first_on]
	dif = next_off-first_on 
	thresh = dif*2

	files = os.listdir(dirb)
	files = np.sort(files)
	flag = False
	for f in files:
		path = dirb+f
		t,dr = ntk.load_digital_binary(path)
		max_pos=np.where(dr==1)

		for i in range(len(max_pos[0])-thresh):
			if (max_pos[0][i+thresh]-max_pos[0][i]) == thresh:
				print('binary file:',path, '\nindex of the pulse in the max_pos array: ', i)
				plt.plot(dr[max_pos[0][i]-5000:max_pos[0][i]+5000])
				flag = True
				break
		if flag:
			break