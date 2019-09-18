import os
import numpy as np
import matplotlib.pyplot as plt
import neuraltoolkit.ntk_ecube as ntk
import glob

def findPulse(dirb, df):
	'''
	finds the binary file that contains the sync pulse for the camera
	dirb: digital binary directory
	df: the first file in that directory '''
	thresh = 800
	files = glob.glob(dirb+'/*.bin')
	files = np.sort(files)
	flag = False
	for f in files:
		t,dr = ntk.load_digital_binary(f)
		max_pos=np.where(dr==1)

		for i in range(len(max_pos[0])-thresh):
			if (max_pos[0][i+thresh]-max_pos[0][i]) == thresh:
				print('binary file:',f, '\nindex of the pulse in the max_pos array: ', i)
				plt.plot(dr[max_pos[0][i]-5000:max_pos[0][i]+5000])
				flag = True
				break
		if flag:
			break