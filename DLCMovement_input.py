#Save time and motion vectors
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy.signal as sig
import operator
import os
import glob
import statistics as st
import scipy.stats as stats
from scipy import signal
import matplotlib.patches as patches 
import cv2
# from neuraltoolkit import NTKVideos 

#first argument = name of h5 file to be processed
#second argument = timeestamp of video save, as obtained from videotimestamp.vidtimestamp()

def get_pos(videoh5file, labels = False):
        ###GET POSITION VECTOR
	#print('here1')
	df = pd.read_hdf(videoh5file)
	#print('here2')
	if labels == False:
		o = df.values
	else:
		coords = np.array(["x", "y", "likelihood"])
		coords_full = np.matlib.repmat(coords.reshape(3,1), len(labels),1)
		coords_full = list(coords_full.flatten())
		b = np.matlib.repmat(labels,3,1)
		labels_full = list(np.transpose(b).flatten())
		a = np.full(np.size(coords_full), df.columns.get_level_values(0)[0])
		a =list(a)
		tuples = list(zip(a,labels_full, coords_full))
		#multi_tuples = [(df.columns.get_level_values(0)[0],labels[0],"x"), (df.columns.get_level_values(0)[0],labels[0],"y"), (df.columns.get_level_values(0)[0],labels[0],"likelihood")]
		multi_cols= pd.MultiIndex.from_tuples(tuples, names=["scorer","bodyparts", "coords"])
		df_cut = pd.DataFrame(df, columns=multi_cols)
		o =  df_cut.values

	return o,df


def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

def get_movement(vidfile, num_labels = 1, plotter = 0, labels = False, savedir = False):
	print('USING THIS VERSION')
	print(vidfile)
	print(type(vidfile))
	basename = vidfile.split('/')[-1][:-3]
	print(basename)
	print(type(basename))
	mp4_1 = cv2.VideoCapture(savedir +basename+'_labeled.mp4')
	if plotter:
			fig1, ax1 = plt.subplots(nrows = 1, ncols = 1, figsize = [15, 4])
			plt.ion()
	print(mp4_1)
	#pos_total = np.zeros(num_labels*3)
	# The order of the ndarray is x, y, likelihood
	pos_i,df = get_pos(vidfile, labels =labels)
	pos_it = pos_i.transpose()
	#pos_total = np.vstack((pos_total,pos_i))
	#pos_total = np.delete(pos_total, (0), axis = 0)

	timevect_frame = np.arange(pos_i.shape[0])

	try:
		fr = mp4_1.get(5)
	except :
		fr = 15
		print('it failed to load')

	#print(fr)

	timevect_nansec = (timevect_frame/fr) * 1000 * 1000 * 1000 # in nanose

	if num_labels>1:
    	#combine time array and pos mat
		x_idx = np.arange(0, np.shape(pos_i)[1],3)
		y_idx = np.arange(1, np.shape(pos_i)[1],3)
		like_idx = np.arange(2, np.shape(pos_i)[1],3)
		time_and_x = np.vstack((timevect_nansec, pos_it[x_idx]))
		dtx = np.diff(time_and_x)
		time_and_y = np.vstack((timevect_nansec, pos_it[y_idx]))
		dty = np.diff(time_and_y)
		like = pos_it[like_idx]


		best_label = [np.where(like[:,i] == np.max(like[:,i]))[0][0] for i in np.arange(1,  np.shape(like)[1])]
		best_like  = [np.max(like[:,i]) for i in np.arange(1,  np.shape(like)[1])]
		bad_like   = np.where(np.array(best_like)<0.5)

		best_dx = [dtx[best_label[i]+1,i] for i in np.arange(np.size(best_label))]
		best_dx = np.asarray(best_dx)


		best_dy = [dty[best_label[i]+1,i] for i in np.arange(np.size(best_label))]
		best_dy = np.asarray(best_dy)


		best_dxy = np.sqrt(np.square(best_dx) + np.square(best_dy))

	else:

		time_and_x = np.vstack((timevect_nansec, pos_it[0]))
		dtx = np.diff(time_and_x)
		time_and_y = np.vstack((timevect_nansec, pos_it[1]))
		dty = np.diff(time_and_y)
		best_dxy = np.sqrt(np.square(dtx[1]) + np.square(dty[1]))
    

    #time_and_pos_rectime = time_and_pos[2480:,:]

	time_sec = timevect_nansec*1e-9
	#print(time_sec)
	dt = time_sec[1]
	#print(dt)
	binsz = int(round(1/dt))
	#print(binsz)
	dxy = best_dxy

	while np.size(dxy) < mp4_1.get(7):
		dxy = np.append(dxy, np.float('nan'))


	np.save(savedir + basename+'_full_movement_trace.npy', dxy)

	t1 = int(basename[basename.find('T')+3:basename.find('T')+5])
	t2 = int(basename[basename.find('T')+10:basename.find('T')+12]) 

	if t2-t1 < 0:
		print('This is the last video, I am going to help it reshape!')
		while int(np.size(dxy)/binsz) != np.size(dxy)/binsz:
			print('edit')
			dxy = np.append(dxy, np.float('nan'))


        # reshape and calculate averaged locomoation activity
	try:
		rs_dxy = np.reshape(dxy,[int(np.size(dxy)/binsz), binsz])
	except ValueError:
		#dxy = np.append(dxy, np.float('nan'))
		print('cant reshape array with size'+str(np.size(dxy)))
		m = np.size(dxy) % fr
		if m<7:
			right_size = np.size(dxy) - m
		else:
			right_size = np.size(dxy) + (fr-m)
		while np.size(dxy) > right_size:
			dxy = dxy[0:-1]
		while np.size(dxy) < right_size:
			dxy = np.append(dxy, np.float('nan'))
	try:
		rs_dxy = np.reshape(dxy,[int(np.size(dxy)/binsz), binsz])
	except ValueError:
		print('excepted an error')
		dxy = np.append(dxy, np.float('nan'))
		rs_dxy = np.reshape(dxy, [int(np.size(dxy) / binsz), binsz])
    
    

	med = np.median(rs_dxy, axis = 1)
	#binned_dxy = np.mean(rs_dxy, axis = 1)

	x_vals = np.linspace(0,60,np.size(med))
	hist = np.histogram(med[~np.isnan(med)], bins = 1000)
	csum = np.cumsum(hist[0])
	th = np.size(med)*0.95
	if max(csum) > th:
		outliers_idx = np.where(csum>th)[0][0]
		outliers = np.where(med>hist[1][outliers_idx])[0]

		for i in outliers:
			med[i] = med[i-1]
			a = i-1
			while med[i] > hist[1][outliers_idx]:
				a = i-1
				med[i] = med[a]
		#plt.plot(x_vals, binned_dxy)
		if plotter:
			plt.plot(x_vals, med)

		sorted_med = np.sort(med)
		idx = np.where(sorted_med>int(max(sorted_med)*0.40))[0][0]

		if idx == 0:
			thresh = sorted_med[idx] 
		#print(int(max(sorted_med)*0.50))
		else:
			thresh = np.nanmean(sorted_med[0:idx])
		# ymax = plt.gca().get_ylim()[1]
		# plt.figure()
		# plt.plot(sorted_med)
		# plt.plot([idx,idx],[0, ymax])
		#print(thresh)

		# moving = np.where(med > thresh)[0]
		if plotter:
			moving = np.where(dxy > thresh)[0]
			h = plt.gca().get_ylim()[1]
			# consec = group_consecutives(np.where(med > thresh)[0])
			consec = group_consecutives(np.where(med > thresh)[0])
			for vals in consec:
				if len(vals)>5:
					x = x_vals[vals[0]]
					y = 0
					width = x_vals[vals[-1]]-x
					rect = patches.Rectangle((x,y), width, h, color = '#b7e1a1', alpha = 0.5)
					ax1.add_patch(rect)
			plt.show()
			plt.title(vidfile)

	np.save(savedir + basename+'_movement_trace.npy', med)
	
	print('here')
	return(basename)













