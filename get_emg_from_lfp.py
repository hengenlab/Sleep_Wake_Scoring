# pull data from all 4 sites
# resample at 1250 Hz
# filter 300 – 600 Hz
# pairwise pearson correlation from randomly nominated high quality pair from different sites
# mean of all pairwise correlations measured in each 0.5-second bin

import numpy as np
import scipy.signal as signal
import neuraltoolkit as ntk
from matplotlib import pyplot as plt
import pandas as pd
import glob
import os
from sys import platform
import json
import psutil

def get_highfreq_1chan(dgc_1chan,finalfs,reclen):
	bdgc = ntk.butter_bandpass(dgc_1chan, 300, 600, fs, 3)
	downsamp = signal.resample(bdgc, int(finalfs*reclen))
	return(downsamp)

def get_emg_1chan(dgc,nprobes,probechans,window,finalfs,reclen):
	randchan1 = np.random.randint(0,probechans)
	randchan2 = np.random.randint(0,probechans)

	randprobes = np.random.choice(np.arange(nprobes),2,replace=False)

	dgc_probeA = dgc[(randprobes[0]*probechans):((randprobes[0]*probechans)+probechans),:]
	dgc_probeB = dgc[(randprobes[1]*probechans):((randprobes[1]*probechans)+probechans),:]

	hfreq_p1 = get_highfreq_1chan(dgc_probeA[randchan1,:],finalfs=finalfs,reclen=reclen)
	hfreq_p2 = get_highfreq_1chan(dgc_probeB[randchan2,:],finalfs=finalfs,reclen=reclen)

	chdf = pd.DataFrame({'ch1': hfreq_p1, 'ch2': hfreq_p2})
	emg = chdf['ch1'].rolling(int(window*finalfs)).corr(chdf['ch2']).values
	return(emg)

def emg_from_lfp(filename_sw):
	'''
	1. pull data from all probes
	2. resample at 1250 Hz
	3. filter 300 – 600 Hz
	4. pairwise correlation from randomly nominated high quality pair of channels from different probes
	5. mean of all pairwise correlations measured in each 0.5-second bin
	'''

	if platform == "darwin":

		plt.switch_backend('TkAgg')
	else:

		plt.switch_backend('Agg')

	with open(filename_sw, 'r') as f:
		d = json.load(f)

	rawdat_dir = str(d['rawdat_dir'])
	# motion_dir = str(d['motion_dir'])
	LFP_dir = str(d['LFP_dir'])
	animal = str(d['animal'])
	num = int(d['num'])
	video_fr = int(d['video_fr'])
	offset = int(d['offset'])
	number_of_channels = int(d['numchan'])
	hstype = d['HS']  # Channel map
	nprobes = int(d['nprobes'])
	probechans = 64  #  number of channels per probe (symmetric)
	probenum = int(d['probenum'])  # which probe to return (starts from zero)

	fs = 25000
	finalfs=1250

	if(rawdat_dir[-4:] == '.txt'):
		file1 = open(rawdat_dir, 'r')
		lines = file1.readlines()
		files = [line.strip('\n') for line in lines]
	else:
		os.chdir(rawdat_dir)
		files = sorted(glob.glob('H*.bin'))

	filesindex = np.arange((num*12),np.size(files),12)
	if len(filesindex) == 0:
		raise ValueError('No files in this directory, check num')

	rawfiles = sorted(glob.glob(rawdat_dir + 'H*.bin')) #every hour

	for fil in filesindex:
		print('STARTING LOOP: '+str(fil))
		print('THIS IS THE STARTING USAGE: ' + str(psutil.virtual_memory()))
		start_label = rawfiles[fil][30:-4]
		end_label = rawfiles[fil+12][30:-4]
		# THIS CAN BE CONSOLIDATED INTO A FEW LINES LATER. THIS IS WHEN SAM IS LEARNING...
		# start importing your data

		# select the 12 files that you will work on at a time. this is to prevent
		# overloading RAM; since the default is to write 5 minute files, this will
		# effectively load an hour's worth of data at a time
		load_files = rawfiles[fil:fil+12]

		print('Working on hour ' + str(int((fil+12)/12)))

		full_selected_emg = np.array([])
		for a in np.arange(0,np.size(load_files)):
			print('merging file {}'.format(a))
			t, dgc = ntk.load_raw_binary_gain_chmap(rawfiles[a], number_of_channels=number_of_channels, hstype=hstype, nprobes=nprobes)

			reclen = dgc.shape[1]/fs

			#n_emg_chans = 4
			n_emg_chans = 50
			emg_chans = np.zeros([n_emg_chans,int(reclen*finalfs)])

			for e in np.arange(n_emg_chans):
				print(f'Getting emg: {e+1}/{n_emg_chans}')
				emg_chans[e,:] = get_emg_1chan(dgc,nprobes,probechans,window=0.5,finalfs=finalfs,reclen=reclen)

			emg_selected = np.mean(emg_chans,axis=0)

			#padding step necessary?
			disp = finalfs*reclen - np.size(emg_selected)
			if disp > 0:
				emg_selected = np.pad(emg_selected, (0,disp), 'constant')
			elif disp < 0:
				emg_selected = emg_selected[0:int(reclen*finalfs)]

			# full_selected_emg = np.concatenate([full_selected_emg, emg_selected], axis = 1)
			full_selected_emg = np.hstack([full_selected_emg,emg_selected])

		# plt.plot(emg_final,color='black',alpha=0.7)
		# plt.plot(emg_chans[0,:],color='blue',alpha=0.5)
		# plt.plot(emg_chans[1,:],color='red',alpha=0.5)
		# plt.plot(emg_chans[2,:],color='green',alpha=0.5)
		# plt.plot(emg_chans[3,:],color='yellow',alpha=0.5)
		# plt.savefig('/media/HlabShare/SamB/emg_from_data_test.pdf',dpi=300)
		# plt.show()

		np.save(LFP_dir + f'emg_from_lfp_hr{int((fil+12)/12)}.npy', full_selected_emg)



