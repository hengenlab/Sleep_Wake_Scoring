#from __future__ import print_function
import os
#os.chdir('/media/HlabShare/Lizzie_Work/Remote_Git')
#print('moving to Remote_Git')
from scipy.signal import savgol_filter
import numpy as np
import scipy.signal as signal
from scipy.integrate import simps
import matplotlib.patches as patch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import load_intan_rhd_format_hlab as intan
import neuraltoolkit as ntk
import seaborn as sns
import sys
import time as timer
import glob
from sklearn.decomposition import PCA
from samb_work import videotimestamp
from samb_work import DLCMovement_input
import psutil
import cv2
import sys

def check1(h5files):
# Checks to make sure that all of the h5 files are the same size
	sizes = [os.stat(i).st_size for i in h5files]
	if np.size(np.unique(sizes))>1:
		sys.exit('Not all of the h5 files are the same size')
def check2(files):
	timestamps = [files[i][79:88] for i in np.arange(np.size(files))]
	for i in np.arange(np.size(files)-1):
		hr1 = timestamps[i+1][0:4]
		hr2 = timestamps[i][5:9]
		if hr2 != hr1:
			sys.exit('hour '+str(i) + ' is not continuous with hour ' + str(i+1))
def check3(h5files, vidfiles):
	timestamps_h5 = [h5files[i][79:88] for i in np.arange(np.size(h5files))]
	timestamps_vid = [vidfiles[i][79:88] for i in np.arange(np.size(h5files))]
	if timestamps_h5 != timestamps_vid:
		sys.exit('h5 files and video files not aligned')
# Checks to make sure that all of the h5 files are continuous
digi_dir = '/media/bs002r/SCF0405/cam_2018-12-05_16-18-15'
motion_dir = '/media/HlabShare/Lizzie_Work/LIT_dlc/SCF00004/Analyzed_Videos/'
rawdat_dir = '/media/bs002r/SCF0405/SCF00004_2018-12-05_16-15-10/'
os.chdir(digi_dir)
stmp = videotimestamp.vidtimestamp('Digital_64_Channels_int64_2018-12-05_16-18-15.bin')
#stmp = (num-1)*3600*1000*1000*1000  
h5 = sorted(glob.glob(motion_dir+'*.h5'))
vidfiles = sorted(glob.glob(motion_dir+'*labeled.mp4'))

move_flag = input('Have you already created aligned movement arrays? (y/n): ')
num = int(input('What hour are you starting on?: '))

os.chdir(rawdat_dir)
files = sorted(glob.glob('*.bin'))

if move_flag == 'n':
	check1(h5)
	check2(h5)
	check2(vidfiles)
	check3(h5, vidfiles)

	leng = []
	which_vid = []
	frame = []
	for a in np.arange(np.size(vidfiles)):
	    videofilename = vidfiles[a]
	    lstream = 0
	    # get video attributes
	    v = ntk.NTKVideos(videofilename, lstream)
	    which_vid.append(np.full((1,int(v.length)), videofilename[-96:])[0])
	    leng.append(v.length)
	    frame.append(np.arange(int(v.length)))
	leng = np.array(leng)
	which_vid = np.concatenate(which_vid)  
	frame = np.concatenate(frame)
	#posi = np.cumsum(leng)
	#frame = np.arange(np.sum(leng))


	time, dat = ntk.load_raw_binary(files[0],64)
	offset = (stmp-time[0])
	alignedtime = (1000*1000*1000)*np.arange(np.sum(leng))/15 + offset
	# Time in column 2 is in nanoseconds and time in column 3 is in hours 
	mot_vect = []
	basenames = []

	for i in np.arange(np.size(h5)):
		b = h5[i]
		basename = DLCMovement_input.get_movement(b, first_vid = 0)
		vect = np.load(basename+'_full_movement_trace.npy')
		if np.size(vect)>leng[i]:
			#print('removing one nan')
			vect = vect[0:-1]
		#print(np.size(vect))
		mot_vect.append(vect)
		basenames.append(basename)
	mot_vect = np.concatenate(mot_vect)
	dt = alignedtime[1]-alignedtime[0]
	n_phantom_frames = int(round(offset/dt))


	phantom_frames = np.zeros(n_phantom_frames)
	phantom_frames[:] = np.nan 
	novid_time = np.arange(dt, alignedtime[0], dt)

	corrected_motvect = np.concatenate([phantom_frames, mot_vect])
	corrected_frames  = np.concatenate([phantom_frames, frame])
	full_alignedtime = np.concatenate([novid_time, alignedtime])
	which_vid = np.concatenate([np.full((1, np.size(phantom_frames)), 'no video yet')[0], which_vid])


	aligner = np.column_stack((full_alignedtime,full_alignedtime/(1000*1000*1000*3600), corrected_motvect))
	#video_aligner = np.column_stack((corrected_frames, which_vid))

	reorganized_mot = []
	nhours = int(aligner[-1,1])
	for h in np.arange(num, nhours):
		tmp_idx = np.where((aligner[:,1]>(h)) & (aligner[:,1]<(h+1)))[0]      
		time_move = (np.vstack((aligner[tmp_idx, 2], aligner[tmp_idx,1])))
		video_key = (np.vstack((full_alignedtime[tmp_idx], which_vid[tmp_idx])))
		np.save(motion_dir+basenames[h]+'_tmove.npy', time_move)
		np.save(motion_dir+basenames[h]+'_vidkey.npy', video_key)


os.chdir(rawdat_dir)
filesindex = np.arange((num*12),np.size(files),12)

cort = input('Cortical screw? (y/n): ')
EEG = int(input('Enter EEG channel: ')) - 1
EMGinput = int(input('Enter EMG channel (0 if using motion): ')) - 1
HS = input('Enter array type (hs64, eibless64): ')
#reclen = int(input('Enter recording length in seconds: ')) #recording length in seconds
reclen = 3600

if HS == 'hs64':
    chan_map = np.array([26, 30, 6,  2,  18, 22, 14, 10, 12, 16, 8,  4,
                         28, 32, 24, 20, 48, 44, 36, 40, 64, 60, 52, 56,
                         54, 50, 42, 46, 62, 58, 34, 38, 39, 35, 59, 63,
                         47, 43, 51, 55, 53, 49, 57, 61, 37, 33, 41, 45,
                         17, 21, 29, 25, 1,  5,  13, 9,  11, 15, 23, 19,
                          3,  7,  31, 27]) - 1
elif HS == 'eibless64':
    chan_map = np.array([1,  5,  9,  13, 3,  7,  11, 15, 17, 21, 25, 29,
                         19, 23, 27, 31, 33, 37, 41, 45, 35, 39, 43, 47,
                         49, 53, 57, 61, 51, 55, 59, 63, 2,  6,  10, 14,
                         4,  8,  12, 16, 18, 22, 26, 30, 20, 24, 28, 32,
                         34, 38, 42, 46, 36, 40, 44, 48, 50, 54, 58, 62,
                         52, 56, 60, 64]) - 1
EEG2 = EEG + 1
EEG2 = np.where(chan_map==EEG2)
EEG3 = EEG + 2
EEG3 = np.where(chan_map==EEG3)
EEG = np.where(chan_map==EEG)
EEG = EEG[0][0]
if EMGinput!=-1:
	EMG = np.where(chan_map==EMGinput)
	EMG = EMG[0][0]
else:
	EMG = 0
tic = timer.time()

#time, dat = ntk.ntk_ecube.load_raw_binary(files[0], 64)

for fil in filesindex:
	print('STARTING LOOP: '+str(fil))
	print('THIS IS THE STARTING USAGE: ' + str(psutil.virtual_memory()))
	start_label = files[fil][29:-4]
	end_label = files[fil+12][29:-4]
	# THIS CAN BE CONSOLIDATED INTO A FEW LINES LATER. THIS IS WHEN SAM IS LEARNING...
	# start importing your data

	# select the 12 files that you will work on at a time. this is to prevent
	# overloading RAM; since the default is to write 5 minute files, this will
	# effectively load an hour's worth of data at a time
	load_files = files[fil:fil+12]

	time 	= []
	dat 	= []
	dat2	= []
	eeg 	= []
	emg 	= []

	print('Working on hour ' + str(int((fil+12)/12)))

	for a in np.arange(0,np.size(load_files)):

		#print('This is usage at step 2: ' + str(psutil.virtual_memory()))
		if cort == 'n':
			print('Importing data from binary file...')
			time, dat 	= ntk.ntk_ecube.load_raw_binary(load_files[a], 64)
			dat = ntk.ntk_channelmap.channel_map_data(dat, 64, 'hs64')

			print('merging file {}'.format(a))
			eeg 		= np.concatenate((eeg,np.average((dat[EEG],np.squeeze(dat[EEG2]),np.squeeze(dat[EEG3])),axis=0)),axis=0)
			if EMGinput != -1:
				emg 		= np.concatenate((emg,dat[EMG]),axis=0)

		else:
			print('Importing data from binary file...')
			time, dat 	= ntk.ntk_ecube.load_raw_binary(load_files[a], 64)
			dat 		= ntk.ntk_channelmap.channel_map_data(dat, 64, 'hs64')

			print('merging file {}'.format(a))
			eeg 		= np.concatenate((eeg,dat[EEG]),axis=0)
			if EMGinput != -1:
				emg 		= np.concatenate((emg,dat[EMG]),axis=0)
	#	print('This is usage at step 3: ' + str(psutil.virtual_memory()))
	#print('This is usage at step 4: ' + str(psutil.virtual_memory()))
	#filters raw data
	print('Filtering and downsampling...')
	fs = 25000 # sample frequency
	nyq = 0.5*fs # nyquist
	N  = 3    # Filter order
	Wn = [0.5/nyq,400/nyq] # Cutoff frequencies
	B, A = signal.butter(N, Wn, btype='bandpass',output='ba')
	datf = signal.filtfilt(B,A, eeg)
	Wn = [10/nyq] # Cutoff frequencies
	B, A = signal.butter(N, Wn, btype='highpass',output='ba')
	if EMGinput != -1:
		emg = signal.filtfilt(B,A, emg)
		EMGabs = abs(emg)
		disp = 25000*reclen - np.size(EMGabs)
		if disp > 0:
			EMGabs = np.pad(EMGabs, (0,disp), 'constant')
		elif disp < 0:
			EMGabs = EMGabs[0:reclen*25000]
		EMGabs = EMGabs.reshape(-1,6250)
		EMGi = np.trapz(EMGabs, x=None, dx=1.0, axis=-1)
		EMGs = savgol_filter(EMGi, 51, 3) # window size 51, polynomial order 3
		EMGamp = EMGs
		#EMGamp = (EMGs - np.average(EMGs))/np.std(EMGs)
		EMGfor = np.zeros(int(np.size(emg)/(4*fs)))
		for i in np.arange(np.size(EMGfor)):
			EMGfor[i] = np.var(emg[4*fs*(i):(4*fs*(i+1))])
		#	print (i)
		EMGfor = (EMGfor - np.average(EMGfor))/np.std(EMGfor)
		#np.save('EMGfor' + str(int((fil+12)/12)) + '.npy',EMGfor)
	elif EMGinput == -1:
		EMGamp = mot[int(3600*(fil/12)):int((3600*fil/12+3600))]
	#downsample the data for LFP
	finalfs = 200
	R = fs/finalfs
	print(int(R*np.size(datf)))
	#downdatlfp = np.zeros(int(R*np.size(datf)))
	downsamp = np.zeros(fs*reclen)
	downsamp = datf[0:fs*reclen]
	disp = fs*reclen - np.size(downsamp)
	if disp > 0:
		downsamp = np.pad(downsamp, (0,disp), 'constant')
	downsamp = downsamp.reshape(-1,int(R))
	downdatlfp = downsamp[:,1]
	dat = []
	np.save('EEGhr' + str(int((fil+12)/12)),downdatlfp)
	np.save('EMGhr' + str(int((fil+12)/12)),EMGamp)
	print('Calculating bandpower...')
	#print('This is usage at step 5: ' + str(psutil.virtual_memory()))
	fsd = 200
	f, t_spec, x_spec = signal.spectrogram(downdatlfp, fs=fsd, window='hanning', nperseg=1000, noverlap=1000-1, mode='psd')
	fmax = 64
	fmin = 1
	x_mesh, y_mesh = np.meshgrid(t_spec, f[(f<fmax) & (f>fmin)])
	plt.figure(figsize=(16,2))
	p1 = plt.pcolormesh(x_mesh, y_mesh, np.log10(x_spec[(f<fmax) & (f>fmin)]), cmap='jet')
	#print('This is usage at step 5: ' + str(psutil.virtual_memory()))
	#plt.savefig('spect2.jpg')
	del(p1)
	fsemg = 4
	if EMGinput == -1:
		fsemg = 1
	realtime = np.arange(np.size(EMGamp))/fsemg
	plt.plot(realtime,(EMGamp - np.nanmean(EMGamp))/np.nanstd(EMGamp))
	plt.ylim(1,64)
	plt.xlim(0,3600)
	plt.yscale('log')
	plt.title('Time Period: ' + start_label + '-' + end_label)
	plt.savefig('specthr' + str(int((fil+12)/12)) + '.jpg')
	plt.cla()
	plt.clf()
	plt.close('all')
	delt = sum(x_spec[np.where(np.logical_and(f>=1,f<=4))])
	thetw = sum(x_spec[np.where(np.logical_and(f>=2,f<=16))])
	thetn = sum(x_spec[np.where(np.logical_and(f>=5,f<=10))])
	thet = thetn/thetw
	delt = (delt-np.average(delt))/np.std(delt)
	thet = (thet-np.average(thet))/np.std(thet)
	dispt = 4*reclen - np.size(thet)
	dispd = 4*reclen - np.size(thet)
	if dispt > 0:
		thet = np.pad(thet, (0,dispt), 'constant')
	if dispd > 0:
		delt = np.pad(delt, (0,dispd), 'constant')
	del(x_spec)
	del(f)
	del(t_spec)
	print('THIS IS THE USAGE AT THE END OF LOOP: ' + str(psutil.virtual_memory()))
	#np.save('delt' + str(int((fil+12)/12)) + '.npy',delt)
	#np.save('thet' + str(int((fil+12)/12)) + '.npy',thet)
