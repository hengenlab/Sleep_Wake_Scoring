#from __future__ import print_function
from scipy.signal import savgol_filter
import numpy as np
import scipy.signal as signal
from scipy.integrate import simps
import matplotlib.patches as patch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import copy
#import load_intan_rhd_format_hlab as intan
import neuraltoolkit as ntk
import seaborn as sns
import sys
import time as timer
import os
from lizzie_work import DLCMovement_input
import cv2

#for integrating random forest into this
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from joblib import dump, load
import pandas as pd
def create_new_df(features):
	#features = ['Animal_Name', 'Time_Interval','delta_pre','EEGdelta','theta_pre','EEGtheta','EEGalpha','EEGbeta','EEGgamma','EEGamp','Motion']
	SleepStateAtt = pd.DataFrame(columns = features)
	return SleepStateAtt
# def add_new_model_feature(dataframe):
# 	# make this function at some point
def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier(n_estimators=200)
    clf.fit(features, target)
    return clf
def press(event):
	if event.key == '1':
		State[int(i)] = 1
		print('coded Awake')
	else:
		if event.key == '2':
			State[int(i)] = 2
			print('coded NREM')
		else:
			if event.key == '3':
				State[int(i)] = 3
				print('coded REM')
			else:
				State[int(i)] = 4
				print('code invalid, or other')
	cv2.destroyAllWindows()
sys.stdout.flush()

def check1(h5files):
# Checks to make sure that all of the h5 files are the same size
	sizes = [os.stat(i).st_size for i in h5files]
	if np.size(np.unique(sizes))>1:
		sys.exit('Not all of the h5 files are the same size')
def check2(files):
	timestamps = [files[i][81:90] for i in np.arange(np.size(files))]
	for i in np.arange(np.size(files)-1):
		hr1 = timestamps[i+1][0:4]
		hr2 = timestamps[i][5:9]
		if hr2 != hr1:
			sys.exit('hour '+str(i) + ' is not continuous with hour ' + str(i+1))
def check3(h5files, vidfiles):
	timestamps_h5 = [h5files[i][81:90] for i in np.arange(np.size(h5files))]
	timestamps_vid = [vidfiles[i][81:90] for i in np.arange(np.size(h5files))]
	if timestamps_h5 != timestamps_vid:
		sys.exit('h5 files and video files not aligned')

motion_dir = '/Volumes/HlabShare/Lizzie_Work/LIT_dlc/SCF00005/Analyzed_Videos/'
rawvid_dir = '/Volumes/rawdata-3/Watchtower/SCF00005/'
rawdat_dir = '/Volumes/rawdata-2/SCF0405/SCF00005_2018-12-05_16-17-34/'


os.chdir(rawdat_dir)
animal = input('What animal is this?')
hr  = input('What hour are you working on? (starts at 1): ')
epochlen = int(input('Epoch length: '))
fs = int(input('sampling rate: '))
delt = np.load('delt' + hr + '.npy')
thet = np.load('thet' + hr + '.npy')
downdatlfp = np.load('EEGhr' + hr + '.npy')
movement_files = np.sort(glob.glob(motion_dir+'*tmove.npy'))
check2(movement_files)
vidkey_files = np.sort(glob.glob(motion_dir+'*vidkey.npy'))
check2(vidkey_files)
check3(movement_files, vidkey_files)
#EMGamp = np.load('EMG_EAB26.npy')
emg = input('Do you have emg info? y/n: ')
if emg == 'y':
	EMGamp = np.load('EMGhr' + hr + '.npy')
	EMGamp = (EMGamp-np.average(EMGamp))/np.std(EMGamp)


pos = input('Do you have a motion vector? y/n: ')

if pos == 'y':
	movement = np.load(movement_files[int(hr)-1])
	video_key = np.load(vidkey_files[int(hr)-1])
	time = movement[1]
	time_sec = time*3600
	dt = time_sec[2]-time_sec[1]
	dxy = movement[0]
	binsz = int(round(1/dt))

	rs_dxy = np.reshape(dxy,[int(np.size(dxy)/binsz), binsz])
	time_min = np.linspace(0, 60, np.size(dxy))

	med = np.median(rs_dxy, axis = 1)
	binned_dxy = np.mean(rs_dxy, axis = 1)
	hist = np.histogram(med[~np.isnan(med)], bins = 1000)
	csum = np.cumsum(hist[0])
	th = np.size(med)*0.95
	outliers_idx = np.where(csum>th)[0][0]
	outliers = np.where(med>hist[1][outliers_idx])[0]

	for i in outliers:
		med[i] = med[i-1]
		a = i-1
		while med[i] > hist[1][outliers_idx]:
			a = i-1
			med[i] = med[a]
	binned_mot = np.nanmean(np.reshape(med, (900, 4)), axis = 1)

ratio2 = 12*4

model = input('Use a random forest?: ')
satisfaction = []

#####test
if model == 'y':
	mod_name = input('Which model? (young rat, adult rat, mouse)')
	os.chdir('/Volumes/HlabShare/Sleep_Model/')
	clf = load('NewModelTest.joblib')
	os.chdir(rawdat_dir) 
	FullFeaturesTrain = np.empty((0,9))
	FullFeaturesTest = np.empty((0,9))
	fs = 200

	#Generate average/max EEG amplitude, EEG frequency, EMG amplitude for each bin

	print('Generating EEG vectors...')
	epochlen = 4
	bin = 4 #bin size in seconds
	binl = bin * fs #bin size in array slots
	EEGamp = np.zeros(int(np.size(downdatlfp)/(4*fs)))
	EEGmean = np.zeros(int(np.size(downdatlfp)/(4*fs)))
	for i in np.arange(np.size(EEGamp)):
		EEGamp[i] = np.var(downdatlfp[4*fs*(i):(4*fs*(i+1))])
		EEGmean[i] = np.mean(np.abs(downdatlfp[4*fs*(i):(4*fs*(i+1))]))
	EEGamp = (EEGamp - np.average(EEGamp))/np.std(EEGamp)

	EEGmax = np.zeros(int(np.size(downdatlfp)/(4*fs)))
	for i in np.arange(np.size(EEGmax)):
		EEGmax[i] = np.max(downdatlfp[4*fs*(i):(4*fs*(i+1))])
	EEGmax = (EEGmax - np.average(EEGmax))/np.std(EEGmax)
	

	fse = 4
	EMG = np.zeros(int(np.size(EMGamp)/(4*fse)))
	for i in np.arange(np.size(EMG)):
		EMG[i] = np.average(EMGamp[4*fse*(i):(4*fse*(i+1))])

	#Calculate delta and theta bandpower for entire dataset
	# Define window length (4 seconds)
	print('Extracting delta bandpower...')
	win = 4 * fs
	EEGdelta = np.zeros(int(np.size(downdatlfp)/(4*fs)))
	#Vectorized bandpower calculation
	EEGreshape = np.reshape(downdatlfp,(-1,fs*epochlen))
	win = 4*fs
	freqs, psd = signal.welch(EEGreshape, fs, nperseg=win, scaling='density')
	low, high = 0.5, 4
	idx_min = np.argmax(freqs > low) - 1
	idx_max = np.argmax(freqs > high) - 1
	idx_delta = np.zeros(dtype=bool, shape=freqs.shape)
	idx_delta[idx_min:idx_max] = True
	EEGdelta = simps(psd[:,idx_delta], freqs[idx_delta])

	print('Extracting theta bandpower...')
	EEGtheta = np.zeros(int(np.size(downdatlfp)/(4*fs)))
	win = 4*fs
	freqs, psd = signal.welch(EEGreshape, fs, nperseg=win, scaling='density')
	low, high = 4, 8
	idx_min = np.argmax(freqs > low) - 1
	idx_max = np.argmax(freqs > high) - 1
	idx_theta = np.zeros(dtype=bool, shape=freqs.shape)
	idx_theta[idx_min:idx_max] = True
	EEGtheta = simps(psd[:,idx_theta], freqs[idx_theta])

	delt_thet = EEGdelta/EEGtheta
	delt_thet = (delt_thet - np.average(delt_thet))/np.std(delt_thet)

	EEGdelta = (EEGdelta - np.average(EEGdelta))/np.std(EEGdelta)



	print('Extracting alpha bandpower...')
	EEGalpha = np.zeros(int(np.size(downdatlfp)/(4*fs)))
	win = 4*fs
	freqs, psd = signal.welch(EEGreshape, fs, nperseg=win, scaling='density')
	low, high = 8, 12
	idx_min = np.argmax(freqs > low) - 1
	idx_max = np.argmax(freqs > high) - 1
	idx_alpha = np.zeros(dtype=bool, shape=freqs.shape)
	idx_alpha[idx_min:idx_max] = True
	EEGalpha = simps(psd[:,idx_alpha], freqs[idx_alpha])
	EEGalpha = (EEGalpha - np.average(EEGalpha))/np.std(EEGalpha)

	print('Extracting beta bandpower...')
	EEGbeta = np.zeros(int(np.size(downdatlfp)/(4*fs)))
	win = 4*fs
	freqs, psd = signal.welch(EEGreshape, fs, nperseg=win, scaling='density')
	low, high = 12, 30
	idx_min = np.argmax(freqs > low) - 1
	idx_max = np.argmax(freqs > high) - 1
	idx_beta = np.zeros(dtype=bool, shape=freqs.shape)
	idx_beta[idx_min:idx_max] = True
	EEGbeta = simps(psd[:,idx_beta], freqs[idx_beta])
	EEGbeta = (EEGbeta - np.average(EEGbeta))/np.std(EEGbeta)

	print('Extracting gamma bandpower...')
	EEGgamma = np.zeros(int(np.size(downdatlfp)/(4*fs)))
	win = 4*fs
	freqs, psd = signal.welch(EEGreshape, fs, nperseg=win, scaling='density')
	low, high = 30, 80
	idx_min = np.argmax(freqs > low) - 1
	idx_max = np.argmax(freqs > high) - 1
	idx_gamma = np.zeros(dtype=bool, shape=freqs.shape)
	idx_gamma[idx_min:idx_max] = True
	EEGgamma = simps(psd[:,idx_gamma], freqs[idx_gamma])
	EEGgamma = (EEGgamma - np.average(EEGgamma))/np.std(EEGgamma)

	print('Extracting narrow-band theta bandpower...')
	EEG_broadtheta = np.zeros(int(np.size(downdatlfp)/(4*fs)))
	win = 4*fs
	freqs, psd = signal.welch(EEGreshape, fs, nperseg=win, scaling='density')
	low, high = 2, 16
	idx_min = np.argmax(freqs > low) - 1
	idx_max = np.argmax(freqs > high) - 1
	idx_broadtheta = np.zeros(dtype=bool, shape=freqs.shape)
	idx_broadtheta[idx_min:idx_max] = True
	EEG_broadtheta = simps(psd[:,idx_broadtheta], freqs[idx_broadtheta])
	EEGnb = EEGtheta/EEG_broadtheta
	EEGnb= (EEGnb - np.average(EEGnb))/np.std(EEGnb)
	EEGtheta = (EEGtheta - np.average(EEGtheta))/np.std(EEGtheta)


	print('Boom. Boom. FIYA POWER...')
	EEGfire = np.zeros(int(np.size(downdatlfp)/(4*fs)))
	win = 4*fs
	freqs, psd = signal.welch(EEGreshape, fs, nperseg=win, scaling='density')
	low, high = 4, 20
	idx_min = np.argmax(freqs > low) - 1
	idx_max = np.argmax(freqs > high) - 1
	idx_fire = np.zeros(dtype=bool, shape=freqs.shape)
	idx_fire[idx_min:idx_max] = True
	EEGfire = simps(psd[:,idx_fire], freqs[idx_fire])
	EEGfire = (EEGfire - np.average(EEGfire))/np.std(EEGfire)


	delt_thet = EEGdelta/EEGtheta

	delta_post = np.append(EEGdelta,0)
	delta_post = np.delete(delta_post,0)
	delta_pre = np.append(0,EEGdelta)
	delta_pre = delta_pre[0:-1]

	theta_post = np.append(EEGtheta,0)
	theta_post = np.delete(theta_post,0)
	theta_pre = np.append(0,EEGtheta)
	theta_pre = theta_pre[0:-1]

	delta_post2 = np.append(delta_post,0)
	delta_post2 = np.delete(delta_post2,0)
	delta_pre2 = np.append(0,delta_pre)
	delta_pre2 = delta_pre2[0:-1]

	theta_post2 = np.append(theta_post,0)
	theta_post2 = np.delete(theta_post2,0)
	theta_pre2 = np.append(0,theta_pre)
	theta_pre2 = theta_pre2[0:-1]

	delta_post3 = np.append(delta_post2,0)
	delta_post3 = np.delete(delta_post3,0)
	delta_pre3 = np.append(0,delta_pre2)
	delta_pre3 = delta_pre3[0:-1]

	theta_post3 = np.append(theta_post2,0)
	theta_post3 = np.delete(theta_post3,0)
	theta_pre3 = np.append(0,theta_pre2)
	theta_pre3 = theta_pre3[0:-1]

	nb_post = np.append(EEGnb,0)
	nb_post = np.delete(nb_post,0)
	nb_pre = np.append(0,EEGnb)
	nb_pre = nb_pre[0:-1]
	

	FeatureList = [delta_pre, delta_pre2,delta_pre3,delta_post,delta_post2,delta_post3,EEGdelta,theta_pre,theta_pre2,theta_pre3,theta_post,theta_post2,theta_post3,
	EEGtheta,EEGalpha,EEGbeta,EEGgamma,EEGnb,nb_pre,delt_thet,EEGfire,EEGamp,EEGmax,EEGmean,EMG]
	Features = np.column_stack((FeatureList))

	Predict_y = clf.predict(Features) 

	plt.ion()
	figy = plt.figure(figsize=(11,6))
	ax1 = plt.subplot2grid((2, 1), (0, 0))
	plt.title('Spectrogram w/ EMG')
	img=mpimg.imread(rawdat_dir+'specthr'+ hr + '.jpg')
	imgplot = plt.imshow(img,aspect='auto')
	plt.xlim(199,1441)
	plt.ylim(178,24)
	ax1.set_xticks(np.linspace(199,1441, 13))
	ax1.set_xticklabels(np.arange(0, 65, 5))
	ticksy = [35,100,150]
	labelsy = [60,6,2]
	plt.yticks(ticksy, labelsy)
	ax2 = plt.subplot2grid((2, 1), (1, 0))
	plt.title('Predicted States')
	plt.ion()
	for state in np.arange(np.size(Predict_y)):
		if Predict_y[state] == 0:
			rect7 = patch.Rectangle((state,0),3.8,height=1,color='green')
			ax2.add_patch(rect7)
		elif Predict_y[state] == 2:
			rect7 = patch.Rectangle((state,0),3.8,height=1,color='blue')
			ax2.add_patch(rect7)
		elif Predict_y[state] == 5:
			rect7 = patch.Rectangle((state,0),3.8,height=1,color='red')
			ax2.add_patch(rect7)
	plt.ylim(0.3,1)
	plt.xlim(0,900)
	Predictions  = clf.predict_proba(Features)  
	predConf = np.max(Predictions,1)
	plt.plot(predConf, color= 'k')

	plt.tight_layout()
	plt.show()
	satisfaction = input('Satisfied?: ')

if satisfaction == 'y':
	filename = input('File name: ')
	np.save(filename,Predict_y)
	exit()

#####
#end random forest test

print('Ready for scoring, opening GUI.')
#for scoring
plt.ion()
State = np.zeros(int(np.size(downdatlfp)/(4*fs)))
realtime = np.arange(np.size(downdatlfp))/fs
if emg == 'y':
	EMGamp = np.pad(EMGamp, (0,100), 'constant')
if emg == 'n':
	EMGamp = False
vid_samp = np.linspace(0, 3600*fs, np.size(video_key[0]))
for i in np.arange(np.size(downdatlfp)/(fs*4)-4):
	if model == 'y':
		Prediction = clf.predict(Features[int(i),:].reshape(1,-1))
		if Prediction == 0:
			Prediction='Wake'  
		elif Prediction == 2:
			Prediction='NREM'
		elif Prediction == 5:
			Prediction='REM'
		Predictions  = clf.predict_proba(Features[int(i),:].reshape(1,-1))  
		predConf = np.max(Predictions,1)
	print(i)
	clicks = []
	start = int(i*fs*epochlen)
	end = int(i*fs*epochlen+fs*3*epochlen)
	vid_win_idx = np.where(np.logical_and(vid_samp>=start, vid_samp<end))[0]
	vid_win = video_key[2][vid_win_idx]
	if np.size(np.where(vid_win == 'nan')[0])>0:
		print('There is no video here')
	else:
		vid_win = [int(float(i)) for i in vid_win]
		if np.size(np.unique(video_key[1][vid_win_idx]))>1:
			print('This period is between two windows. No video to display')
		else:
			vidfilename =  np.unique(video_key[1][vid_win_idx])[0]
			score_win = np.arange(int(vid_win[0]) + int(np.size(vid_win)/3), int(vid_win[0]) + int((np.size(vid_win)/3)*2))
	x = (end-start)/ratio2
	length = np.arange(int(end/x-start/x))
	bottom = np.zeros(int(end/x-start/x))
	if i == 0:
		fig1 = plt.figure(figsize=(11,6))
		ax1 = plt.subplot2grid((4, 1), (0, 0))
		line1, = ax1.plot(realtime[start:end],downdatlfp[start:end])
		plt.xlim(start/fs,end/fs)
		plt.title('LFP')
		plt.ylim(-5000,5000)
		bot = ax1.get_ylim()[0]
		rect = patch.Rectangle((start/fs+4,bot),4,height=-bot/5)
		ax1.add_patch(rect)
		ax2 = plt.subplot2grid((4, 1), (1, 0))
		line2, = ax2.plot(delt[start:end])
		plt.xlim(0,end-start)
		plt.ylim(np.min(delt),np.max(delt)/3)
		bot2 = ax2.get_ylim()[0]
		rect2 = patch.Rectangle((fs*4,bot2),fs*4,height=float(-bot2/5))
		ax2.add_patch(rect2)
		plt.title('Delta power (0.5 - 4 Hz)')
		ax3 = plt.subplot2grid((4, 1), (2, 0))
		line3, = ax3.plot(thet[start:end])
		plt.xlim(0,end-start)
		plt.ylim(np.min(thet),np.max(thet)/3)
		plt.title('Theta power (4 - 8 Hz)')
		bot3 = ax3.get_ylim()[0]
		rect3 = patch.Rectangle((fs*4,bot3),fs*4,height=-bot3/5)
		ax3.add_patch(rect3)
		ax5 = plt.subplot2grid((4, 1), (3, 0))
		if EMGamp.any() == False:
			plt.text(0.5, 0.5, 'There is no EMG')
		else:
			plt.fill_between(length,bottom,EMGamp[int(i*4*epochlen):int(i*4*epochlen)+int(end/x-start/x)],color='red')
			plt.title('EMG power')
			plt.xlim(0,int((end-start)/x)-1)
			plt.ylim(-1,5)

		plt.tight_layout()

		fig2 = plt.figure(figsize = (11,8))
		ax6 = plt.subplot2grid((7, 1), (0, 0), rowspan = 2)
		rect6 = patch.Rectangle((4.1,0),3.8,height=2)
		ax6.add_patch(rect6)
		plt.title('States')
		if model == 'y':
			t1 = plt.text(1800,1,str(Prediction))
			t2 = plt.text(1800,0.75,str(predConf))
		ax6.set_xlim(0,3600)
		ax6.set_xticks(np.linspace(0,3600, 13))
		ax6.set_xticklabels(np.arange(0, 65, 5))
		plt.ylim(0.5,2)
		fig2.canvas.mpl_connect('key_press_event', press)
		ax7= plt.subplot2grid((7, 1), (2, 0),rowspan=3)
		img=mpimg.imread(rawdat_dir+'specthr'+ hr + '.jpg')
		imgplot = plt.imshow(img,aspect='auto')
		plt.xlim(199,1441)
		plt.ylim(178,0)
		ax7.set_xticks(np.linspace(199,1441, 13))
		ax7.set_xticklabels(np.arange(0, 65, 5))
		ticksy = [50,150]
		labelsy = [10,2.5]
		plt.yticks(ticksy, labelsy)

		ax8= plt.subplot2grid((7, 1), (5, 0), rowspan = 2)
		x_vals = np.linspace(0,60,np.size(med))
		plt.plot(x_vals, med)
		ax8.set_xlim(0, 60)
		ax8.set_xticks(np.linspace(0,60, 13))
		title_idx = [movement_files[int(hr)-1].find('e3v'), movement_files[int(hr)-1].find('DeepCut')]
		plt.title(movement_files[int(hr)-1][title_idx[0]:title_idx[1]])
		sorted_med = np.sort(med)
		idx = np.where(sorted_med>int(max(sorted_med)*0.05))[0][0]

		if idx == 0:
			thresh = sorted_med[idx] 
		#print(int(max(sorted_med)*0.50))
		else:
			thresh = np.nanmean(sorted_med[0:idx])

		moving = np.where(dxy > thresh)[0]
		h = plt.gca().get_ylim()[1]
		# consec = group_consecutives(np.where(med > thresh)[0])
		consec = DLCMovement_input.group_consecutives(np.where(med > thresh)[0])
		for vals in consec:
			if len(vals)>5:
				x = x_vals[vals[0]]
				#x = time_min[vals[0]]
				y = 0
				width = x_vals[vals[-1]]-x
				#width = time_min[vals[-1]]-x
				rect = patch.Rectangle((x,y), width, h, color = '#b7e1a1', alpha = 0.5)
				ax8.add_patch(rect)

		plt.show()
		plt.xlim([0,60])
		#plt.title("Find this video at: " + vidfile)

		plt.tight_layout()
		plt.show()
		keyboardClick=None
		while keyboardClick != True:
			keyboardClick=plt.waitforbuttonpress()
			if keyboardClick == False:
				print('pulling up video: '+ vidfilename)
				cap = cv2.VideoCapture(motion_dir +vidfilename)
				 
				# Check if camera opened successfully
				if (cap.isOpened()== False): 
				  print("Error opening video stream or file")
				for f in np.arange(vid_win[0], vid_win[-1]):
					cap.set(1, f)
					ret, frame = cap.read()
					if ret == True:
				 
				    # Display the resulting frame
						if f in score_win:
							cv2.putText(frame, "SCORE WINDOW",(50, 105),cv2.FONT_HERSHEY_PLAIN,4,(225,0,0), 2)

						cv2.imshow('Frame',frame)
							#cv2.waitKey(int((dt*10e2)/2))
						cv2.waitKey(int((dt*10e2)/4))

				cap.release()
				continue
	elif i == np.size(downdatlfp)/(fs*4) - 4:
		plt.close('all')
		plt.figure()
		print('Scoring done, plotting sleep states.')
		plt.plot(State)
	else:
		line1.set_ydata(downdatlfp[start:end])
		if model == 'y':
			t1.set_text(str(Prediction))
			t2.set_text(str(predConf))
		line2.set_ydata(delt[start:end])
		line3.set_ydata(thet[start:end])
		ax5.collections.clear()
		ax5.fill_between(length,bottom,EMGamp[int(i*4*epochlen):int(i*4*epochlen+4*3*epochlen)],color='red')
		rect6 = patch.Rectangle((start/fs+4.1,0),3.8,height=2)
		ax6.add_patch(rect6)
		if State[int(i-1)] == 1:
			rect7 = patch.Rectangle((start/fs,0),3.8,height=2,color='green')
			ax6.add_patch(rect7)
		elif State[int(i-1)] == 2:
			rect7 = patch.Rectangle((start/fs,0),3.8,height=2,color='blue')
			ax6.add_patch(rect7)
		elif State[int(i-1)] == 3:
			rect7 = patch.Rectangle((start/fs,0),3.8,height=2,color='red')
			ax6.add_patch(rect7)
		else:
			print('Invalid entry, coded as 4')
		fig1.canvas.draw()
		fig1.canvas.flush_events()
		fig2.canvas.draw()
		fig2.canvas.flush_events()
		keyboardClick=None
		while keyboardClick != True:
			keyboardClick=plt.waitforbuttonpress()
			if keyboardClick == False:
				print('pulling up video...')
				cap = cv2.VideoCapture(motion_dir +vidfilename)
				 
				# Check if camera opened successfully
				if (cap.isOpened()== False): 
				  print("Error opening video stream or file")
				for f in np.arange(vid_win[0], vid_win[-1]):
					cap.set(1, f)
					ret, frame = cap.read()
					if ret == True:
				 
				    # Display the resulting frame
						if f in score_win:
							cv2.putText(frame, "SCORE WINDOW",(50, 105),cv2.FONT_HERSHEY_PLAIN,4,(225,0,0), 2)

						cv2.imshow('Frame',frame)
							#cv2.waitKey(int((dt*10e2)/2))
						cv2.waitKey(int((dt*10e2)/4))

				cap.release()
				continue		 
plt.show(block=True)
decision = input('Save sleep states? y/n: ')
if decision == 'y':
	filename = input('File name: ')
	np.save(filename,State)
update = input('Update model?: ')
if update == 'y':
	# os.chdir(rawdat_dir)
	# downdatlfp = np.load('EEGhr' + hr + '.npy')
	# os.chdir('/Volumes/HlabShare/Sleep_Model/')
	# EEGfull = np.load('EEGmerge.npy')
	# StatesFull = np.load('SleepStates.npy')
	# downdatlfp = np.concatenate((EEGfull,downdatlfp))
	# Aligned = np.concatenate((StatesFull,State))
	ymot = input('Use motion?: ')
	yemg = input('Use EMG?: ')
	# if ymot == 'y':
	#     N = 9
	#     if yemg == 'y':
	#         N = 10
	# elif yemg == 'y':
	#     N = 9

	# FullTestY = []
	# FullTrainY = []
	# FullFeaturesTrain = np.empty((0,N))
	# FullFeaturesTest = np.empty((0,N))

	fs = 200

	#Generate average/max EEG amplitude, EEG frequency, EMG amplitude for each bin
	if model == 'n':
		print('Generating EEG vectors...')
		epochlen = 4
		bin = 4 #bin size in seconds
		binl = bin * fs #bin size in array slots
		EEGamp = np.zeros(int(np.size(downdatlfp)/(4*fs)))
		EEGmean = np.zeros(int(np.size(downdatlfp)/(4*fs)))
		for i in np.arange(np.size(EEGamp)):
			EEGamp[i] = np.var(downdatlfp[4*fs*(i):(4*fs*(i+1))])
			EEGmean[i] = np.mean(np.abs(downdatlfp[4*fs*(i):(4*fs*(i+1))]))
		EEGamp = (EEGamp - np.average(EEGamp))/np.std(EEGamp)

		EEGmax = np.zeros(int(np.size(downdatlfp)/(4*fs)))
		for i in np.arange(np.size(EEGmax)):
			EEGmax[i] = np.max(downdatlfp[4*fs*(i):(4*fs*(i+1))])
		EEGmax = (EEGmax - np.average(EEGmax))/np.std(EEGmax)


		fse = 4
		EMG = np.zeros(int(np.size(EMGamp)/(4*fse)))
		for i in np.arange(np.size(EMG)):
			EMG[i] = np.average(EMGamp[4*fse*(i):(4*fse*(i+1))])

		print('Extracting delta bandpower...')
		win = 4 * fs
		EEGdelta = np.zeros(int(np.size(downdatlfp)/(4*fs)))
		#Vectorized bandpower calculation
		EEGreshape = np.reshape(downdatlfp,(-1,fs*epochlen))
		win = 4*fs
		freqs, psd = signal.welch(EEGreshape, fs, nperseg=win, scaling='density')
		low, high = 0.5, 4
		idx_min = np.argmax(freqs > low) - 1
		idx_max = np.argmax(freqs > high) - 1
		idx_delta = np.zeros(dtype=bool, shape=freqs.shape)
		idx_delta[idx_min:idx_max] = True
		EEGdelta = simps(psd[:,idx_delta], freqs[idx_delta])
		

		print('Extracting theta bandpower...')
		EEGtheta = np.zeros(int(np.size(downdatlfp)/(4*fs)))
		win = 4*fs
		freqs, psd = signal.welch(EEGreshape, fs, nperseg=win, scaling='density')
		low, high = 4, 8
		idx_min = np.argmax(freqs > low) - 1
		idx_max = np.argmax(freqs > high) - 1
		idx_theta = np.zeros(dtype=bool, shape=freqs.shape)
		idx_theta[idx_min:idx_max] = True
		EEGtheta = simps(psd[:,idx_theta], freqs[idx_theta])

		delt_thet = EEGdelta/EEGtheta
		delt_thet = (delt_thet - np.average(delt_thet))/np.std(delt_thet)

		EEGdelta = (EEGdelta - np.average(EEGdelta))/np.std(EEGdelta)

		print('Extracting alpha bandpower...')
		EEGalpha = np.zeros(int(np.size(downdatlfp)/(4*fs)))
		win = 4*fs
		freqs, psd = signal.welch(EEGreshape, fs, nperseg=win, scaling='density')
		low, high = 8, 12
		idx_min = np.argmax(freqs > low) - 1
		idx_max = np.argmax(freqs > high) - 1
		idx_alpha = np.zeros(dtype=bool, shape=freqs.shape)
		idx_alpha[idx_min:idx_max] = True
		EEGalpha = simps(psd[:,idx_alpha], freqs[idx_alpha])
		EEGalpha = (EEGalpha - np.average(EEGalpha))/np.std(EEGalpha)

		print('Extracting beta bandpower...')
		EEGbeta = np.zeros(int(np.size(downdatlfp)/(4*fs)))
		win = 4*fs
		freqs, psd = signal.welch(EEGreshape, fs, nperseg=win, scaling='density')
		low, high = 12, 30
		idx_min = np.argmax(freqs > low) - 1
		idx_max = np.argmax(freqs > high) - 1
		idx_beta = np.zeros(dtype=bool, shape=freqs.shape)
		idx_beta[idx_min:idx_max] = True
		EEGbeta = simps(psd[:,idx_beta], freqs[idx_beta])
		EEGbeta = (EEGbeta - np.average(EEGbeta))/np.std(EEGbeta)

		print('Extracting gamma bandpower...')
		EEGgamma = np.zeros(int(np.size(downdatlfp)/(4*fs)))
		win = 4*fs
		freqs, psd = signal.welch(EEGreshape, fs, nperseg=win, scaling='density')
		low, high = 30, 80
		idx_min = np.argmax(freqs > low) - 1
		idx_max = np.argmax(freqs > high) - 1
		idx_gamma = np.zeros(dtype=bool, shape=freqs.shape)
		idx_gamma[idx_min:idx_max] = True
		EEGgamma = simps(psd[:,idx_gamma], freqs[idx_gamma])
		EEGgamma = (EEGgamma - np.average(EEGgamma))/np.std(EEGgamma)

		print('Extracting narrow-band theta bandpower...')
		EEG_broadtheta = np.zeros(int(np.size(downdatlfp)/(4*fs)))
		win = 4*fs
		freqs, psd = signal.welch(EEGreshape, fs, nperseg=win, scaling='density')
		low, high = 2, 16
		idx_min = np.argmax(freqs > low) - 1
		idx_max = np.argmax(freqs > high) - 1
		idx_broadtheta = np.zeros(dtype=bool, shape=freqs.shape)
		idx_broadtheta[idx_min:idx_max] = True
		EEG_broadtheta = simps(psd[:,idx_broadtheta], freqs[idx_broadtheta])
		EEGnb = EEGtheta/EEG_broadtheta
		EEGnb= (EEGnb - np.average(EEGnb))/np.std(EEGnb)
		EEGtheta = (EEGtheta - np.average(EEGtheta))/np.std(EEGtheta)

		print('Boom. Boom. FIYA POWER...')
		EEGfire = np.zeros(int(np.size(downdatlfp)/(4*fs)))
		win = 4*fs
		freqs, psd = signal.welch(EEGreshape, fs, nperseg=win, scaling='density')
		low, high = 4, 20
		idx_min = np.argmax(freqs > low) - 1
		idx_max = np.argmax(freqs > high) - 1
		idx_fire = np.zeros(dtype=bool, shape=freqs.shape)
		idx_fire[idx_min:idx_max] = True
		EEGfire = simps(psd[:,idx_fire], freqs[idx_fire])
		EEGfire = (EEGfire - np.average(EEGfire))/np.std(EEGfire)



	delta_post = np.append(EEGdelta,0)
	delta_post = np.delete(delta_post,0)
	delta_pre = np.append(0,EEGdelta)
	delta_pre = delta_pre[0:-1]

	theta_post = np.append(EEGtheta,0)
	theta_post = np.delete(theta_post,0)
	theta_pre = np.append(0,EEGtheta)
	theta_pre = theta_pre[0:-1]

	delta_post2 = np.append(delta_post,0)
	delta_post2 = np.delete(delta_post2,0)
	delta_pre2 = np.append(0,delta_pre)
	delta_pre2 = delta_pre2[0:-1]

	theta_post2 = np.append(theta_post,0)
	theta_post2 = np.delete(theta_post2,0)
	theta_pre2 = np.append(0,theta_pre)
	theta_pre2 = theta_pre2[0:-1]

	delta_post3 = np.append(delta_post2,0)
	delta_post3 = np.delete(delta_post3,0)
	delta_pre3 = np.append(0,delta_pre2)
	delta_pre3 = delta_pre3[0:-1]

	theta_post3 = np.append(theta_post2,0)
	theta_post3 = np.delete(theta_post3,0)
	theta_pre3 = np.append(0,theta_pre2)
	theta_pre3 = theta_pre3[0:-1]

	nb_post = np.append(EEGnb,0)
	nb_post = np.delete(nb_post,0)
	nb_pre = np.append(0,EEGnb)
	nb_pre = nb_pre[0:-1]
	
	State[State == 1] = 0
	State[State == 2] = 2
	State[State == 3] = 5

	#Make a data frame with the new information and then concat to OG


	if ymot == 'y':
		movement = np.load(movement_files[int(hr)-1])
		video_key = np.load(vidkey_files[int(hr)-1])
		time = movement[1]
		time_sec = time*3600
		dt = time_sec[2]-time_sec[1]
		dxy = movement[0]
		binsz = int(round(1/dt))

		rs_dxy = np.reshape(dxy,[int(np.size(dxy)/binsz), binsz])
		time_min = np.linspace(0, 60, np.size(dxy))

		med = np.median(rs_dxy, axis = 1)
		binned_dxy = np.mean(rs_dxy, axis = 1)
		hist = np.histogram(med[~np.isnan(med)], bins = 1000)
		csum = np.cumsum(hist[0])
		th = np.size(med)*0.95
		outliers_idx = np.where(csum>th)[0][0]
		outliers = np.where(med>hist[1][outliers_idx])[0]

		for i in outliers:
			med[i] = med[i-1]
			a = i-1
			while med[i] > hist[1][outliers_idx]:
				a = i-1
				med[i] = med[a]
		binned_mot = np.nanmean(np.reshape(med, (900, 4)), axis = 1)

	animal_name = np.full(np.size(delta_pre), animal)
	basename = movement_files[int(hr)-1][title_idx[0]:title_idx[1]]
	time_int = np.full(np.size(animal_name),basename[8:])
	nans = np.zeros(np.size(animal_name))
	nans[:] = np.nan

	if (ymot == 'y' and yemg == 'n'):
		data = np.vstack([animal_name, time_int, State, delta_pre, delta_pre2,delta_pre3,delta_post,delta_post2,delta_post3,EEGdelta,theta_pre,theta_pre2,theta_pre3,theta_post,theta_post2,theta_post3,
		EEGtheta,EEGalpha,EEGbeta,EEGgamma,EEGnb,nb_pre,delt_thet,EEGfire,EEGamp,EEGmax,EEGmean,nans,binned_mot])	

	elif (ymot == 'n' and yemg == 'n'):
		data = np.vstack([animal_name, time_int, State, delta_pre, delta_pre2,delta_pre3,delta_post,delta_post2,delta_post3,EEGdelta,theta_pre,theta_pre2,theta_pre3,theta_post,theta_post2,theta_post3,
		EEGtheta,EEGalpha,EEGbeta,EEGgamma,EEGnb,nb_pre,delt_thet,EEGfire,EEGamp,EEGmax,EEGmean,nans,nans])	

#	    FeatureList = [delta_pre,EEGdelta,theta_pre,EEGtheta,EEGalpha,EEGbeta,EEGgamma,EEGamp,mot]
	elif (ymot == 'n' and yemg == 'y'):
		data = np.vstack([animal_name, time_int, State, delta_pre, delta_pre2,delta_pre3,delta_post,delta_post2,delta_post3,EEGdelta,theta_pre,theta_pre2,theta_pre3,theta_post,theta_post2,theta_post3,
		EEGtheta,EEGalpha,EEGbeta,EEGgamma,EEGnb,nb_pre,delt_thet,EEGfire,EEGamp,EEGmax,EEGmean,EMG,nans])		
#	    FeatureList = [delta_pre,EEGdelta,theta_pre,EEGtheta,EEGalpha,EEGbeta,EEGgamma,EEGamp,EMG]
	elif(ymot == 'y' and yemg == 'y'):
		data = np.vstack([animal_name, time_int, State, delta_pre, delta_pre2,delta_pre3,delta_post,delta_post2,delta_post3,EEGdelta,theta_pre,theta_pre2,theta_pre3,theta_post,theta_post2,theta_post3,
		EEGtheta,EEGalpha,EEGbeta,EEGgamma,EEGnb,nb_pre,delt_thet,EEGfire,EEGamp,EEGmax,EEGmean,EMG,binned_mot])	
#		FeatureList = [delta_pre,EEGdelta,theta_pre,EEGtheta,EEGalpha,EEGbeta,EEGgamma,EEGamp,EMG,mot]
	FeatureList = ['Animal_Name', 'Time_Interval','State','delta_pre','delta_pre2','delta_pre3','delta_post','delta_post2','delta_post3','EEGdelta','theta_pre','theta_pre2','theta_pre3','theta_post',
	'theta_post2','theta_post3','EEGtheta','EEGalpha','EEGbeta','EEGgamma','EEGnarrow','nb_pre','delta/theta','EEGfire','EEGamp','EEGmax','EEGmean','EMG', 'Motion']
	#Features = np.column_stack((FeatureList))
	#add delta-theta ratio
	#narrowband theta

	df_additions = pd.DataFrame(columns = FeatureList, data = data.T)

	try:
		Sleep_Model = np.load('/Volumes/HlabShare/Sleep_Model/' + mod_name + '_model.pkl')
		Sleep_Model = Sleep_Model.append(df_additions)
	except FileNotFoundError:
		print('no model created...I will save this one')
		df_additions.to_pickle('/Volumes/HlabShare/Sleep_Model/'+mod_name + '_model.pkl')
		Sleep_Model = df_additions
	Sleep_Model.to_pickle('/Volumes/HlabShare/Sleep_Model/'+mod_name + '_model.pkl')

	x_features = copy.deepcopy(FeatureList)
	[x_features.remove(i) for i in ['Animal_Name', 'Time_Interval','State']]
	if ymot == 'n':
		x_features.remove('Motion')
	if yemg == 'n':
		x_features.remove('EMG')

	prop = 1/2
	model_inputs = Sleep_Model[x_features][0:int((max(Sleep_Model.index)+1)*prop)].apply(pd.to_numeric)
	train_x = model_inputs.values
	model_input_states = Sleep_Model['State'][0:int((max(Sleep_Model.index)+1)*prop)].apply(pd.to_numeric)
	train_y = model_input_states.values

	model_test = Sleep_Model[x_features][int((max(Sleep_Model.index)+1)*prop):].apply(pd.to_numeric)
	test_x = model_test.values
	model_test_states = Sleep_Model['State'][int((max(Sleep_Model.index)+1)*prop):].apply(pd.to_numeric)
	test_y = model_test_states.values

	# TrainX = Features[0:int(prop*np.size(Features[:,0])),:]
	# TestX = Features[int(prop*np.size(Features[:,0])):None,:]
	# TrainY = Aligned[0:int(prop*np.size(Aligned))]
	# TestY = Aligned[int(prop*np.size(Aligned)):None]


	print('Calculating tree...')
	clf = random_forest_classifier(train_x, train_y)
	Predict_y = clf.predict(test_x)
	print ("Train Accuracy :: ", accuracy_score(train_y, clf.predict(train_x)))    
	print ("Test Accuracy :: ", accuracy_score(test_y, clf.predict(test_x)))

	Satisfaction = input('Satisfied?: ')
	if Satisfaction == 'y':
	    clf = random_forest_classifier(Sleep_Model[x_features].apply(pd.to_numeric).values, Sleep_Model['State'].apply(pd.to_numeric).values)
	    print ("Train Accuracy :: ", accuracy_score( Sleep_Model['State'].apply(pd.to_numeric).values, clf.predict(Sleep_Model[x_features].apply(pd.to_numeric).values)))
	    dump(clf, 'NewModelTest.joblib') 









