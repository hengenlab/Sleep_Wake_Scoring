from scipy.signal import savgol_filter
import numpy as np
import scipy.signal as signal
from scipy.integrate import simps
import matplotlib.patches as patch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import copy
import neuraltoolkit as ntk
import seaborn as sns
import sys
import time as timer
import os
from lizzie_work import DLCMovement_input
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from joblib import dump, load
import pandas as pd
import warnings


#this is bad. i feel bad doing it. but here we are
print('this code is supressing warnings because they were excessive and annoying. \nIf someething weird is happening delete line 26 and try again\n')
warnings.filterwarnings("ignore")


def create_new_df(features):
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
    clf = RandomForestClassifier(n_estimators=300)
    clf.fit(features, target)
    return clf
def press(event):
	'''
	to score sleep states
	'''
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
	#makes sure time stamps on videos are continuous
	str_idx = files[0].find('e3v') + 17
	timestamps = [files[i][str_idx:str_idx+9] for i in np.arange(np.size(files))]
	if timestamps[0] == timestamps[1]:
		chk = input('Were these videos seperated for DLC? (y/n)')
	for i in np.arange(np.size(files)-1):
		hr1 = timestamps[i][0:4]
		hr2 = timestamps[i][5:9]
		hr3 = timestamps[i+1][0:4]
		hr4 = timestamps[i+1][5:9]
		if hr2 != hr3:
			if chk == 'n':
				sys.exit('hour '+str(i) + ' is not continuous with hour ' + str(i+1))

def check3(h5files, vidfiles):
	#makes sure that videos and h5 files are over the same period of time
	str_idx = h5files[0].find('e3v') + 17
	timestamps_h5 = [h5files[i][str_idx:str_idx+9] for i in np.arange(np.size(h5files))]
	timestamps_vid = [vidfiles[i][str_idx:str_idx+9] for i in np.arange(np.size(vidfiles))]
	if timestamps_h5 != timestamps_vid:
		sys.exit('h5 files and video files not aligned')

def normMean(hr):
	#---function
	if int(hr)-1<12:
		normmean = np.mean(meanEEG_perhr[0:24])
		normvar = np.mean(var_EEG_perhr[0:24])
	elif np.size(meanEEG_perhr)-int(hr) < 12:
		normmean = np.mean(meanEEG_perhr[np.size(meanEEG_perhr)-24: np.size(meanEEG_perhr)])
		normvar = np.mean(var_EEG_perhr[np.size(var_EEG_perhr)-24: np.size(var_EEG_perhr)])
	else:
		normmean = np.mean(meanEEG_perhr[int(hr)-12: int(hr)+12])
		normvar = np.mean(var_EEG_perhr[int(hr)-12: int(hr)+12])
	normstd = np.sqrt(normvar)
	return normmean, normstd
	#----function end

def EEG(downdatlfp):
	#----function (downdatlfp) returns eegmax eegamp
	bin = 4 #bin size in seconds
	binl = bin * fs #bin size in array slots
	EEGamp = np.zeros(int(np.size(downdatlfp)/(4*fs)))
	EEGmean = np.zeros(int(np.size(downdatlfp)/(4*fs)))
	for i in np.arange(np.size(EEGamp)):
		EEGamp[i] = np.var(downdatlfp[4*fs*(i):(4*fs*(i+1))])
		EEGmean[i] = np.mean(np.abs(downdatlfp[4*fs*(i):(4*fs*(i+1))]))
	EEGamp = (EEGamp - normmean)/normstd

	EEGmax = np.zeros(int(np.size(downdatlfp)/(4*fs)))
	for i in np.arange(np.size(EEGmax)):
		EEGmax[i] = np.max(downdatlfp[4*fs*(i):(4*fs*(i+1))])
	EEGmax = (EEGmax - np.average(EEGmax))/np.std(EEGmax)
	return EEGamp, EEGmax, EEGmean
	#-----end function

def EMG_one(EMGamp):
	#-----function (emgamp) returns EMG
	fse = int(np.size(EMGamp)/900)
	EMG_idx = np.arange(0, 900*fse+fse , fse)
	EMG = np.zeros(900)
	for i in np.arange(np.size(EMG)):
		EMG[i] = np.average(EMGamp[EMG_idx[i]:EMG_idx[i+1]])
	return EMG
	#-----function end

#-----function(low, high) return non normalized eegDelta and idx delta
#Calculate delta and theta bandpower for entire dataset
# Define window length (4 seconds)
#WIN = WINDOW? SAME AS EPOCH LENGTH? SHOULD IT STILL BE HARDCODED?
def bandPower(low, high, downdatlfp):
	win = 4 * fs
	EEG = np.zeros(int(np.size(downdatlfp)/(4*fs)))
	#Vectorized bandpower calculation
	EEGreshape = np.reshape(downdatlfp,(-1,fs*epochlen))
	win = 4*fs
	freqs, psd = signal.welch(EEGreshape, fs, nperseg=win, scaling='density')
	idx_min = np.argmax(freqs > low) - 1
	idx_max = np.argmax(freqs > high) - 1
	idx = np.zeros(dtype=bool, shape=freqs.shape)
	idx[idx_min:idx_max] = True
	EEG = simps(psd[:,idx], freqs[idx])
	return EEG, idx
	#----- end function

def normalize(toNorm):
	norm = (toNorm - np.average(toNorm))/np.std(toNorm)
	return norm


#funcitonalize this stuff too
#1 function for 1 before and after, different function for 2, different function for 3
def post_pre(post, pre):
	post = np.append(post, 0)
	post = np.delete(post, 0)
	pre = np.append(0, pre)
	pre = pre[0:-1]
	return post, pre

def fix_states(states, alter_nums = False):
	if alter_nums == True:
		states[states == 1] = 0
		states[states == 3] = 5

	for ss in np.arange(np.size(states)-1):
		#check if it is a flicker state
		if (ss != 0 and ss < np.size(states)-1):
			if states[ss+1] == states[ss-1]:
				states[ss] = states[ss+1]

		if (states[ss] == 0 and states[ss+1] == 5):
			states[ss] = 2
	if alter_nums == True:
		states[states == 0] = 1
		states[states == 5] = 3
	return states

def plot_predicted():
	figy = plt.figure(figsize=(11,6))
	plt.ion()
	ax1 = plt.subplot2grid((3, 1), (0, 0))
	plt.title('Spectrogram w/ EMG')
	img=mpimg.imread(rawdat_dir+'specthr'+ hr + '.jpg')
	imgplot = plt.imshow(img,aspect='auto')
	plt.xlim(199,1441)
	plt.ylim(178,0)
	ax1.set_xticks(np.linspace(199,1441, 13))
	ax1.set_xticklabels(np.arange(0, 65, 5))
	ticksy = [35,100,150]
	labelsy = [60,6,2]
	plt.yticks(ticksy, labelsy)
	ax2 = plt.subplot2grid((3, 1), (1, 0))
	plt.title('Predicted States')
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
		elif Predict_y[state] == 4:
			rect7 = patch.Rectangle((state,0),3.8,height=1,color='#a8a485')
			ax2.add_patch(rect7)
	plt.ylim(0.3,1)
	plt.xlim(0,900)
	Predictions  = clf.predict_proba(Features)
	predConf = np.max(Predictions,1)
	plt.plot(predConf, color= 'k')
	ax3 = plt.subplot2grid((3, 1), (2, 0))
	x_vals = np.linspace(0,60,np.size(med))
	plt.plot(x_vals, med)
	ax3.set_xlim(0, 60)
	ax3.set_xticks(np.linspace(0,60, 13))
	ax3.set_ylim(0,10)
	#title_idx = [movement_files[int(hr)-1].find('e3v'), movement_files[int(hr)-1].find('DeepCut')]
	title = [np.unique(video_key[1,:])[i][0:42]+' \n ' for i in np.arange(np.size(np.unique(video_key[1,:])))]
	title = ''.join(title)
	title = title[0:-3]
	plt.title(title)
	sorted_med = np.sort(med)
	idx = np.where(sorted_med>max(sorted_med)*0.50)[0][0]

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
			ax3.add_patch(rect)

	plt.xlim([0,60])

#rawdat_dir = 'Y:/HellWeek/EAB00025/EAB25_2018-10-19_18-25-07_p7c3_grounded2/'
#motion_dir = 'Y:/HellWeek/EAB00025/labeled_video_new/'
#motion_dir = 'Y:/HellWeek/EAB00023/labeled_videos_new/'
#rawdat_dir = 'Y:/HellWeek/EAB00023/EAB23_2018-10-19_18-28-50_p10c5_grounded2/'

#motion_dir= 'U:/rawdata/HellWeek/Grounded2/labeled_video_new/'
#rawdat_dir = 'U:/rawdata/HellWeek/Grounded2/EAB26_2018-10-19_18-23-50_p6c2/'

#EAB26 - 'Y:/HellWeek/Grounded2/EAB26_2018-10-19_18-23-50_p6c2'
#sleep_scoring_videos

#EAB40
#rawdat_dir = 'V:EAB00040/EAB00040_2019-03-29_10-28-27_p9_c5/'
#motion_dir = 'V:EAB00040/EAB00040_2019-03-29_10-28-27_p9_c5_labeled_vid/'
rawdat_dir = 'V:/EAB00040/EAB00040_2019-04-02_11-49-53_p9_c4/'
motion_dir = 'V:/EAB00040/EAB00040_2019-04-02_11-49-53_p9_c4_labeled_vid/'


os.chdir(rawdat_dir)
meanEEG_perhr = np.load(rawdat_dir+'Average_EEG_perhr.npy')
var_EEG_perhr = np.load(rawdat_dir+'Var_EEG_perhr.npy')

#inputing information for sleep scoring
animal = input('What animal is this?')
hr  = input('What hour are you working on? (starts at 1): ')
mod_name = input('Which model? (young_rat, adult_rat, mouse, rat_mouse)')
epochlen = int(input('Epoch length: '))
fs = int(input('sampling rate: '))
emg = input('Do you have emg info? y/n: ')
pos = input('Do you have a motion vector? y/n: ')


#creating delta theta EEG
delt = np.load('delt' + hr + '.npy')
delt = np.concatenate((500*[0],delt,500*[0]))
thet = np.load('thet' + hr + '.npy')
thet = np.concatenate((500*[0],thet,500*[0]))
downdatlfp = np.load('EEGhr' + hr + '.npy')

#pulling movement and video files
movement_files = np.sort(glob.glob(motion_dir+'*tmove.npy'))
vidkey_files = np.sort(glob.glob(motion_dir+'*vidkey.npy'))

#running quality checks
check2(movement_files)
check2(vidkey_files)
check3(movement_files, vidkey_files)


if emg == 'y':
	EMGamp = np.load('EMGhr' + hr + '.npy')
	EMGamp = (EMGamp-np.average(EMGamp))/np.std(EMGamp)

if pos == 'y':
	movement = np.load(movement_files[int(hr)-1])
	video_key = np.load(vidkey_files[int(hr)-1])
	time = movement[1]
	time_sec = time*3600
	dt = time_sec[2]-time_sec[1]
	dxy = movement[0]
	binsz = int(round(1/dt))

	bindxy = dxy.reshape(900,60)
	raw_var = np.nanvar(bindxy, axis=1)



	rs_dxy = np.reshape(dxy,[int(np.size(dxy)/binsz), binsz])
	time_min = np.linspace(0, 60, np.size(dxy))

	med = np.median(rs_dxy, axis = 1)
	binned_dxy = np.mean(rs_dxy, axis = 1)
	hist = np.histogram(med[~np.isnan(med)], bins = 1000)
	csum = np.cumsum(hist[0])
	th = np.size(med)*0.95
	outliers_idx = np.where(csum>th)[0][0]
	outliers = np.where(med>hist[1][outliers_idx])[0]
	print('size of outliers...', np.shape(outliers))
	for i in outliers:
		if i == 0:
			med[i] = med[i+2]
		else:
			med[i] = med[i-1]
			a = i-1
		while med[i] > hist[1][outliers_idx]:
			a = i-1
			med[i] = med[a]
	binned_mot = np.nanmean(np.reshape(med, (900, 4)), axis = 1)
	binned_mot[np.where(np.isnan(binned_mot))] = 0


ratio2 = 12*4

model = input('Use a random forest?: ')
satisfaction = []

if model == 'y':
	#choosing correct model
	os.chdir('X:/Sleep_Model/')
	if (pos == 'y' and emg == 'y'):
		clf = load(mod_name+'_Motion_EMG.joblib')
	if (pos == 'n' and emg == 'y'):
		clf = load(mod_name+'_EMG.joblib')
	if (pos == 'y' and emg == 'n'):
		clf = load(mod_name+'_Motion.joblib')
	if (pos == 'n' and emg == 'n'):
		clf = load(mod_name+'_no_move.joblib')

	#creates normalizing value around surrounding 24 hours
	os.chdir(rawdat_dir)

	normmean, normstd = normMean(hr)

	#Generate average/max EEG amplitude, EEG frequency, EMG amplitude for each bin
	print('Generating EEG vectors...')
	EEGamp, EEGmax, EEGmean = EEG(downdatlfp)

	if emg == 'y':
		EMG = EMG_one(EMGamp)

	#BANDPOWER
	print('Extracting delta bandpower...')
	EEGdelta, idx_delta = bandPower(0.5, 4, downdatlfp)

	print('Extracting theta bandpower...')
	EEGtheta, idx_theta = bandPower(4, 8, downdatlfp)

	print('Extracting alpha bandpower...')
	EEGalpha, idx_alpha = bandPower(8, 12, downdatlfp)

	print('Extracting beta bandpower...')
	EEGbeta, idx_beta = bandPower(12, 30, downdatlfp)

	print('Extracting gamma bandpower...')
	EEGgamma, idx_gamma = bandPower(30, 80, downdatlfp)

	print('Extracting narrow-band theta bandpower...')
	EEG_broadtheta, idx_broadtheta = bandPower(2, 16, downdatlfp)

	print('Boom. Boom. FIYA POWER...')
	EEGfire, idx_fire = bandPower(4, 20, downdatlfp)

	#RATIOS
	EEGnb = EEGtheta/EEG_broadtheta
	delt_thet = EEGdelta/EEGtheta

	#NORMALIZE
	EEGdelta = normalize(EEGdelta)
	EEGalpha = normalize(EEGalpha)
	EEGbeta = normalize(EEGbeta)
	EEGgamma = normalize(EEGbeta)
	EEGnb= normalize(EEGnb)
	EEGtheta = normalize(EEGtheta)
	EEGfire = normalize(EEGfire)
	delt_thet = normalize(delt_thet)


	#get pre and post (1,2,3) for delta, theta, and narrow band
	delta_post, delta_pre = post_pre(EEGdelta, EEGdelta)
	theta_post, theta_pre = post_pre(EEGtheta, EEGtheta)
	delta_post2, delta_pre2 = post_pre(delta_post, delta_pre)
	theta_post2, theta_pre2 = post_pre(theta_post, theta_pre)
	delta_post3, delta_pre3 = post_pre(delta_post2, delta_pre2)
	theta_post3, theta_pre3 = post_pre(theta_post2, theta_pre2)
	nb_post, nb_pre = post_pre(EEGnb, EEGnb)

	animal_name = np.full(np.size(delta_pre), animal)
	animal_num = np.zeros(900)
	animal_num[:] = int(animal_name[0][3:])

	#create feature list based on presence of EMG and movement information



	if (pos == 'y' and emg == 'n'):
		FeatureList = [animal_num, delta_pre, delta_pre2,delta_pre3,delta_post,delta_post2,delta_post3,EEGdelta,theta_pre,theta_pre2,theta_pre3,theta_post,theta_post2,theta_post3,
		EEGtheta,EEGalpha,EEGbeta,EEGgamma,EEGnb,nb_pre,delt_thet,EEGfire,EEGamp,EEGmax,EEGmean,binned_mot,raw_var]

	elif (pos == 'n' and emg == 'n'):
		FeatureList = [animal_num, delta_pre, delta_pre2,delta_pre3,delta_post,delta_post2,delta_post3,EEGdelta,theta_pre,theta_pre2,theta_pre3,theta_post,theta_post2,theta_post3,
		EEGtheta,EEGalpha,EEGbeta,EEGgamma,EEGnb,nb_pre,delt_thet,EEGfire,EEGamp,EEGmax,EEGmean,nans,nans]

	elif (pos == 'n' and emg == 'y'):
		FeatureList =[animal_num, delta_pre, delta_pre2,delta_pre3,delta_post,delta_post2,delta_post3,EEGdelta,theta_pre,theta_pre2,theta_pre3,theta_post,theta_post2,theta_post3,
		EEGtheta,EEGalpha,EEGbeta,EEGgamma,EEGnb,nb_pre,delt_thet,EEGfire,EEGamp,EEGmax,EEGmean,EMG,nans]

	elif (pos == 'y' and emg == 'y'):
		FeatureList = [animal_num, delta_pre, delta_pre2,delta_pre3,delta_post,delta_post2,delta_post3,EEGdelta,theta_pre,theta_pre2,theta_pre3,theta_post,theta_post2,theta_post3,
		EEGtheta,EEGalpha,EEGbeta,EEGgamma,EEGnb,nb_pre,delt_thet,EEGfire,EEGamp,EEGmax,EEGmean,EMG,binned_mot,raw_var]

	#FeatureList = FeatureList[1:-1]

	#runs random forest
	FeatureList_smoothed = []
	for f in FeatureList:
		FeatureList_smoothed.append(signal.medfilt(f, 5))

	Features = np.column_stack((FeatureList_smoothed))
	Predict_y = clf.predict(Features)

	#Making changes to predicted states
	Predict_y = fix_states(Predict_y)


	plot_predicted()


	plt.tight_layout()
	#plt.show()
	plt.savefig('C:/Users/clayt/Desktop/preview.png')
	satisfaction = input('Satisfied?: ')

if satisfaction == 'y':
	filename = animal + '_SleepStates' + hr + '.npy'
	np.save(filename,Predict_y)
	State = Predict_y
	update = input('Update model?(y/n): ')
	if update == 'y':
		ymot = input('Use motion?: ')
		yemg = input('Use EMG?: ')

		if model == 'n':
			normmean, normstd = normMean(hr)

			#Generate average/max EEG amplitude, EEG frequency, EMG amplitude for each bin
			print('Generating EEG vectors...')
			EEGamp, EEGmax, EEGmean = EEG(downdatlfp)

			EMG = EMG_one(EMGamp)

			#BANDPOWER
			print('Extracting delta bandpower...')
			EEGdelta, idx_delta = bandPower(0.5, 4, downdatlfp)

			print('Extracting theta bandpower...')
			EEGtheta, idx_theta = bandPower(4, 8, downdatlfp)

			print('Extracting alpha bandpower...')
			EEGalpha, idx_alpha = bandPower(8, 12, downdatlfp)

			print('Extracting beta bandpower...')
			EEGbeta, idx_beta = bandPower(12, 30, downdatlfp)

			print('Extracting gamma bandpower...')
			EEGgamma, idx_gamma = bandPower(30, 80, downdatlfp)

			print('Extracting narrow-band theta bandpower...')
			EEG_broadtheta, idx_broadtheta = bandPower(2, 16, downdatlfp)

			print('Boom. Boom. FIYA POWER...')
			EEGfire, idx_fire = bandPower(4, 20, downdatlfp)

			#RATIOS
			EEGnb = EEGtheta/EEG_broadtheta
			delt_thet = EEGdelta/EEGtheta

			#NORMALIZE
			EEGdelta = normalize(EEGdelta)
			EEGalpha = normalize(EEGalpha)
			EEGbeta = normalize(EEGbeta)
			EEGgamma = normalize(EEGbeta)
			EEGnb= normalize(EEGnb)
			EEGtheta = normalize(EEGtheta)
			EEGfire = normalize(EEGfire)
			delt_thet = normalize(delt_thet)


		#get pre and post (1,2,3) for delta, theta, and narrow band
		delta_post, delta_pre = post_pre(EEGdelta, EEGdelta)
		theta_post, theta_pre = post_pre(EEGtheta, EEGtheta)
		delta_post2, delta_pre2 = post_pre(delta_post, delta_pre)
		theta_post2, theta_pre2 = post_pre(theta_post, theta_pre)
		delta_post3, delta_pre3 = post_pre(delta_post2, delta_pre2)
		theta_post3, theta_pre3 = post_pre(theta_post2, theta_pre2)
		nb_post, nb_pre = post_pre(EEGnb, EEGnb)


		State[State == 1] = 0
		State[State == 2] = 2
		State[State == 3] = 5

		#Make a data frame with the new information and then concat to OG

		#this part might not be necessary
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
				if i == 0:
					med[i] = med[i+2]
				else:
					med[i] = med[i-1]
					a = i-1
				while med[i] > hist[1][outliers_idx]:
					a = i-1
					med[i] = med[a]
			binned_mot = np.nanmean(np.reshape(med, (900, 4)), axis = 1)
			binned_mot[np.isnan(binned_mot)] = 0

		animal_name = np.full(np.size(delta_pre), animal)
		time_int = [video_key[1,i][0:26] for i in np.arange(0, np.size(video_key[1,:]),int(np.size(video_key[1,:])/np.size(animal_name)))]
		nans = np.zeros(np.size(animal_name))
		nans[:] = np.nan
		if np.size(np.where(pd.isnull(EMG))[0]) > 0:
			 EMG[np.isnan(EMG)] = 0
		animal_num = np.zeros(900)
		animal_num[:] = int(animal_name[0][3:])
		#creating correct feature list
		if (ymot == 'y' and yemg == 'n'):
			data = np.vstack([animal_name, animal_num,time_int, State, delta_pre, delta_pre2,delta_pre3,delta_post,delta_post2,delta_post3,EEGdelta,theta_pre,theta_pre2,theta_pre3,theta_post,theta_post2,theta_post3,
			EEGtheta,EEGalpha,EEGbeta,EEGgamma,EEGnb,nb_pre,delt_thet,EEGfire,EEGamp,EEGmax,EEGmean,nans,binned_mot,raw_var])

		elif (ymot == 'n' and yemg == 'n'):
			data = np.vstack([animal_name, animal_num,time_int, State, delta_pre, delta_pre2,delta_pre3,delta_post,delta_post2,delta_post3,EEGdelta,theta_pre,theta_pre2,theta_pre3,theta_post,theta_post2,theta_post3,
			EEGtheta,EEGalpha,EEGbeta,EEGgamma,EEGnb,nb_pre,delt_thet,EEGfire,EEGamp,EEGmax,EEGmean,nans,nans,nans])

		elif (ymot == 'n' and yemg == 'y'):
			data = np.vstack([animal_name, animal_num,time_int, State, delta_pre, delta_pre2,delta_pre3,delta_post,delta_post2,delta_post3,EEGdelta,theta_pre,theta_pre2,theta_pre3,theta_post,theta_post2,theta_post3,
			EEGtheta,EEGalpha,EEGbeta,EEGgamma,EEGnb,nb_pre,delt_thet,EEGfire,EEGamp,EEGmax,EEGmean,EMG,nans,nans])
		elif(ymot == 'y' and yemg == 'y'):
			data = np.vstack([animal_name, animal_num,time_int, State, delta_pre, delta_pre2,delta_pre3,delta_post,delta_post2,delta_post3,EEGdelta,theta_pre,theta_pre2,theta_pre3,theta_post,theta_post2,theta_post3,
			EEGtheta,EEGalpha,EEGbeta,EEGgamma,EEGnb,nb_pre,delt_thet,EEGfire,EEGamp,EEGmax,EEGmean,EMG,binned_mot,raw_var])
		FeatureList = ['Animal_Name','animal_num', 'Time_Interval','State','delta_pre','delta_pre2','delta_pre3','delta_post','delta_post2','delta_post3','EEGdelta','theta_pre','theta_pre2','theta_pre3','theta_post',
		'theta_post2','theta_post3','EEGtheta','EEGalpha','EEGbeta','EEGgamma','EEGnarrow','nb_pre','delta/theta','EEGfire','EEGamp','EEGmax','EEGmean','EMG', 'Motion', 'raw_var']

		df_additions = pd.DataFrame(columns = FeatureList, data = data.T)

		try:
			Sleep_Model = np.load(file='X:/Sleep_Model/' + mod_name + '_model.pkl', allow_pickle=True)
			Sleep_Model = Sleep_Model.append(df_additions, ignore_index=True)
		except FileNotFoundError:
			print('no model created...I will save this one')
			df_additions.to_pickle('X:/Sleep_Model/'+mod_name + '_model.pkl')
			Sleep_Model = df_additions
		Sleep_Model.to_pickle('X:/Sleep_Model/'+mod_name + '_model.pkl')

		x_features = copy.deepcopy(FeatureList)
		[x_features.remove(i) for i in ['Animal_Name', 'Time_Interval','State']]

		if (ymot == 'n' and yemg == 'y'):
			x_features.remove('Motion')
			jobname =mod_name+'_EMG.joblib'

		if (yemg == 'n' and ymot == 'y'):
			x_features.remove('EMG')
			jobname =mod_name+'_Motion.joblib'

		if (yemg == 'y' and ymot == 'y'):
			jobname = mod_name+'_Motion_EMG.joblib'

		if (yemg == 'n' and ymot == 'n'):
			x_features.remove('EMG')
			x_features.remove('Motion')
			jobname = mod_name+'_no_move.joblib'
			print('Just so you know...this model has no EMG and no Motion')

		if yemg == 'y':
			Sleep_Model = Sleep_Model.drop(index = np.where(Sleep_Model['EMG'].isin(['nan']))[0])

		#retrain the model!!
		prop = 1/2
		model_inputs = Sleep_Model[x_features][0:int((max(Sleep_Model.index)+1)*prop)].apply(pd.to_numeric)
		train_x = model_inputs.values
		model_input_states = Sleep_Model['State'][0:int((max(Sleep_Model.index)+1)*prop)].apply(pd.to_numeric)
		train_y = model_input_states.values

		model_test = Sleep_Model[x_features][int((max(Sleep_Model.index)+1)*prop):].apply(pd.to_numeric)
		test_x = model_test.values
		model_test_states = Sleep_Model['State'][int((max(Sleep_Model.index)+1)*prop):].apply(pd.to_numeric)
		test_y = model_test_states.values

		model_dir = 'X:/Sleep_Model/'


		print('Calculating tree...')
		clf = random_forest_classifier(train_x, train_y)
		Predict_y = clf.predict(test_x)
		print ("Train Accuracy :: ", accuracy_score(train_y, clf.predict(train_x)))
		print ("Test Accuracy :: ", accuracy_score(test_y, clf.predict(test_x)))

		Satisfaction = input('Satisfied?: ')
		if Satisfaction == 'y':
		    clf = random_forest_classifier(Sleep_Model[x_features].apply(pd.to_numeric).values, Sleep_Model['State'].apply(pd.to_numeric).values)
		    print ("Train Accuracy :: ", accuracy_score( Sleep_Model['State'].apply(pd.to_numeric).values, clf.predict(Sleep_Model[x_features].apply(pd.to_numeric).values)))
		    dump(clf, model_dir+jobname)

	sys.exit()



fix = input("Do you want to fix states?")
if fix == 'y':
	idxs = []
	inp = input('Which bins? (#-#, done)')
	while(inp != 'done'):
		idxs.append(inp)
		inp = input('Which bins? (#-#, done)')

	indicies=[]
	for i in idxs:
		bin1 = int(i[:i.index('-')])
		bin2 = int(i[i.index('-')+1:])
		indicies.extend(np.arange(bin1-1, bin2))
	if indicies[0]<0:
		indicies = indicies[1:]
	if 0 not in indicies:
		indicies.insert(0,0)

	print('Ready for scoring, opening GUI.')
	#for scoring
	plt.ion()
	State = copy.deepcopy(Predict_y)
	State[State == 0] = 1
	State[State == 2] = 2
	State[State == 5] = 3

	realtime = np.arange(np.size(downdatlfp))/fs
	if emg == 'y':
		EMGamp = np.pad(EMGamp, (0,100), 'constant')
	if emg == 'n':
		EMGamp = False
	vid_samp = np.linspace(0, 3600*fs, np.size(video_key[0]))


	for a,i in enumerate(indicies):
		print('a = ' + str(a))
		print('i = ' + str(i))
		if i > indicies[a-1] + 1:
			print('adding last state before jumping')
			if State[int(indicies[a-1])] == 1:
				rect7 = patch.Rectangle(((start/fs)-epochlen,0),3.8,height=2,color='green')
				ax6.add_patch(rect7)
			elif State[int(indicies[a-1])] == 2:
				rect7 = patch.Rectangle(((start/fs)-epochlen,0),3.8,height=2,color='blue')
				ax6.add_patch(rect7)
			elif State[int(indicies[a-1])] == 3:
				rect7 = patch.Rectangle(((start/fs)-epochlen,0),3.8,height=2,color='red')
				ax6.add_patch(rect7)
			else:
				print('Ambiguous State, coded as 4')
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
			if mod_name == 'mouse':
				plt.ylim(-1000,1000)
			else:
				plt.ylim(-1000,1000)
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
			if np.size(EMGamp)<2:
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
			ax6.set_xticklabels(np.linspace(0, 60,13))
			plt.ylim(0.5,2)

			for s in np.arange(np.size(Predict_y)):
				st = s*epochlen
				if s in indicies[1:]:
					rect7 = patch.Rectangle((st,0),3.8,height=2,color='white')
					ax6.add_patch(rect7)
				else:
					if Predict_y[s] == 0:
						rect7 = patch.Rectangle((st,0),3.8,height=2,color='green')
						ax6.add_patch(rect7)
					elif Predict_y[s] == 2:
						rect7 = patch.Rectangle((st,0),3.8,height=2,color='blue')
						ax6.add_patch(rect7)
					elif Predict_y[s] == 5:
						rect7 = patch.Rectangle((st,0),3.8,height=2,color='red')
						ax6.add_patch(rect7)
					elif Predict_y[s] == 4:
						rect7 = patch.Rectangle((st,0),3.8,height=2,color='#a8a485')
						ax6.add_patch(rect7)

			fig2.canvas.mpl_connect('key_press_event', press)
			ax7= plt.subplot2grid((7, 1), (2, 0),rowspan=3)
			img=mpimg.imread(rawdat_dir+'specthr'+ hr + '.jpg')
			spect_x = np.linspace(199, 1442, np.size(State))
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
			move_x = np.linspace(0,60, np.size(State))
			plt.plot(x_vals, med)
			ax8.set_xlim(0, 60)
			ax8.set_xticks(np.linspace(0,60, 13))
			ax8.set_ylim(0,10)
			title_idx = [movement_files[int(hr)-1].find('e3v'), movement_files[int(hr)-1].find('DeepCut')]
			plt.title(movement_files[int(hr)-1][title_idx[0]:title_idx[1]])

			ylims = ax7.get_ylim()
			marker_line1 = ax7.plot([0, 0], [ylims[0], ylims[1]], linewidth = 0.5, color = 'k')
			ml1 = marker_line1.pop(0)
			ax7.set_ylim([ylims[0], ylims[1]])

			ylims = ax8.get_ylim()
			marker_line2 = ax8.plot([0, 0], [ylims[0], ylims[1]], linewidth = 0.5, color = 'k')
			ml2 = marker_line2.pop(0)
			ax8.set_ylim([ylims[0], ylims[1]])

			sorted_med = np.sort(med)
			idx = np.where(sorted_med>max(sorted_med)*0.50)[0][0]

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
				# keyboard function ------ void
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
		elif i == np.size(downdatlfp)/(fs*4) - 2:
			plt.figure()
			print('Scoring done, plotting sleep states.')
			State = State[:-1]
			plt.plot(State)
			first = int(input('Enter first sleep state: '))
			first = np.array(first)
			last = int(input('Enter last sleep state: '))
			State = np.append(first,State)
			State[-1] = last
			plt.close('all')
			break
		else:
			line1.set_ydata(downdatlfp[start:end])
			if model == 'y':
				t1.set_text(str(Prediction))
				t2.set_text(str(predConf))
			line2.set_ydata(delt[start:end])
			line3.set_ydata(thet[start:end])
			if np.size(EMGamp)>1:
				ax5.collections.clear()
				ax5.fill_between(length,bottom,EMGamp[int(i*4*epochlen):int(i*4*epochlen+4*3*epochlen)],color='red')

			ml1.remove()
			ml2.remove()

			ylims = ax7.get_ylim()
			marker_line1 = ax7.plot([spect_x[int(i)], spect_x[int(i)]], [ylims[0], ylims[1]], linewidth = 0.5, color = 'k')
			ml1 = marker_line1.pop(0)
			ax7.set_ylim([ylims[0], ylims[1]])

			ylims = ax8.get_ylim()
			marker_line2 = ax8.plot([move_x[int(i)], move_x[int(i)]], [ylims[0], ylims[1]], linewidth = 0.5, color = 'k')
			ml2 = marker_line2.pop(0)
			ax8.set_ylim([ylims[0], ylims[1]])


			if State[int(i-1)] == 1:
				rect7 = patch.Rectangle(((start/fs)-epochlen,0),3.8,height=2,color='green')
				ax6.add_patch(rect7)
			elif State[int(i-1)] == 2:
				rect7 = patch.Rectangle(((start/fs)-epochlen,0),3.8,height=2,color='blue')
				ax6.add_patch(rect7)
			elif State[int(i-1)] == 3:
				rect7 = patch.Rectangle(((start/fs)-epochlen,0),3.8,height=2,color='red')
				ax6.add_patch(rect7)
			else:
				print('Ambiguous State, coded as 4')
			fig1.canvas.draw()
			fig1.canvas.flush_events()
			fig2.canvas.draw()
			fig2.canvas.flush_events()
			keyboardClick=None
			while keyboardClick != True:
				#----keybboard function
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
		if i == indicies[-1]:
			plt.figure()
			print('Scoring done, plotting sleep states.')
			plt.plot(State)
			plt.close('all')
			break

else:
	redo = input('Are you continuing a previously closed session? (y/n): ')
	if redo == 'y':
		filename = input('File name containing sleep states: ')
	else:
		filename = input('File name for saving sleep states: ')

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

	n = 0
	for i in np.arange(np.size(downdatlfp)/(fs*4)-1):
		i = i + n
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
			if redo == 'y':
				try:
					State = np.load(filename)
				except:
					print('Sleep states file not found.')
				if State[0]!=0:
					State = np.append(0,State)
				n = int(np.where(State==0)[0][1])
				i = n
				start = int(i*fs*epochlen)
				end = int(i*fs*epochlen+fs*3*epochlen)
				length = np.arange(int(end/x-start/x))
				bottom = np.zeros(int(end/x-start/x))
				fig1 = plt.figure(figsize=(11,6))
				ax1 = plt.subplot2grid((4, 1), (0, 0))
				line1, = ax1.plot(realtime[start:end],downdatlfp[start:end])
				plt.xlim(start/fs,end/fs)
				plt.title('LFP')
				if mod_name == 'mouse':
					plt.ylim(-1000,1000)
				else:
					plt.ylim(-1000,1000)
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
				if np.size(EMGamp)<2:
					plt.text(0.5, 0.5, 'There is no EMG')
				else:
					plt.fill_between(length,bottom,EMGamp[int(i*4*epochlen):int(i*4*epochlen)+int(end/x-start/x)],color='red')
					plt.title('EMG power')
					plt.xlim(0,int((end-start)/x)-1)
					plt.ylim(-1,5)

				plt.tight_layout()

				fig2 = plt.figure(figsize = (11,8))
				ax6 = plt.subplot2grid((7, 1), (0, 0), rowspan = 2)
				for idx in np.arange(np.size(State)):
					if State[idx]==0:
						rect = patch.Rectangle((idx*4,0),1,height=2,color='g',alpha=0)
					if State[idx]==1:
						rect = patch.Rectangle((idx*4,0),1,height=2,color='g')
					if State[idx]==2:
						rect = patch.Rectangle((idx*4,0),1,height=2,color='b')
					if State[idx]==3:
						rect = patch.Rectangle((idx*4,0),1,height=2,color='r')
					ax6.add_patch(rect)
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
				spect_x = np.linspace(199, 1442, np.size(State))
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
				ax8.set_ylim(0,10)
				title_idx = [movement_files[int(hr)-1].find('e3v'), movement_files[int(hr)-1].find('DeepCut')]
				plt.title(movement_files[int(hr)-1][title_idx[0]:title_idx[1]])
				sorted_med = np.sort(med)
				idx = np.where(sorted_med>int(max(sorted_med)*0.05))[0][0]

				ylims = ax7.get_ylim()
				marker_line1 = ax7.plot([0, 0], [ylims[0], ylims[1]], linewidth = 0.5, color = 'k')
				ml1 = marker_line1.pop(0)
				ax7.set_ylim([ylims[0], ylims[1]])

				ylims = ax8.get_ylim()
				marker_line2 = ax8.plot([0, 0], [ylims[0], ylims[1]], linewidth = 0.5, color = 'k')
				ml2 = marker_line2.pop(0)
				ax8.set_ylim([ylims[0], ylims[1]])

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
			else:
				fig1 = plt.figure(figsize=(11,6))
				ax1 = plt.subplot2grid((4, 1), (0, 0))
				line1, = ax1.plot(realtime[start:end],downdatlfp[start:end])
				plt.xlim(start/fs,end/fs)
				plt.title('LFP')
				if mod_name == 'mouse':
					plt.ylim(-1000,1000)
				else:
					plt.ylim(-1000,1000)
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
				if np.size(EMGamp)>1:
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
				ax8.set_ylim(0,10)
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
		elif i == np.size(downdatlfp)/(fs*4) - 2:
			plt.figure()
			print('Scoring done, plotting sleep states.')
			State = State[:-1]
			plt.plot(State)
			first = int(input('Enter first sleep state: '))
			first = np.array(first)
			last = int(input('Enter last sleep state: '))
			State = np.append(first,State)
			State[-1] = last
			plt.close('all')
			break
		else:
			line1.set_ydata(downdatlfp[start:end])
			if model == 'y':
				t1.set_text(str(Prediction))
				t2.set_text(str(predConf))
			line2.set_ydata(delt[start:end])
			line3.set_ydata(thet[start:end])
			if np.size(EMG)>1:
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
				print('Ambiguous State, coded as 4')
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
			np.save(filename,State)

#if you fuck up, run from this line to the end
State = fix_states(State, alter_nums = True)
# plt.show(block=True) #if you fuck up comment this line out
decision = input('Save sleep states? y/n: ')
if decision == 'y':
	filename = animal + '_SleepStates' + hr + '.npy'
	np.save(filename,State)
update = input('Update model?(y/n): ')
if update == 'y':
	ymot = input('Use motion?: ')
	yemg = input('Use EMG?: ')

	if model == 'n':
		normmean, normstd = normMean(hr)

		#Generate average/max EEG amplitude, EEG frequency, EMG amplitude for each bin
		print('Generating EEG vectors...')
		EEGamp, EEGmax, EEGmean = EEG(downdatlfp)

		EMG = EMG_one(EMGamp)

		#BANDPOWER
		print('Extracting delta bandpower...')
		EEGdelta, idx_delta = bandPower(0.5, 4, downdatlfp)

		print('Extracting theta bandpower...')
		EEGtheta, idx_theta = bandPower(4, 8, downdatlfp)

		print('Extracting alpha bandpower...')
		EEGalpha, idx_alpha = bandPower(8, 12, downdatlfp)

		print('Extracting beta bandpower...')
		EEGbeta, idx_beta = bandPower(12, 30, downdatlfp)

		print('Extracting gamma bandpower...')
		EEGgamma, idx_gamma = bandPower(30, 80, downdatlfp)

		print('Extracting narrow-band theta bandpower...')
		EEG_broadtheta, idx_broadtheta = bandPower(2, 16, downdatlfp)

		print('Boom. Boom. FIYA POWER...')
		EEGfire, idx_fire = bandPower(4, 20, downdatlfp)

		#RATIOS
		EEGnb = EEGtheta/EEG_broadtheta
		delt_thet = EEGdelta/EEGtheta

		#NORMALIZE
		EEGdelta = normalize(EEGdelta)
		EEGalpha = normalize(EEGalpha)
		EEGbeta = normalize(EEGbeta)
		EEGgamma = normalize(EEGbeta)
		EEGnb= normalize(EEGnb)
		EEGtheta = normalize(EEGtheta)
		EEGfire = normalize(EEGfire)
		delt_thet = normalize(delt_thet)


	#get pre and post (1,2,3) for delta, theta, and narrow band
	delta_post, delta_pre = post_pre(EEGdelta, EEGdelta)
	theta_post, theta_pre = post_pre(EEGtheta, EEGtheta)
	delta_post2, delta_pre2 = post_pre(delta_post, delta_pre)
	theta_post2, theta_pre2 = post_pre(theta_post, theta_pre)
	delta_post3, delta_pre3 = post_pre(delta_post2, delta_pre2)
	theta_post3, theta_pre3 = post_pre(theta_post2, theta_pre2)
	nb_post, nb_pre = post_pre(EEGnb, EEGnb)


	State[State == 1] = 0
	State[State == 2] = 2
	State[State == 3] = 5

	#Make a data frame with the new information and then concat to OG

	#this part might not be necessary
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
			if i == 0:
				med[i] = med[i+2]
			else:
				med[i] = med[i-1]
				a = i-1
			while med[i] > hist[1][outliers_idx]:
				a = i-1
				med[i] = med[a]
		binned_mot = np.nanmean(np.reshape(med, (900, 4)), axis = 1)
		binned_mot[np.isnan(binned_mot)] = 0

	animal_name = np.full(np.size(delta_pre), animal)
	time_int = [video_key[1,i][0:26] for i in np.arange(0, np.size(video_key[1,:]),int(np.size(video_key[1,:])/np.size(animal_name)))]
	nans = np.zeros(np.size(animal_name))
	nans[:] = np.nan
	if yemg == 'y':
		if np.size(np.where(pd.isnull(EMG))[0]) > 0:
		 	EMG[np.isnan(EMG)] = 0

	animal_num = np.zeros(900)
	animal_num[:] = int(animal_name[0][3:])
	#creating correct feature list
	if (ymot == 'y' and yemg == 'n'):
		data = np.vstack([animal_name, animal_num, time_int, State, delta_pre, delta_pre2,delta_pre3,delta_post,delta_post2,delta_post3,EEGdelta,theta_pre,theta_pre2,theta_pre3,theta_post,theta_post2,theta_post3,
		EEGtheta,EEGalpha,EEGbeta,EEGgamma,EEGnb,nb_pre,delt_thet,EEGfire,EEGamp,EEGmax,EEGmean,nans,binned_mot,raw_var])

	elif (ymot == 'n' and yemg == 'n'):
		data = np.vstack([animal_name, animal_num, time_int, State, delta_pre, delta_pre2,delta_pre3,delta_post,delta_post2,delta_post3,EEGdelta,theta_pre,theta_pre2,theta_pre3,theta_post,theta_post2,theta_post3,
		EEGtheta,EEGalpha,EEGbeta,EEGgamma,EEGnb,nb_pre,delt_thet,EEGfire,EEGamp,EEGmax,EEGmean,nans,nans,nans])

	elif (ymot == 'n' and yemg == 'y'):
		data = np.vstack([animal_name, animal_num, time_int, State, delta_pre, delta_pre2,delta_pre3,delta_post,delta_post2,delta_post3,EEGdelta,theta_pre,theta_pre2,theta_pre3,theta_post,theta_post2,theta_post3,
		EEGtheta,EEGalpha,EEGbeta,EEGgamma,EEGnb,nb_pre,delt_thet,EEGfire,EEGamp,EEGmax,EEGmean,EMG,nans,nans])
	elif(ymot == 'y' and yemg == 'y'):
		data = np.vstack([animal_name, animal_num,time_int, State, delta_pre, delta_pre2,delta_pre3,delta_post,delta_post2,delta_post3,EEGdelta,theta_pre,theta_pre2,theta_pre3,theta_post,theta_post2,theta_post3,
		EEGtheta,EEGalpha,EEGbeta,EEGgamma,EEGnb,nb_pre,delt_thet,EEGfire,EEGamp,EEGmax,EEGmean,EMG,binned_mot,raw_var])
	FeatureList = ['Animal_Name','animal_num' ,'Time_Interval','State','delta_pre','delta_pre2','delta_pre3','delta_post','delta_post2','delta_post3','EEGdelta','theta_pre','theta_pre2','theta_pre3','theta_post',
	'theta_post2','theta_post3','EEGtheta','EEGalpha','EEGbeta','EEGgamma','EEGnarrow','nb_pre','delta/theta','EEGfire','EEGamp','EEGmax','EEGmean','EMG', 'Motion', 'raw_var']

	df_additions = pd.DataFrame(columns = FeatureList, data = data.T)

	try:
		Sleep_Model = np.load(file='X:/Sleep_Model/' + mod_name + '_model.pkl', allow_pickle=True)
		Sleep_Model = Sleep_Model.append(df_additions, ignore_index=True)
	except FileNotFoundError:
		print('no model created...I will save this one')
		df_additions.to_pickle('X:/Sleep_Model/'+mod_name + '_model.pkl')
		Sleep_Model = df_additions
	Sleep_Model.to_pickle('X:/Sleep_Model/'+mod_name + '_model.pkl')

	x_features = copy.deepcopy(FeatureList)
	[x_features.remove(i) for i in ['Animal_Name', 'Time_Interval','State']]

	if (ymot == 'n' and yemg == 'y'):
		x_features.remove('Motion')
		jobname =mod_name+'_EMG.joblib'

	if (yemg == 'n' and ymot == 'y'):
		x_features.remove('EMG')
		jobname =mod_name+'_Motion.joblib'

	if (yemg == 'y' and ymot == 'y'):
		jobname = mod_name+'_Motion_EMG.joblib'

	if (yemg == 'n' and ymot == 'n'):
		x_features.remove('EMG')
		x_features.remove('Motion')
		jobname = mod_name+'_no_move.joblib'
		print('Just so you know...this model has no EMG and no Motion')

	if yemg == 'y':
		Sleep_Model = Sleep_Model.drop(index = np.where(Sleep_Model['EMG'].isin(['nan']))[0])

	#retrain the model!!
	prop = 1/2
	model_inputs = Sleep_Model[x_features][0:int((max(Sleep_Model.index)+1)*prop)].apply(pd.to_numeric)
	train_x = model_inputs.values
	model_input_states = Sleep_Model['State'][0:int((max(Sleep_Model.index)+1)*prop)].apply(pd.to_numeric)
	train_y = model_input_states.values

	model_test = Sleep_Model[x_features][int((max(Sleep_Model.index)+1)*prop):].apply(pd.to_numeric)
	test_x = model_test.values
	model_test_states = Sleep_Model['State'][int((max(Sleep_Model.index)+1)*prop):].apply(pd.to_numeric)
	test_y = model_test_states.values

	model_dir = 'X:/Sleep_Model/'


	print('Calculating tree...')
	clf = random_forest_classifier(train_x, train_y)
	Predict_y = clf.predict(test_x)
	print ("Train Accuracy :: ", accuracy_score(train_y, clf.predict(train_x)))
	print ("Test Accuracy :: ", accuracy_score(test_y, clf.predict(test_x)))

	Satisfaction = input('Satisfied?: ')
	if Satisfaction == 'y':
	    clf = random_forest_classifier(Sleep_Model[x_features].apply(pd.to_numeric).values, Sleep_Model['State'].apply(pd.to_numeric).values)
	    print ("Train Accuracy :: ", accuracy_score( Sleep_Model['State'].apply(pd.to_numeric).values, clf.predict(Sleep_Model[x_features].apply(pd.to_numeric).values)))
	    dump(clf, model_dir+jobname)
