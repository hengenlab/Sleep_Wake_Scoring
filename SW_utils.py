import os
import numpy as np
import sys
from scipy.integrate import simps
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patch
from lizzie_work import DLCMovement_input
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load
import pandas as pd


def check_h5_file_size(h5files): # previously Check1
    # Checks to make sure that all of the h5 files are the same size
    sizes = [os.stat(i).st_size for i in h5files]
    if np.size(np.unique(sizes)) > 1:
        sys.exit('Not all of the h5 files are the same size')

def check_time_stamps(files): # previously check2
    # makes sure time stamps on videos are continuous
    str_idx = files[0].find('e3v') + 17
    timestamps = [files[i][str_idx:str_idx + 9] for i in np.arange(np.size(files))]
    if timestamps[0] == timestamps[1]:
        chk = input('Were these videos seperated for DLC? (y/n)')
    for i in np.arange(np.size(files) - 1):
        hr1 = timestamps[i][0:4]
        hr2 = timestamps[i][5:9]
        hr3 = timestamps[i + 1][0:4]
        hr4 = timestamps[i + 1][5:9]
        if hr2 != hr3:
            if chk == 'n':
                sys.exit('hour ' + str(i) + ' is not continuous with hour ' + str(i + 1))

def check_time_period(h5files, vidfiles): # previously check3
    # makes sure that videos and h5 files are over the same period of time
    str_idx = h5files[0].find('e3v') + 17
    timestamps_h5 = [h5files[i][str_idx:str_idx + 9] for i in np.arange(np.size(h5files))]
    timestamps_vid = [vidfiles[i][str_idx:str_idx + 9] for i in np.arange(np.size(vidfiles))]
    if timestamps_h5 != timestamps_vid:
        sys.exit('h5 files and video files not aligned')

def init_motion(movement):
    time = movement[1]
    time_sec = time * 3600
    dt = time_sec[2] - time_sec[1]
    dxy = movement[0]
    binsz = int(round(1 / dt))
    bindxy = dxy.reshape(900, 60)

    raw_var = np.nanvar(bindxy, axis = 1)
    rs_dxy = np.reshape(dxy, [int(np.size(dxy) / binsz), binsz])

    med = np.median(rs_dxy, axis = 1)
    hist = np.histogram(med[~np.isnan(med)], bins = 1000)
    csum = np.cumsum(hist[0])
    th = np.size(med) * 0.95
    outliers_idx = np.where(csum > th)[0][0]
    outliers = np.where(med > hist[1][outliers_idx])[0]

    for i in outliers:
        if i == 0:
            med[i] = med[i + 2]
        else:
            med[i] = med[i - 1]
            a = i - 1
        while med[i] > hist[1][outliers_idx]:
            a = i - 1
            med[i] = med[a]
    binned_mot = np.nanmean(np.reshape(med, (900, 4)), axis = 1)
    binned_mot[np.where(np.isnan(binned_mot))] = 0
    raw_var[np.where(np.isnan(raw_var))] = 0
    return binned_mot, raw_var, dxy, med, dt

def normMean(meanEEG_perhr, var_EEG_perhr, hr):
    if int(hr) - 1 < 12:
        normmean = np.mean(meanEEG_perhr[0:24])
        normvar = np.mean(var_EEG_perhr[0:24])
    elif np.size(meanEEG_perhr) - int(hr) < 12:
        normmean = np.mean(meanEEG_perhr[np.size(meanEEG_perhr) - 24: np.size(meanEEG_perhr)])
        normvar = np.mean(var_EEG_perhr[np.size(var_EEG_perhr) - 24: np.size(var_EEG_perhr)])
    else:
        normmean = np.mean(meanEEG_perhr[int(hr) - 12: int(hr) + 12])
        normvar = np.mean(var_EEG_perhr[int(hr) - 12: int(hr) + 12])
    normstd = np.sqrt(normvar)
    return normmean, normstd

def generate_EEG(downdatlfp, bin, fs, normmean, normstd):
    binl = bin * fs  # bin size in array slots
    EEGamp = np.zeros(int(np.size(downdatlfp) / (4 * fs)))
    EEGmean = np.zeros(int(np.size(downdatlfp) / (4 * fs)))
    for i in np.arange(np.size(EEGamp)):
        EEGamp[i] = np.var(downdatlfp[4 * fs * (i):(4 * fs * (i + 1))])
        EEGmean[i] = np.mean(np.abs(downdatlfp[4 * fs * (i):(4 * fs * (i + 1))]))
    EEGamp = (EEGamp - normmean) / normstd

    EEGmax = np.zeros(int(np.size(downdatlfp) / (4 * fs)))
    for i in np.arange(np.size(EEGmax)):
        EEGmax[i] = np.max(downdatlfp[4 * fs * (i):(4 * fs * (i + 1))])
    EEGmax = (EEGmax - np.average(EEGmax)) / np.std(EEGmax)
    return EEGamp, EEGmax, EEGmean

def generate_EMG(EMGamp):
    fse = int(np.size(EMGamp) / 900)
    EMG_idx = np.arange(0, 900 * fse + fse, fse)
    EMG = np.zeros(900)
    for i in np.arange(np.size(EMG)):
        EMG[i] = np.average(EMGamp[EMG_idx[i]:EMG_idx[i + 1]])
    return EMG

def bandPower(low, high, downdatlfp, epochlen, fs):
	win = epochlen * fs
	EEG = np.zeros(int(np.size(downdatlfp)/(epochlen*fs)))
	EEGreshape = np.reshape(downdatlfp,(-1,fs*epochlen))
	freqs, psd = signal.welch(EEGreshape, fs, nperseg=win, scaling='density')
	idx_min = np.argmax(freqs > low) - 1
	idx_max = np.argmax(freqs > high) - 1
	idx = np.zeros(dtype=bool, shape=freqs.shape)
	idx[idx_min:idx_max] = True
	EEG = simps(psd[:,idx], freqs[idx])
	return EEG, idx

def normalize(toNorm):
	norm = (toNorm - np.average(toNorm))/np.std(toNorm)
	return norm

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

def random_forest_classifier(features, target):
    clf = RandomForestClassifier(n_estimators=300)
    clf.fit(features, target)
    return clf

def plot_motion(ax, med, video_key=False):
    x_vals = np.linspace(0, 60, np.size(med))
    ax.plot(x_vals, med)
    ax.set_xlim(0, 60)
    ax.set_xticks(np.linspace(0, 60, 13))
    ax.set_ylim(0, 10)
    if video_key is False:
        title = [np.unique(video_key[1, :])[i][0:42] + ' \n ' for i in np.arange(np.size(np.unique(video_key[1, :])))]
        title = ''.join(title)
        title = title[0:-3]
    else:
        title = 'there is no video file. at some point should fix this so it pulls from the DLC data not just the video title'
    ax.set_title(title)
    sorted_med = np.sort(med)
    idx = np.where(sorted_med > max(sorted_med) * 0.50)[0][0]
    if idx == 0:
        thresh = sorted_med[idx]
    else:
        thresh = np.nanmean(sorted_med[0:idx])
    h = plt.gca().get_ylim()[1]
    consec = DLCMovement_input.group_consecutives(np.where(med > thresh)[0])
    for vals in consec:
        if len(vals) > 5:
            x = x_vals[vals[0]]
            y = 0
            width = x_vals[vals[-1]] - x
            rect = patch.Rectangle((x, y), width, h, color = '#b7e1a1', alpha = 0.5)
            ax.add_patch(rect)
    ax.set_xlim([0, 60])

def plot_spectrogram(ax, rawdat_dir, hr):
    ax.set_title('Spectrogram w/ EMG')
    img = mpimg.imread(rawdat_dir + 'specthr' + hr + '.jpg')
    ax.imshow(img, aspect = 'auto')
    ax.set_xlim(199, 1441)
    ax.set_ylim(178, 0)
    ax.set_xticks(np.linspace(199, 1441, 13))
    ax.set_xticklabels(np.arange(0, 65, 5))
    ticksy = [35, 100, 150]
    labelsy = [60, 6, 2]
    ax.set_yticks(ticksy, labelsy)

def plot_predicted(ax, Predict_y, clf, Features):
    ax.set_title('Predicted States')
    for state in np.arange(np.size(Predict_y)):
        if Predict_y[state] == 0:
            rect7 = patch.Rectangle((state, 0), 3.8, height = 1, color = 'green')
            ax.add_patch(rect7)
        elif Predict_y[state] == 2:
            rect7 = patch.Rectangle((state, 0), 3.8, height = 1, color = 'blue')
            ax.add_patch(rect7)
        elif Predict_y[state] == 5:
            rect7 = patch.Rectangle((state, 0), 3.8, height = 1, color = 'red')
            ax.add_patch(rect7)
        elif Predict_y[state] == 4:
            rect7 = patch.Rectangle((state, 0), 3.8, height = 1, color = '#a8a485')
            ax.add_patch(rect7)
    ax.set_ylim(0.3, 1)
    ax.set_xlim(0, 900)
    predictions = clf.predict_proba(Features)
    confidence = np.max(predictions, 1)
    ax.plot(confidence, color = 'k')

def create_prediction_figure(rawdat_dir, hr, Predict_y, clf, Features, pos, med=False, video_key=False):
    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1, figsize = (11, 6))
    plot_spectrogram(ax1,rawdat_dir, hr)
    plot_predicted(ax2, Predict_y, clf, Features)
    if pos:
        plot_motion(ax3, med, video_key)
    fig.tight_layout()
    return fig, ax1, ax2, ax3

def update_sleep_model(model_dir, mod_name, df_additions):
    try:
        Sleep_Model = np.load(file = model_dir + mod_name + '_model.pkl', allow_pickle = True)
        Sleep_Model = Sleep_Model.append(df_additions, ignore_index = True)
    except FileNotFoundError:
        print('no model created...I will save this one')
        df_additions.to_pickle(model_dir + mod_name + '_model.pkl')
        Sleep_Model = df_additions
    Sleep_Model.to_pickle(model_dir + mod_name + '_model.pkl')
    return Sleep_Model

def load_joblib(FeatureList, ymot, yemg, mod_name):
    x_features = copy.deepcopy(FeatureList)
    [x_features.remove(i) for i in ['Animal_Name', 'Time_Interval', 'State']]
    jobname =''
    if not ymot and yemg :
        x_features.remove('Motion')
        jobname = mod_name + '_EMG.joblib'

    if not yemg and ymot :
        x_features.remove('EMG')
        jobname = mod_name + '_Motion.joblib'

    if yemg and ymot :
        jobname = mod_name + '_Motion_EMG.joblib'

    if not yemg and not ymot :
        x_features.remove('EMG')
        x_features.remove('Motion')
        jobname = mod_name + '_no_move.joblib'
        print('Just so you know...this model has no EMG and no Motion')
    return jobname, x_features

def retrain_model(Sleep_Model, x_features, model_dir, jobname):
    prop = 1 / 2
    model_inputs = Sleep_Model[x_features][0:int((max(Sleep_Model.index) + 1) * prop)].apply(pd.to_numeric)
    train_x = model_inputs.values
    model_input_states = Sleep_Model['State'][0:int((max(Sleep_Model.index) + 1) * prop)].apply(pd.to_numeric)
    train_y = model_input_states.values

    model_test = Sleep_Model[x_features][int((max(Sleep_Model.index) + 1) * prop):].apply(pd.to_numeric)
    test_x = model_test.values
    model_test_states = Sleep_Model['State'][int((max(Sleep_Model.index) + 1) * prop):].apply(pd.to_numeric)
    test_y = model_test_states.values

    print('Calculating tree...')
    clf = random_forest_classifier(train_x, train_y)
    print("Train Accuracy :: ", accuracy_score(train_y, clf.predict(train_x)))
    print("Test Accuracy :: ", accuracy_score(test_y, clf.predict(test_x)))

    Satisfaction = input('Satisfied?: ')
    if Satisfaction == 'y':
        clf = random_forest_classifier(Sleep_Model[x_features].apply(pd.to_numeric).values,
                                       Sleep_Model['State'].apply(pd.to_numeric).values)
        print("Train Accuracy :: ", accuracy_score(Sleep_Model['State'].apply(pd.to_numeric).values,
                                                   clf.predict(Sleep_Model[x_features].apply(pd.to_numeric).values)))
        dump(clf, model_dir + jobname)


def pull_up_movie(index, vid_sample, video_key, motion_dir):
    start = int(index * fs * epochlen)
    end = start + int(3 * fs * epochlen)
    vid_win_idx = np.where(np.logical_and(vid_sample >= start, vid_sample < end))[0]
    vid_win = video_key[2][vid_win_idx]
    if np.size(np.where(vid_win == 'nan')[0]) > 0:
        print('There is no video here')
    else:
        vid_win = [int(float(i)) for i in vid_win]
        if np.size(np.unique(video_key[1][vid_win_idx])) > 1:
            print('This period is between two windows. No video to display')
        else:
            vidfilename = np.unique(video_key[1][vid_win_idx])[0]
            score_win = np.arange(int(vid_win[0]) + int(np.size(vid_win) / 3), int(vid_win[0]) + int((np.size(vid_win) / 3) * 2))
    x = (end - start) / ratio2
    length = np.arange(int(end / x - start / x))
    bottom = np.zeros(int(end / x - start / x))
    cap = cv2.VideoCapture(motion_dir + vidfilename)
    if not cap.isOpened():
        print("Error opening video stream or file")
    for f in np.arange(vid_win[0], vid_win[-1]):
        cap.set(1, f)
        ret, frame = cap.read()
        if ret:
            if f in score_win:
                cv2.putText(frame, "SCORE WINDOW", (50, 105), cv2.FONT_HERSHEY_PLAIN, 4, (225, 0, 0), 2)
            cv2.imshow('Frame', frame)
            cv2.waitKey(int((dt * 10e2) / 4))
    cap.release()
def pull_up_raw_trace():
    print('pull up the second figure for that bin - maybe. Option to click through a few bins around it?')


