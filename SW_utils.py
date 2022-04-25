import numpy as np
import sys
from scipy.integrate import simps
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patch
# from lizzie_work import DLCMovement_input
import DLCMovement_input
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# from joblib import dump, load
from joblib import dump
import pandas as pd
import cv2
# import neuraltoolkit as ntk
import math
import os.path as op
import re
from datetime import datetime


def check_h5_file_size(h5files):
    '''
    Checks to make sure that all of the h5 files are the same size

    check_h5_file_size(h5files)
        h5files : list of h5files locations
    '''

    sizes = \
        [op.getsize(i) if op.isfile(i)
         else sys.exit('file ' + i + ' not found') for i in h5files]
    if np.size(np.unique(sizes)) > 1:
        raise \
            ValueError('Not all of the h5 files are the same size')


def get_time_from_h5_videofiles(sw_input_files_list):
    '''
    Get timestamps from files with file names *20210126T051544-061545* format

    get_time_from_h5_videofiles(sw_input_files_list)
        sw_input_files_list : list of h5 files or video files
    '''
    sw_input_file_realtime_dt_list = None
    sw_input_file_realtime_dt_list = []
    for sw_input_file in sw_input_files_list:
        sw_input_file_indx = \
            re.search(r'\d\d\d\d\d\d\d\dT\d\d\d\d\d\d', sw_input_file)
        sw_input_file_realtime_dt = \
            datetime.strptime(sw_input_file[sw_input_file_indx.start():
                                            sw_input_file_indx.end()]
                              .replace("T", " "), "%Y%m%d %H%M%S")
        sw_input_file_realtime_dt_list.append(sw_input_file_realtime_dt)
        # print(time.mktime(dt.timetuple()))
    return sw_input_file_realtime_dt_list


def check_time_stamps(files):  # previously check2
    # # makes sure time stamps on videos are continuous
    # str_idx = files[0].find('e3v') + 17
    # timestamps = [files[i][str_idx:str_idx + 9]
    #               for i in np.arange(np.size(files))]
    # if (timestamps[0] == timestamps[1]):
    #     chk = str(input('Were1 these videos seperated for DLC? (y/n)'))
    # for i in np.arange(np.size(files) - 1):
    #     # hr1 = timestamps[i][0:4]
    #     hr2 = timestamps[i][5:9]
    #     hr3 = timestamps[i + 1][0:4]
    #     # hr4 = timestamps[i + 1][5:9]
    #     if hr2 != hr3:
    #         if chk == 'n':
    #             sys.exit('hour ' + str(i) +
    #                      ' is not continuous with hour ' + str(i + 1))
    sw_input_file_realtime_dt_list =\
            get_time_from_h5_videofiles(files)
    if sw_input_file_realtime_dt_list != \
            sorted(sw_input_file_realtime_dt_list):
        print("\n")
        print("="*42)
        print(sw_input_file_realtime_dt_list)
        print("="*42)
        print("\n")
        print("NOT SAME AS ")
        print("\n")
        print("="*42)
        print(sorted(sw_input_file_realtime_dt_list))
        print("="*42)
        print("\n")
        print("Files are not continouse in time")
        chk = str(input('To continue? (y/n)'))
        if chk.lower() == 'n':
            raise ValueError('Files are not continous')


def check_time_period(h5files, vidfiles):  # previously check3
    # makes sure that videos and h5 files are over the same period of time
    str_idx = h5files[0].find('e3v') + 17
    timestamps_h5 = [h5files[i][str_idx:str_idx + 9]
                     for i in np.arange(np.size(h5files))]
    timestamps_vid = [vidfiles[i][str_idx:str_idx + 9]
                      for i in np.arange(np.size(vidfiles))]
    if timestamps_h5 != timestamps_vid:
        sys.exit('h5 files and video files not aligned')


def init_motion(movement):

    # Time to seconds
    time = movement[1]
    time_sec = time * 3600

    # Get bin size Movement
    dt = time_sec[2] - time_sec[1]
    # print("dt ", dt)
    # tmove is the dxy file with shape (2,54000) if video is 15Hz
    dxy = movement[0]

    # Get video fps
    # 1/dt gives 15 fps if dt =0.06666
    binsz = int(round(1 / dt))
    # print(dxy.shape)

    # Adjust if there is extra frames
    # reshape if there are extra frames such as 54001
    if (dxy.shape[0] > 54000) and (dxy.shape[0] < 54015):
        dxy = dxy[:54000]
    elif (dxy.shape[0] > 108000) and (dxy.shape[0] < 108030):
        dxy = dxy[:108000]

    # Reshape 4s bins
    # 900 windows for 3600s; 15Hz --> 60 per window; 30Hz --> 120 per window
    bindxy = dxy.reshape(900, int(binsz*4))
    # print("sh bindxy ", bindxy.shape)
    # Variance for each 4seconds
    raw_var = np.nanvar(bindxy, axis=1)
    # print("sh dxy ", np.size(dxy))
    # Reshape to 1 second
    rs_dxy = np.reshape(dxy, [int(np.size(dxy) / binsz), binsz])
    # print("sh rs_dxy ", rs_dxy.shape)

    # median for each second
    med = np.median(rs_dxy, axis=1)

    # Changing to 4s bins
    binned_mot = np.nanmean(np.reshape(med, (900, 4)), axis=1)

    # Nan is set to zero
    # binned_mot[np.where(np.isnan(binned_mot))] = 0
    # raw_var[np.where(np.isnan(raw_var))] = 0
    # print("sh binned_mot ", binned_mot.shape)
    return binned_mot, raw_var, dxy, med, dt


def normMean(meanEEG_perhr, var_EEG_perhr, hr):
    if int(hr) - 1 < 12:
        normmean = np.mean(meanEEG_perhr[0:24])
        normvar = np.mean(var_EEG_perhr[0:24])
    elif np.size(meanEEG_perhr) - int(hr) < 12:
        normmean = np.mean(meanEEG_perhr[np.size(meanEEG_perhr) - 24:
                                         np.size(meanEEG_perhr)])
        normvar = np.mean(var_EEG_perhr[np.size(var_EEG_perhr) - 24:
                                        np.size(var_EEG_perhr)])
    else:
        normmean = np.mean(meanEEG_perhr[int(hr) - 12: int(hr) + 12])
        normvar = np.mean(var_EEG_perhr[int(hr) - 12: int(hr) + 12])
    normstd = np.sqrt(normvar)
    return normmean, normstd


def generate_EEG(downdatlfp, bin, fs, normmean, normstd):
    # binl = bin * fs  # bin size in array slots
    EEGamp = np.zeros(int(np.size(downdatlfp) / (4 * fs)))
    EEGmean = np.zeros(int(np.size(downdatlfp) / (4 * fs)))
    for i in np.arange(np.size(EEGamp)):
        EEGamp[i] = np.var(downdatlfp[4 * fs * (i):(4 * fs * (i + 1))])
        EEGmean[i] = np.mean(np.abs(downdatlfp[4 * fs * (i):
                                    (4 * fs * (i + 1))]))
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
    EEGreshape = np.reshape(downdatlfp, (-1, fs*epochlen))
    freqs, psd = signal.welch(EEGreshape, fs, nperseg=win, scaling='density')
    idx_min = np.argmax(freqs > low) - 1
    idx_max = np.argmax(freqs > high) - 1
    idx = np.zeros(dtype=bool, shape=freqs.shape)
    idx[idx_min:idx_max] = True
    EEG = simps(psd[:, idx], freqs[idx])
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


def fix_states(states, alter_nums=False):
    if alter_nums:
        states[states == 1] = 0
        states[states == 3] = 5

    for ss in np.arange(np.size(states)-1):
        # check if it is a flicker state
        if (ss != 0 and ss < np.size(states)-1):
            if states[ss+1] == states[ss-1]:
                states[ss] = states[ss+1]

            if (states[ss] == 0 and states[ss+1] == 5):
                states[ss] = 2
    if alter_nums:
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
    if video_key is not False:
        title = [np.unique(video_key[1, :])[i][0:42] + ' \n '
                 for i in np.arange(np.size(np.unique(video_key[1, :])))]
        title = ''.join(title)
        title = title[0:-3]
    else:
        title = "there is no video file, later fix pulls it from the DLC data"
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
            rect = patch.Rectangle((x, y), width, h, color='#b7e1a1',
                                   alpha=0.5)
            ax.add_patch(rect)
    ax.set_xlim([0, 60])


def plot_spectrogram(ax, rawdat_dir, hr):
    ax.set_title('Spectrogram w/ EMG')
    img = mpimg.imread(rawdat_dir + 'specthr' + hr + '.jpg')
    ax.imshow(img, aspect='auto')
    ax.set_xlim(199, 1441)
    ax.set_ylim(178, 0)
    ax.set_xticks(np.linspace(199, 1441, 13))
    ax.set_xticklabels(np.arange(0, 65, 5))
    ticksy = [35, 100, 150]
    labelsy = [60, 6, 2]
    ax.set_yticks(ticksy, labelsy)


def plot_predicted(ax, Predict_y, clf, Features):
    if clf is not None:
        ax.set_title('Predicted States')
    else:
        ax.set_title('Loaded Scores')
    # 1 – Active Wake, Green
    # 2 – NREM, Blue
    # 3 – REM, red
    # 5 – QW, White
    for state in np.arange(np.size(Predict_y)):
        if Predict_y[state] == 0:
            rect7 = patch.Rectangle((state, 0), 3.8,
                                    height=1, color='green')
            ax.add_patch(rect7)
        elif Predict_y[state] == 2:
            rect7 = patch.Rectangle((state, 0), 3.8,
                                    height=1, color='blue')
            ax.add_patch(rect7)
        elif Predict_y[state] == 5:
            rect7 = patch.Rectangle((state, 0), 3.8,
                                    height=1, color='red')
            ax.add_patch(rect7)
        elif Predict_y[state] == 4:
            rect7 = patch.Rectangle((state, 0), 3.8,
                                    height=1, color='#a8a485')
            ax.add_patch(rect7)
    ax.set_ylim(0.3, 1)
    ax.set_xlim(0, 900)
    if clf is not None:
        predictions = clf.predict_proba(Features)
        confidence = np.max(predictions, 1)
        ax.plot(confidence, color='k')


def create_prediction_figure(rawdat_dir, hr, Predict_y, clf,
                             Features, pos, med=False, video_key=False):
    plt.ion()
    fig, (ax1, ax2, ax3) = \
        plt.subplots(nrows=3, ncols=1, figsize=(11, 6))
    plot_spectrogram(ax1, rawdat_dir, hr)
    plot_predicted(ax2, Predict_y, clf, Features)
    if pos:
        plot_motion(ax3, med, video_key)
    fig.tight_layout()
    return fig, ax1, ax2, ax3


def update_sleep_model(model_dir, mod_name, df_additions):
    try:
        Sleep_Model = np.load(file=model_dir + mod_name + '_model.pkl',
                              allow_pickle=True)
        Sleep_Model = Sleep_Model.append(df_additions, ignore_index=True)
    except FileNotFoundError:
        print('no model created...I will save this one')
        df_additions.to_pickle(model_dir + mod_name + '_model.pkl')
        Sleep_Model = df_additions
    Sleep_Model.to_pickle(model_dir + mod_name + '_model.pkl')
    return Sleep_Model


def load_joblib(FeatureList, ymot, yemg, mod_name):
    x_features = copy.deepcopy(FeatureList)
    [x_features.remove(i) for i in ['Animal_Name', 'Time_Interval', 'State']]
    jobname = ''
    if not ymot and yemg:
        x_features.remove('Motion')
        jobname = mod_name + '_EMG.joblib'

    if not yemg and ymot:
        x_features.remove('EMG')
        jobname = mod_name + '_Motion.joblib'

    if yemg and ymot:
        jobname = mod_name + '_Motion_EMG.joblib'

    if not yemg and not ymot:
        x_features.remove('EMG')
        x_features.remove('Motion')
        jobname = mod_name + '_no_move.joblib'
        print('Just so you know...this model has no EMG and no Motion')
    return jobname, x_features


def retrain_model(Sleep_Model, x_features, model_dir, jobname):
    prop = 1 / 2
    model_inputs = \
        Sleep_Model[x_features][0:
                                int((max(Sleep_Model.index) + 1) * prop)]\
        .apply(pd.to_numeric)
    train_x = model_inputs.values
    model_input_states = \
        Sleep_Model['State'][0:
                             int((max(Sleep_Model.index) + 1) * prop)]\
        .apply(pd.to_numeric)
    train_y = model_input_states.values

    model_test = \
        Sleep_Model[x_features][int((max(Sleep_Model.index) + 1) * prop):]\
        .apply(pd.to_numeric)
    test_x = model_test.values
    model_test_states = \
        Sleep_Model['State'][int((max(Sleep_Model.index) + 1) * prop):]\
        .apply(pd.to_numeric)
    test_y = model_test_states.values

    print('Calculating tree...')
    clf = random_forest_classifier(train_x, train_y)
    print("Train Accuracy :: ", accuracy_score(train_y, clf.predict(train_x)))
    print("Test Accuracy :: ", accuracy_score(test_y, clf.predict(test_x)))

    Satisfaction = input('Satisfied?: ')
    if Satisfaction == 'y':
        clf = \
            random_forest_classifier(Sleep_Model[x_features]
                                     .apply(pd.to_numeric).values,
                                     Sleep_Model['State']
                                     .apply(pd.to_numeric).values)
        print("Train Accuracy :: ",
              accuracy_score(Sleep_Model['State'].apply(pd.to_numeric).values,
                             clf.predict(Sleep_Model[x_features]
                             .apply(pd.to_numeric).values)))
        dump(clf, model_dir + jobname)


def pull_up_movie(start, end, vid_sample, video_key, motion_dir, fs,
                  epochlen, ratio2, dt):
    print(f'start: {start} end: {end}')
    vid_win_idx = np.where(np.logical_and(vid_sample >= start,
                                          vid_sample < end))[0]
    vid_win = video_key[2][vid_win_idx]
    if np.size(np.where(vid_win == 'nan')[0]) > 0:
        print('There is no video here')
        return
    else:
        vid_win = [int(float(i)) for i in vid_win]
        if np.size(np.unique(video_key[1][vid_win_idx])) > 1:
            print('This period is between two windows. No video to display')
            return
        else:
            vidfilename = np.unique(video_key[1][vid_win_idx])[0]
            score_win = \
                np.arange(int(vid_win[0]) + int(np.size(vid_win) / 3),
                          int(vid_win[0]) + int((np.size(vid_win) / 3) * 2))
    # x = (end - start) / ratio2
    # length = np.arange(int(end / x - start / x))
    # bottom = np.zeros(int(end / x - start / x))
    print('Pulling up video ....')
    cap = cv2.VideoCapture(motion_dir + vidfilename)
    if not cap.isOpened():
        print("Error opening video stream or file")
    for f in np.arange(vid_win[0], vid_win[-1]):
        cap.set(1, f)
        ret, frame = cap.read()
        if ret:
            if f in score_win:
                cv2.putText(frame, "SCORE WINDOW", (50, 105),
                            cv2.FONT_HERSHEY_PLAIN, 4, (225, 0, 0), 2)
            cv2.imshow('Frame', frame)
            cv2.waitKey(int((dt * 10e2) / 4))
    cap.release()


def pull_up_raw_trace(i, ax1, ax2, ax3, ax4, emg, start, end, realtime,
                      downdatlfp, fs, mod_name, LFP_ylim, delt,
                      theta, epochlen, EMGamp, ratio2):
    print('pull up the second figure for that bin ')
    print('\t - maybe. Option to click through a few bins around it?')
    x = (end - start) / ratio2
    length = np.arange(int(end / x - start / x))
    bottom = np.zeros(int(end / x - start / x))

    line1 = plot_LFP(start, end, ax1, downdatlfp, realtime, fs, LFP_ylim)
    line2 = plot_delta(delt, start, end, fs, ax2)
    line3 = plot_theta(ax3, start, end, fs, theta)

    if not emg:
        ax4.text(0.5, 0.5, 'There is no EMG')
    else:
        plot_EMG(i, ax4, length, bottom, EMGamp, epochlen, x, start, end)

    return line1, line2, line3


def plot_delta(delt, start, end, fs, ax):
    line2, = ax.plot(delt[start:end])
    ax.set_xlim(0, end-start)
    ax.set_ylim(np.min(delt), np.max(delt) / 3)
    bottom_2 = ax.get_ylim()[0]
    rectangle_2 = patch.Rectangle((fs*4, bottom_2),
                                  fs*4,
                                  height=float(-bottom_2/5))
    ax.add_patch(rectangle_2)
    ax.set_title('Delta power (0.5 - 4 Hz)')
    return line2


def plot_theta(ax, start, end, fs, theta):
    line3, = ax.plot(theta[start:end])
    ax.set_xlim(0, end-start)
    ax.set_ylim(np.min(theta), np.max(theta)/3)
    ax.set_title('Theta power (4 - 8 Hz)')
    bottom_3 = ax.get_ylim()[0]
    rectangle_3 = patch.Rectangle((fs * 4, bottom_3),
                                  fs * 4,
                                  height=-bottom_3 / 5)
    ax.add_patch(rectangle_3)
    return line3


def plot_LFP(start, end, ax, downdatlfp, realtime, fs, LFP_ylim):
    line1, = ax.plot(realtime[start:end], downdatlfp[start:end])
    ax.set_xlim(start/fs, end/fs)
    ax.set_title('LFP')
    ax.set_ylim(-LFP_ylim, LFP_ylim)
    bottom = -LFP_ylim
    rectangle = patch.Rectangle((start/fs+4, bottom),
                                4, height=-bottom/5)
    ax.add_patch(rectangle)
    return line1


def plot_EMG(i, ax, length, bottom, EMGamp, epochlen, x, start, end):
    # anything with EMG will error
    ax.fill_between(length, bottom,
                    EMGamp[int(i * 4 * epochlen):
                           int(i * 4 * epochlen) + int(end / x - start / x)],
                    color='red')
    ax.set_titel('EMG power')
    ax.set_xlim(0, int((end - start) / x) - 1)
    ax.set_ylim(-1, 5)


def clear_bins(bins, ax2):
    for b in np.arange(bins[0], bins[1]):
        b = math.floor(b)
        location = b
        rectangle = patch.Rectangle((location, 0), 1.5,
                                    height=2, color='turquoise')
        ax2.add_patch(rectangle)


def correct_bins(start_bin, end_bin, ax2, new_state):
    for b in np.arange(start_bin, end_bin):
        b = math.floor(b)
        location = b
        color = 'white'
        if new_state == 1:
            color = 'green'
        if new_state == 2:
            color = 'blue'
        if new_state == 3:
            color = 'red'
        rectangle = patch.Rectangle((location, 0), 1.5,
                                    height=2, color=color)
        print('loc: ', location)
        ax2.add_patch(rectangle)


def create_scoring_figure(rawdat_dir, hr, video_key, pos, med):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1,
                                        figsize=(11, 6))
    plot_spectrogram(ax1, rawdat_dir, hr)
    if pos:
        plot_motion(ax3, med, video_key)
    ax2.set_ylim(0.3, 1)
    ax2.set_xlim(0, 900)
    fig.show()
    fig.tight_layout()
    return fig, ax1, ax2, ax3


def update_raw_trace(line1, line2, line3, ax4, fig, start, end, i, downdatlfp,
                     delt, thet, fs, epochlen, emg, ratio2, EMGamp):
    line1.set_ydata(downdatlfp[start:end])
    line2.set_ydata(delt[start:end])
    line3.set_ydata(thet[start:end])
    x = (end - start) / ratio2
    length = np.arange(int(end / x - start / x))
    bottom = np.zeros(int(end / x - start / x))
    if emg:
        ax4.collections.clear()
        ax4.fill_between(length, bottom,
                         EMGamp[int(i * 4 * epochlen):
                                int(i * 4 * epochlen + 4 * 3 * epochlen)],
                         color='red')
    fig.canvas.draw()


def print_instructions():
    print('''\

                            .--,       .--,
                           ( (  \\.---.// ) )
                            '.__//o   o\\_.'
                                {=  ^  =}
                                 >  -  <
        ____________________.""`-------`"".________________________

                              INSTRUCTIONS

        Welcome to Sleep Wake Scoring!

        The figure you're looking at consists of 3 plots:
        1. The spectrogram for the hour you're scoring
        2. The random forest model's predicted states
        3. The binned motion for the hour

        TO CORRECT BINS:
        - click once on the middle figure to select the start of the bin you
          want to change
        - then click the last spot of the bin you want to change
        - now color change to turquoise of the selected region
        - switch to terminal and type the state you want that bin to become
            Valid values are 1, 2, 3, 4 and 5
              1 – Active Wake, Green
              2 – NREM, Blue
              3 – REM, red
              4 micro-arousal (not used often)
              5 – Quiet Wake, White

        VIDEO / RAW DATA:
        - if you hover over the motion figure you enter ~~ movie mode ~~
        - click on that figure where you want to pull up movie and the raw
          trace for the 4 seconds before, during, and after the point that you
          clicked

        CURSOR:
        - clicking figure (spectrogram) will created magenta dashed line across
          all 3 plots to check alignment.
        - clicking again removes old line and shows new line.

        EXITING SCORING:
        - are you done correcting bins?
        - are you sure?
        - are you going to come to me/clayton/lizzie and ask how you 'go back'
          and 'change a few more bins'?
        - think for a second and then, when you're sure, press 'd'
        - it will then ask you if you want to save your states and/or update
          the random forest model
            - choose wisely

        NOTES:
        - all keys pressed should be lowercase. don't 'shift + d'. just 'd'.
        - the video window along with the raw trace figure will remain up and
          update when you click a new bin don't worry about closing them or
          quitting them, it will probably error if you do.
        - slack me any errors if you get them or you have ideas for new
          functionality/GUI design
            - always looking to stay ~fresh~ with those ~graphics~
        - if something isn't working, make sure you're on Figure 2 and not the
          raw trace/terminal/video
        - plz don't toggle line while in motion axes, it messes up the axes
          limits, not sure why, working on it

        coming soon to sleep-wake code near you:
        - coding the state while you're slected in the figure, so you don't
          have to switch to terminal
        - automatically highlighting problem areas where the model isn't sure
          or a red flag is raised (going wake/rem/wake/rem)
        - letting you choose the best fitting model before you fix the states
          to limit the amont of corrections


        ANOUNCEMENTS:
        - if you're trying to code each bin individually (a.k.a. when it asks
                you if you want to correct the model you say 'no') it doesn't
                save afterward yet. you will have to manually save it after
                you're done for the time being

                                               ''')
