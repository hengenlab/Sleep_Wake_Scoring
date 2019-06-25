import numpy as np
import matplotlib.patches as patch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as signal
import glob
import copy
import sys
import os
import math
from lizzie_work import DLCMovement_input
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load
import pandas as pd
import warnings
from Sleep_Wake_Scoring import SW_utils

# this is bad. i feel bad doing it. but here we are
print('this code is supressing warnings because they were excessive and annoying. \nIf something weird is happening delete line 26 and try again\n')
warnings.filterwarnings("ignore")

def on_click(event):
    print(f'xdata:{event.xdata} x:{event.x} axes: {event.inaxes}')
    x_coordinates.append(math.floor(event.xdata))
    CLICKED[0] = True

def on_key(event):
    KEY.append(event.key)
    if event.key in [1,2,3,4]:
        State.append(event.key)


# 3 necessary directories
# maybe have these be inputs into the top level function instead of a line by line deal
rawdat_dir = '/Volumes/bs004r/EAB00040/EAB00040_2019-04-02_11-49-53_p9_c4/'
motion_dir = '/Volumes/bs004r/EAB00040/EAB00040_2019-04-02_11-49-53_p9_c4_labeled_vid/'
model_dir = '/Volumes/HlabShare/Sleep_Model/'

os.chdir(rawdat_dir)
meanEEG_perhr = np.load('Average_EEG_perhr.npy')
var_EEG_perhr = np.load('Var_EEG_perhr.npy')

animal = input('What animal is this?')
hr = input('What hour are you working on? (starts at 1): ')
mod_name = input('Which model? (young_rat, adult_rat, mouse, rat_mouse)')
epochlen = int(input('Epoch length: '))
fs = int(input('sampling rate: '))
emg = input('Do you have emg info? y/n: ') == 'y'
pos = input('Do you have a motion vector? y/n: ') == 'y'
vid = input('Do you have video? y/n: ') == 'y'

print('loading delta and theta...')
delt = np.load('delt' + hr + '.npy')
delt = np.concatenate((500 * [0], delt, 500 * [0]))
thet = np.load('thet' + hr + '.npy')
thet = np.concatenate((500 * [0], thet, 500 * [0]))
downdatlfp = np.load('EEGhr' + hr + '.npy')
ratio2 = 12 * 4

if pos:
    print('loading motion...')
    movement_files = np.sort(glob.glob(motion_dir + '*tmove.npy'))
    SW_utils.check_time_stamps(movement_files)
    movement = np.load(movement_files[int(hr) - 1])
    print('initializing motion...')
    binned_mot, raw_var, dxy, med,dt = SW_utils.init_motion(movement)
if vid:
    print('loading video...')
    vidkey_files = np.sort(glob.glob(motion_dir + '*vidkey.npy'))
    SW_utils.check_time_stamps(vidkey_files)
    video_key = np.load(vidkey_files[int(hr) - 1])
    vid_sample = np.linspace(0, 3600 * fs, np.size(video_key[0]))

if vid and pos:
    SW_utils.check_time_period(movement_files, vidkey_files)

if emg:
    print('loading EMG...')
    EMGamp = np.load('EMGhr' + hr + '.npy')
    EMGamp = (EMGamp - np.average(EMGamp)) / np.std(EMGamp)
    EMG = SW_utils.generate_EMG(EMGamp)

os.chdir(rawdat_dir)
normmean, normstd = SW_utils.normMean(meanEEG_perhr, var_EEG_perhr, hr)

print('Generating EEG vectors...')
EEGamp, EEGmax, EEGmean = SW_utils.generate_EEG(downdatlfp, epochlen, fs, normmean, normstd)

print('Extracting delta bandpower...')
EEGdelta, idx_delta = SW_utils.bandPower(0.5, 4, downdatlfp, epochlen, fs)

print('Extracting theta bandpower...')
EEGtheta, idx_theta = SW_utils.bandPower(4, 8, downdatlfp, epochlen, fs)

print('Extracting alpha bandpower...')
EEGalpha, idx_alpha = SW_utils.bandPower(8, 12, downdatlfp, epochlen, fs)

print('Extracting beta bandpower...')
EEGbeta, idx_beta = SW_utils.bandPower(12, 30, downdatlfp, epochlen, fs)

print('Extracting gamma bandpower...')
EEGgamma, idx_gamma = SW_utils.bandPower(30, 80, downdatlfp, epochlen, fs)

print('Extracting narrow-band theta bandpower...')
EEG_broadtheta, idx_broadtheta = SW_utils.bandPower(2, 16, downdatlfp, epochlen, fs)

print('Boom. Boom. FIYA POWER...')
EEGfire, idx_fire = SW_utils.bandPower(4, 20, downdatlfp, epochlen, fs)

EEGnb = EEGtheta / EEG_broadtheta
delt_thet = EEGdelta / EEGtheta

EEGdelta = SW_utils.normalize(EEGdelta)
EEGalpha = SW_utils.normalize(EEGalpha)
EEGbeta = SW_utils.normalize(EEGbeta)
EEGgamma = SW_utils.normalize(EEGbeta)
EEGnb = SW_utils.normalize(EEGnb)
EEGtheta = SW_utils.normalize(EEGtheta)
EEGfire = SW_utils.normalize(EEGfire)
delt_thet = SW_utils.normalize(delt_thet)

delta_post, delta_pre = SW_utils.post_pre(EEGdelta, EEGdelta)
theta_post, theta_pre = SW_utils.post_pre(EEGtheta, EEGtheta)
delta_post2, delta_pre2 = SW_utils.post_pre(delta_post, delta_pre)
theta_post2, theta_pre2 = SW_utils.post_pre(theta_post, theta_pre)
delta_post3, delta_pre3 = SW_utils.post_pre(delta_post2, delta_pre2)
theta_post3, theta_pre3 = SW_utils.post_pre(theta_post2, theta_pre2)
nb_post, nb_pre = SW_utils.post_pre(EEGnb, EEGnb)

animal_name = np.full(np.size(delta_pre), animal)
animal_num = np.full(np.shape(animal_name), int(animal[3:]))

model = input('Use a random forest? y/n: ') == 'y'

if model:
    final_features = ['Animal_Name', 'animal_num', 'Time_Interval', 'State', 'delta_pre', 'delta_pre2',
                      'delta_pre3', 'delta_post', 'delta_post2', 'delta_post3', 'EEGdelta', 'theta_pre',
                      'theta_pre2', 'theta_pre3',
                      'theta_post', 'theta_post2', 'theta_post3', 'EEGtheta', 'EEGalpha', 'EEGbeta',
                      'EEGgamma', 'EEGnarrow', 'nb_pre', 'delta/theta', 'EEGfire', 'EEGamp', 'EEGmax',
                      'EEGmean', 'EMG', 'Motion', 'raw_var']
    nans = np.full(np.shape(animal_name), np.nan)

    os.chdir(model_dir)
    if pos and emg:
        clf = load(mod_name + '_Motion_EMG.joblib')
    if not pos and emg:
        clf = load(mod_name + '_EMG.joblib')
    if pos and not emg:
        clf = load(mod_name + '_Motion.joblib')
    if not pos and not emg:
        clf = load(mod_name + '_no_move.joblib')

    # feature list
    FeatureList = []
    if pos and not emg:
        FeatureList = [animal_num, delta_pre, delta_pre2, delta_pre3, delta_post, delta_post2, delta_post3, EEGdelta,
                       theta_pre, theta_pre2, theta_pre3, theta_post, theta_post2, theta_post3,
                       EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre, delt_thet, EEGfire, EEGamp, EEGmax,
                       EEGmean, binned_mot, raw_var]
    elif not pos and not emg:
        FeatureList = [animal_num, delta_pre, delta_pre2, delta_pre3, delta_post, delta_post2, delta_post3, EEGdelta,
                       theta_pre, theta_pre2, theta_pre3, theta_post, theta_post2, theta_post3,
                       EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre, delt_thet, EEGfire, EEGamp, EEGmax,
                       EEGmean, nans, nans]

    elif not pos and emg:
        FeatureList = [animal_num, delta_pre, delta_pre2, delta_pre3, delta_post, delta_post2, delta_post3, EEGdelta,
                       theta_pre, theta_pre2, theta_pre3, theta_post, theta_post2, theta_post3,
                       EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre, delt_thet, EEGfire, EEGamp, EEGmax,
                       EEGmean, EMG, nans]
    elif pos and emg:
        FeatureList = [animal_num, delta_pre, delta_pre2, delta_pre3, delta_post, delta_post2, delta_post3, EEGdelta,
                       theta_pre, theta_pre2, theta_pre3, theta_post, theta_post2, theta_post3,
                       EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre, delt_thet, EEGfire, EEGamp, EEGmax,
                       EEGmean, EMG, binned_mot, raw_var]

    FeatureList_smoothed = []
    for f in FeatureList:
        FeatureList_smoothed.append(signal.medfilt(f, 5))
    Features = np.column_stack((FeatureList_smoothed))
    Predict_y = clf.predict(Features)
    Predict_y = SW_utils.fix_states(Predict_y)
    if pos:
        SW_utils.create_prediction_figure(rawdat_dir, hr, Predict_y, clf, Features, pos, med, video_key)
    else:
        SW_utils.create_prediction_figure(rawdat_dir, hr, Predict_y, clf, Features, pos)

    satisfaction = input('Satisfied?: y/n ') == 'y'
    plt.close('all')
    if satisfaction:
        filename = rawdat_dir + animal + '_SleepStates' + hr + '.npy'
        np.save(filename, Predict_y)
        State = Predict_y
        update = input('Update model?(y/n): ') == 'y'
        if update:
            ymot = input('Use motion?: y/n ') == 'y'
            yemg = input('Use EMG?: y/n ') == 'y'
            time_int = [video_key[1, i][0:26] for i in
                        np.arange(0, np.size(video_key[1, :]), int(np.size(video_key[1, :]) / np.size(animal_name)))]

            data = np.vstack(
                [animal_name, animal_num, time_int, State, delta_pre, delta_pre2, delta_pre3, delta_post,
                 delta_post2, delta_post3, EEGdelta, theta_pre, theta_pre2, theta_pre3, theta_post, theta_post2,
                 theta_post3,
                 EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre, delt_thet, EEGfire, EEGamp, EEGmax,
                 EEGmean])

            if yemg:
                if np.size(np.where(pd.isnull(EMG))[0]) > 0:
                    EMG[np.isnan(EMG)] = 0
            if ymot and not yemg:
                data = np.vstack([data, nans, binned_mot, raw_var])
            elif not ymot and not yemg:
                data = np.vstack([data, nans, nans, nans])
            elif not ymot and yemg:
                data = np.vstack([data, EMG, nans, nans])
            elif ymot and yemg:
                data = np.vstack([data, EMG, binned_mot, raw_var])

            df_additions = pd.DataFrame(columns = final_features, data = data.T)

            Sleep_Model = SW_utils.update_sleep_model(model_dir, mod_name, df_additions)
            jobname, x_features = SW_utils.load_joblib(final_features, ymot, yemg, mod_name)
            if yemg:
                Sleep_Model = Sleep_Model.drop(index = np.where(Sleep_Model['EMG'].isin(['nan']))[0])
            SW_utils.retrain_model(Sleep_Model, x_features, model_dir, jobname)
        sys.exit()
    fix = input('Do you want to fix the models states?: y/n')=='y'
    if fix:
        if pos:
            # this should probably be a different figure without the confidence line?
            fig, ax1, ax2, ax3 = SW_utils.create_prediction_figure(rawdat_dir, hr, Predict_y, clf, Features, pos, med, video_key)
        else:
            fig, ax1, ax2, ax3 = SW_utils.create_prediction_figure(rawdat_dir, hr, Predict_y, clf, Features, pos)
        plt.ion()
        State = copy.deepcopy(Predict_y)
        State[State == 0] = 1
        State[State == 2] = 2
        State[State == 5] = 3
        x_coordinates = []
        KEY = []
        CLICKED = []
        cID = fig.canvas.mpl_connect('button_press_event', on_click)
        cID2 = fig.canvas.mpl_connect('key_press_event', on_key)
        DONE = False
        while not DONE:
            print('select a bin to change or press "m" to view a section of the video. Press "d" when finished scoring the hour.')
            press = fig.waitforbuttonpress(timeout = 20)
            # true if key was pressed. false if mouse was clicked.
            if not press:
                fig.waitforbuttonpress(timeout = 20)
                # CLEAR THOSE STATES AND PLOT AS WHITE
                print(f'changing bins: {x_coordinates[0]} to {x_coordinates[1]}')
                for b in np.arange(x_coordinates[0], x_coordinates[1]):
                    b = math.floor(b)
                    location = b
                    rectangle = patch.Rectangle((location, 0), 3.8, height = 2, color = 'white')
                    ax2.add_patch(rectangle)
                fig.canvas.draw()
                new_state = int(input('What state should that bin be?'))
                print('cool. ill change that')
                # REPLOT THOSE STATES AS THE CORRECT COLOR AND MOVE ON
                start_bin = x_coordinates[0]
                end_bin = x_coordinates[1]
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
                    rectangle = patch.Rectangle((location, 0), 3.8, height = 2, color = color)
                    ax2.add_patch(rectangle)
                fig.canvas.draw()
                State[start_bin:end_bin] = new_state
                x_coordinates = []
            elif KEY[-1] == 'm':
                print('---- MOVIE MODE ----\npress "e" to exit this mode')
                if vid:
                    movie = True
                    while movie:
                        print('gon pull up some cute vids. click where you want to pull up the video')
                        plt.waitforbuttonpress(timeout = 20)
                        SW_utils.pull_up_movie(x_coordinates[-1], vid_sample, video_key, motion_dir)
                        if KEY[-1] == 'e':
                            cv2.destroyAllWindows()
                            print('-------- leaving movie mode ------- come back soon')
                            movie = False
                else:
                    print('you dont have video, sorry')
            elif KEY[-1] == 'd':
                DONE = True
        print('successfully left GUI')
        save_states = input('Would you like to save these sleep states?: y/n ') == 'y'
        if save_states:
            fileName = rawdat_dir + animal + '_SleepStates' + hr + '.npy'
            np.save(fileName, State)
        update = input('Would you like to update the model?: y/n ')=='y'
        if update:
            State[State == 1] = 0
            State[State == 2] = 2
            State[State == 3] = 5
            ymot = input('Use motion?: y/n ') == 'y'
            yemg = input('Use EMG?: y/n ') == 'y'
            time_int = [video_key[1, i][0:26] for i in
                        np.arange(0, np.size(video_key[1, :]), int(np.size(video_key[1, :]) / np.size(animal_name)))]
            data = np.vstack(
                [animal_name, animal_num, time_int, State, delta_pre, delta_pre2, delta_pre3, delta_post,
                 delta_post2, delta_post3, EEGdelta, theta_pre, theta_pre2, theta_pre3, theta_post, theta_post2,
                 theta_post3,
                 EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre, delt_thet, EEGfire, EEGamp, EEGmax,
                 EEGmean])
            if yemg:
                if np.size(np.where(pd.isnull(EMG))[0]) > 0:
                    EMG[np.isnan(EMG)] = 0
            if ymot and not yemg:
                data = np.vstack([data, nans, binned_mot, raw_var])
            elif not ymot and not yemg:
                data = np.vstack([data, nans, nans, nans])
            elif not ymot and yemg:
                data = np.vstack([data, EMG, nans, nans])
            elif ymot and yemg:
                data = np.vstack([data, EMG, binned_mot, raw_var])
            df_additions = pd.DataFrame(columns = final_features, data = data.T)

            Sleep_Model = SW_utils.update_sleep_model(model_dir, mod_name, df_additions)
            jobname, x_features = SW_utils.load_joblib(final_features, ymot, yemg, mod_name)
            if yemg:
                Sleep_Model = Sleep_Model.drop(index = np.where(Sleep_Model['EMG'].isin(['nan']))[0])
            SW_utils.retrain_model(Sleep_Model, x_features, model_dir, jobname)
    else:
        print('not fixing states. going to score the whole thing by hand. But with the models predictions and the bar thing')
else:
    print('not using the model. have to score by hand')