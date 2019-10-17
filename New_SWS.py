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
import json
# from lizzie_work import DLCMovement_input
import DLCMovement_input
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load
import pandas as pd
import warnings
# from Sleep_Wake_Scoring import SW_utils
import SW_utils
# from Sleep_Wake_Scoring import Cursor
from SW_Cursor import Cursor

def on_press(event):
    if event.key in ['1','2','3']:
        State[i] = int(event.key)
        print(f'scored: {event.key}')



def start_swscoring(rawdat_dir, motion_dir, model_dir, animal, mod_name,
                    epochlen, fs, emg, pos, vid):
    print('this code is supressing warnings')
    warnings.filterwarnings("ignore")

    # #
    # # rawdat_dir = '/Volumes/carina/EAB00047/EAB00047_2019-06-27_15-14-19_p9_c5/'
    # # motion_dir = '/Volumes/carina/EAB00047/EAB00047_2019-06-27_15-14-19_p9_c5_labeled_video/6_28_first_12/'

    # # rawdat_dir = '/Volumes/bs005r/EAB00047/EAB00047_2019-06-10_15-11-36_p10_c4/'
    # # motion_dir = '/Volumes/bs005r/EAB00047/EAB00047_2019-06-10_15-11-36_p10_c4_labeled_video/'

    # rawdat_dir = '/media/bs004r/KNR00004/KNR00004_2019-08-01_16-43-45_p1_c3/'
    # motion_dir = '/media/bs004r/KNR00004/KNR00004_2019-08-01_16-43-45_p1_c3_labeled_video/'
    # model_dir = '/media/HlabShare/Sleep_Model/'

    # os.chdir(rawdat_dir)
    meanEEG_perhr = np.load('Average_EEG_perhr.npy')
    var_EEG_perhr = np.load('Var_EEG_perhr.npy')

    # # animal = input('What animal is this?')
    # animal = str('KNR00004')
    hr = input('What hour are you working on? (starts at 1): ')
    # # mod_name = input('Which model? (young_rat, adult_rat, mouse, rat_mouse)')
    # mod_name = str('rat_mouse')
    # # epochlen = int(input('Epoch length: '))
    # epochlen = int(4)
    # # fs = int(input('sampling rate: '))
    # fs = int(200)
    # # emg = input('Do you have emg info? y/n: ') == 'y'
    # emg = 0
    # # pos = input('Do you have a motion vector? y/n: ') == 'y'
    # pos = 1
    # # vid = input('Do you have video? y/n: ') == 'y'
    # vid = 1

    print('loading delta and theta...')
    delt = np.load('delt' + hr + '.npy')
    delt = np.concatenate((500 * [0], delt, 500 * [0]))
    thet = np.load('thet' + hr + '.npy')
    thet = np.concatenate((500 * [0], thet, 500 * [0]))
    downdatlfp = np.load('EEGhr' + hr + '.npy')
    ratio2 = 12 * 4

    if mod_name == 'mouse':
        LFP_ylim = 1000
    else:
        LFP_ylim = 250

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
        EMGamp = np.pad(EMGamp, (0, 100), 'constant')
    else:
        EMGamp = False

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

    # model = input('Use a random forest? y/n: ') == 'y'
    model = 1

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

        # satisfaction = input('Satisfied?: y/n ') == 'y'
        plt.close('all')
        satisfaction = 0
        if satisfaction:
            mv_file = movement_files[int(hr)-1]
            t_stamp = mv_file[mv_file.find('_tmove')-18:mv_file.find('_tmove')]
            filename = rawdat_dir + animal + '_SleepStates_' + t_stamp + '.npy'
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
        # fix = input('Do you want to fix the models states?: y/n')=='y'
        fix = 1
        if fix:
            SW_utils.print_instructions()
            start = 0
            end = int(fs * 3 * epochlen)
            realtime = np.arange(np.size(downdatlfp)) / fs
            fig2, (ax4, ax5, ax6, ax7) = plt.subplots(nrows = 4, ncols = 1, figsize = (11,6))
            line1, line2, line3 = SW_utils.pull_up_raw_trace(0, ax4, ax5, ax6, ax7, emg, start, end, realtime, downdatlfp, fs, mod_name, LFP_ylim, delt, thet, epochlen, EMGamp, ratio2)
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
            cursor = Cursor(ax1, ax2, ax3)

            cID = fig.canvas.mpl_connect('button_press_event', cursor.on_click)
            cID2 = fig.canvas.mpl_connect('axes_enter_event', cursor.in_axes)
            cID3 = fig.canvas.mpl_connect('key_press_event', cursor.on_press)

            plt.show()
            DONE = False
            while not DONE:
                plt.waitforbuttonpress()
                if cursor.change_bins:
                    bins = np.sort(cursor.bins)
                    start_bin = cursor.bins[0]
                    end_bin = cursor.bins[1]
                    print(f'changing bins: {start_bin} to {end_bin}')
                    SW_utils.clear_bins(bins, ax2)
                    fig.canvas.draw()
                    new_state = int(input('What state should these be?: '))
                    SW_utils.correct_bins(start_bin, end_bin, ax2, new_state)
                    fig.canvas.draw()
                    State[start_bin:end_bin] = new_state
                    cursor.bins = []
                    cursor.change_bins = False
                if cursor.movie_mode and cursor.movie_bin>0:
                    if vid:
                        start = int(cursor.movie_bin * 60 * fs)
                        end = int(((cursor.movie_bin * 60) + 12) * fs)
                        i=0
                        SW_utils.update_raw_trace(line1, line2, line3, ax4, fig, start, end,i, downdatlfp, delt, thet, fs, epochlen, emg, ratio2, EMGamp)
                        fig2.canvas.draw()
                        fig2.tight_layout()
                        SW_utils.pull_up_movie(start, end, vid_sample, video_key, motion_dir, fs, epochlen, ratio2, dt)
                        cursor.movie_bin = 0

                    else:
                        print("you don't have video, sorry")
                if cursor.DONE:
                    DONE = True

            print('successfully left GUI')
            cv2.destroyAllWindows()
            plt.close('all')
            save_states = input('Would you like to save these sleep states?: y/n ') == 'y'
            if save_states:
                mv_file = movement_files[int(hr) - 1]
                t_stamp = mv_file[mv_file.find('_tmove') - 18:mv_file.find('_tmove')]
                filename = rawdat_dir + animal + '_SleepStates_' + t_stamp + '.npy'
                np.save(filename, State)
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
            # show the model's prediction somewhere
            # connect the cursor again, but make it like the original where just if you click it pulls up the video
            # make the 2 figures. then just redraw the figs every time
            # consider making the LFP_ylim a variable cause it can be variable

            #register clicker for button press


            fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = 4, ncols = 1, figsize = (11,6))
            fig2, ax5, ax6, ax7 = SW_utils.create_scoring_figure(rawdat_dir, hr, video_key, pos, med)
            cursor = Cursor(ax5, ax6, ax7)
            cID3 = fig2.canvas.mpl_connect('key_press_event', on_press)

            i = 0
            start = int(i * fs * epochlen)
            end = int(i * fs * epochlen + fs * 3 * epochlen)
            realtime = np.arange(np.size(downdatlfp)) / fs
            vid_win_idx = np.where(np.logical_and(vid_sample >= start, vid_sample < end))[0]
            vid_win = video_key[2][vid_win_idx]
            line1, line2, line3 = SW_utils.pull_up_raw_trace(i, ax1, ax2, ax3,ax4, emg, start, end, realtime, downdatlfp, fs, mod_name, LFP_ylim, delt, thet, epochlen, EMGamp, ratio2)
            fig.show()
            fig2.show()
            plt.tight_layout()
            State = np.zeros(900)


            for i in range(0,899):
                # input('press enter or quit')
                print(f'here. index: {i}')
                start = int(i * fs * epochlen)
                end = int(i * fs * epochlen + fs * 3 * epochlen)
                vid_win_idx = np.where(np.logical_and(vid_sample >= start, vid_sample < end))[0]
                vid_win = video_key[2][vid_win_idx]
                SW_utils.update_raw_trace(line1, line2, line3, ax4, fig, start, end,i, downdatlfp, delt, thet, fs, epochlen, emg, ratio2, EMGamp)
                if model:
                    Prediction = clf.predict(Features[int(i),:].reshape(1,-1))
                    Predictions = clf.predict_proba(Features[int(i), :].reshape(1, -1))
                    confidence = np.max(Predictions, 1)
                    t1 = ax1.text(1800, 1, str(Prediction))
                    t2 = ax1.text(1800, 0.75, str(confidence))
                    fig.canvas.draw()
                    print('done w model stuff')
                fig.show()
                fig2.show()
                button = False
                while not button:
                    button = fig2.waitforbuttonpress()
                    print('here1')
                    print(f'button: {button}')
                    if not button:
                        SW_utils.pull_up_movie(start, end, vid_sample, video_key, motion_dir, fs, epochlen, ratio2, dt)
                    else:
                        print(f'about to correct bins. Index: {i}')
                        SW_utils.correct_bins(i, i+1, ax6, State[i])
                        fig2.canvas.draw()
                fig2.canvas.flush_events()
                fig.canvas.flush_events()
            print('DONE SCORING')
            cv2.destroyAllWindows()
            plt.close('all')
            last_state = int(input('Enter the last state: '))
            State[-2:] = last_state


            # save and update stuff here !!!

    else:
        print('not using the model. have to score by hand. just copy the last bit of code and put it here')

# rawdat_dir, motion_dir, model_dir, animal, mod_name, epochlen, fs, emg, pos, vid
def load_data_for_sw(filename_sw):
    '''
     load_data_for_sw(filename_sw)
    '''
    with open(filename_sw, 'r') as f:
           d = json.load(f)

    rawdat_dir = str(d['rawdat_dir'])
    motion_dir = str(d['motion_dir'])
    model_dir = str(d['model_dir'])
    animal = str(d['animal'])
    mod_name = str(d['mod_name'])
    epochlen = int(d['epochlen'])
    fs = int(d['fs'])
    emg = int(d['emg'])
    pos = int(d['pos'])
    vid = int(d['vid'])

    os.chdir(rawdat_dir)
    start_swscoring(rawdat_dir, motion_dir, model_dir, animal, mod_name,\
                    epochlen, fs, emg, pos, vid)


