import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import glob
import copy
import os
import os.path as op
import json
from joblib import load
import cv2
import pandas as pd
import warnings
import Sleep_Wake_Scoring.SW_utils as SW_utils
from Sleep_Wake_Scoring.SW_Cursor import Cursor


def start_swscoring_v2(LFP_dir, motion_dir, model_dir, animal, mod_name,
                       epochlen, fs, emg, pos, vid, laccelerometer=0,
                       hr=1):
    print('this code is supressing warnings')
    warnings.filterwarnings("ignore")

    # Valid states you can enter
    valid_sw_states = [1, 2, 3, 4, 5]

    # Keep a copy of mod_name in case load_scores but need to update
    MOD_NAME_JSON = mod_name

    # Plot limit -250 to 250
    LFP_YLIM = 250

    if emg:
        # EMG_CHANNEL = 1

        EMGHIGHPASS = 20
        EMGLOWPASS = 200

    # os.chdir(LFP_dir)
    meanEEG_perhr = np.load('Average_EEG_perhr.npy')
    var_EEG_perhr = np.load('Var_EEG_perhr.npy')
    print("I am here 1.0.0", flush=True)

    # # animal = input('What animal is this?')
    # animal = str('KNR00004')
    # hr = input('What hour are you working on? (starts at 1): ')
    # # mod_name = input('Which model?
    # (young_rat, adult_rat, mouse, rat_mouse)')
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

    if pos:
        print('loading motion...')
        movement_files = np.sort(glob.glob(motion_dir + '*tmove.npy'))
        SW_utils.check_time_stamps(movement_files)
        movement = np.load(movement_files[int(hr) - 1])
        print('initializing motion...')
        binned_mot, raw_var, dxy, med, dt = SW_utils.init_motion(movement)
    if vid:
        print('loading video...')
        vidkey_files = np.sort(glob.glob(motion_dir + '*vidkey.npy'))
        SW_utils.check_time_stamps(vidkey_files)
        video_key = np.load(vidkey_files[int(hr) - 1])
        vid_sample = np.linspace(0, 3600 * fs, np.size(video_key[0]))

    if vid and pos:
        SW_utils.check_time_period(movement_files, vidkey_files)

    print("I am here 2.0.0", flush=True)
    lemg = 0
    if emg:
        print('loading EMG...')
        EMGamp = np.load('EMGhr' + hr + '.npy')
        if laccelerometer:
            print("sh EMGamp ", EMGamp.shape)
            accelerometer_h = np.load('ACC' + hr + '.npy')
            print("sh accelerometer_h ", accelerometer_h.shape)
            EMGamp = np.vstack((EMGamp, accelerometer_h))

        # EMGamp = EMGamp[EMG_CHANNEL, :]
        EMGamp = SW_utils.emg_preprocessing(EMGamp, fs, highpass=EMGHIGHPASS,
                                            lowpass=EMGLOWPASS)

        # emg save as temporarly avoid emg
        lemg = emg
        emg = 0

        if emg:
            # EMGamp = (EMGamp - np.average(EMGamp)) / np.std(EMGamp)
            EMG = SW_utils.generate_EMG(EMGamp)
            # EMGamp = np.pad(EMGamp, (0, 100), 'constant')

    else:
        EMGamp = False

    os.chdir(LFP_dir)
    normmean, normstd = SW_utils.normMean(meanEEG_perhr, var_EEG_perhr, hr)

    print('Generating EEG vectors...')
    EEGamp, EEGmax, EEGmean = SW_utils.generate_EEG(downdatlfp, epochlen,
                                                    fs, normmean, normstd)

    print('Extracting delta bandpower...')
    EEGdelta, idx_delta = SW_utils.bandPower(0.5, 4, downdatlfp, epochlen, fs)
    print("EEGdelta ", EEGdelta)
    print("sh EEGdelta ", EEGdelta.shape)

    print('Extracting theta bandpower...')
    EEGtheta, idx_theta = SW_utils.bandPower(4, 8, downdatlfp, epochlen, fs)

    print('Extracting alpha bandpower...')
    EEGalpha, idx_alpha = SW_utils.bandPower(8, 12, downdatlfp, epochlen, fs)

    print('Extracting beta bandpower...')
    EEGbeta, idx_beta = SW_utils.bandPower(12, 30, downdatlfp, epochlen, fs)

    print('Extracting gamma bandpower...')
    EEGgamma, idx_gamma = SW_utils.bandPower(30, 80, downdatlfp, epochlen, fs)

    print('Extracting narrow-band theta bandpower...')
    EEG_broadtheta, idx_broadtheta = SW_utils.bandPower(2, 16, downdatlfp,
                                                        epochlen, fs)

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
    print("I am here 3.0.0", flush=True)

    # New features list to add
    delta, theta, alpha,  beta, lgamma, hgamma, \
        delta1s, theta1s, alpha1s,  beta1s, lgamma1s, hgamma1s, \
        delta_n, theta_n, alpha_n,  beta_n, lgamma_n, hgamma_n = \
        SW_utils.calculate_features_from_lfp(downdatlfp, epochlen, fs)
    print("I am here 4.0.0", flush=True)

    print("sh delta1s ", delta1s.shape, flush=True)
    delta1s1 = delta1s[0::4]
    delta1s2 = delta1s[1::4]
    delta1s3 = delta1s[2::4]
    delta1s4 = delta1s[3::4]
    print("sh delta1s1 ", delta1s1.shape, flush=True)
    print("sh delta1s2 ", delta1s2.shape, flush=True)
    print("sh delta1s3 ", delta1s3.shape, flush=True)
    print("sh delta1s4 ", delta1s4.shape, flush=True)

    theta1s1 = theta1s[0::4]
    theta1s2 = theta1s[1::4]
    theta1s3 = theta1s[2::4]
    theta1s4 = theta1s[3::4]

    alpha1s1 = alpha1s[0::4]
    alpha1s2 = alpha1s[1::4]
    alpha1s3 = alpha1s[2::4]
    alpha1s4 = alpha1s[3::4]

    beta1s1 = beta1s[0::4]
    beta1s2 = beta1s[1::4]
    beta1s3 = beta1s[2::4]
    beta1s4 = beta1s[3::4]

    lgamma1s1 = lgamma1s[0::4]
    lgamma1s2 = lgamma1s[1::4]
    lgamma1s3 = lgamma1s[2::4]
    lgamma1s4 = lgamma1s[3::4]

    hgamma1s1 = hgamma1s[0::4]
    hgamma1s2 = hgamma1s[1::4]
    hgamma1s3 = hgamma1s[2::4]
    hgamma1s4 = hgamma1s[3::4]

    print("sh med ", med.shape, flush=True)
    binned_mot_med1 = med[0::4]
    binned_mot_med2 = med[0::4]
    binned_mot_med3 = med[0::4]
    binned_mot_med4 = med[0::4]
    print("sh binned_mot_med1 ", binned_mot_med1.shape, flush=True)

    # new set
    delta_by_all1s =\
        np.divide(delta1s,
                  (theta1s + alpha1s + beta1s + lgamma1s + hgamma1s))
    theta_by_all1s = \
        np.divide(theta1s,
                  (delta1s + alpha1s + beta1s + lgamma1s + hgamma1s))
    alpha_by_all1s = \
        np.divide(alpha1s,
                  (delta1s + theta1s + beta1s + lgamma1s + hgamma1s))
    beta_by_all1s = \
        np.divide(beta1s,
                  (delta1s + theta1s + alpha1s + lgamma1s + hgamma1s))
    lgamma_by_all1s = \
        np.divide(lgamma1s,
                  (delta1s + theta1s + alpha1s + beta1s + hgamma1s))
    hgamma_by_all1s = \
        np.divide(hgamma1s,
                  (delta1s + theta1s + alpha1s + beta1s + hgamma1s))

    delta_by_all1s1 = delta_by_all1s[0::4]
    delta_by_all1s2 = delta_by_all1s[1::4]
    delta_by_all1s3 = delta_by_all1s[2::4]
    delta_by_all1s4 = delta_by_all1s[3::4]

    theta_by_all1s1 = theta_by_all1s[0::4]
    theta_by_all1s2 = theta_by_all1s[1::4]
    theta_by_all1s3 = theta_by_all1s[2::4]
    theta_by_all1s4 = theta_by_all1s[3::4]

    alpha_by_all1s1 = alpha_by_all1s[0::4]
    alpha_by_all1s2 = alpha_by_all1s[1::4]
    alpha_by_all1s3 = alpha_by_all1s[2::4]
    alpha_by_all1s4 = alpha_by_all1s[3::4]

    beta_by_all1s1 = beta_by_all1s[0::4]
    beta_by_all1s2 = beta_by_all1s[1::4]
    beta_by_all1s3 = beta_by_all1s[2::4]
    beta_by_all1s4 = beta_by_all1s[3::4]

    lgamma_by_all1s1 = lgamma_by_all1s[0::4]
    lgamma_by_all1s2 = lgamma_by_all1s[1::4]
    lgamma_by_all1s3 = lgamma_by_all1s[2::4]
    lgamma_by_all1s4 = lgamma_by_all1s[3::4]

    hgamma_by_all1s1 = hgamma_by_all1s[0::4]
    hgamma_by_all1s2 = hgamma_by_all1s[1::4]
    hgamma_by_all1s3 = hgamma_by_all1s[2::4]
    hgamma_by_all1s4 = hgamma_by_all1s[3::4]

    # # delta
    # delta_by_theta = np.divide(delta, theta)
    # delta_by_alpha = np.divide(delta, alpha)
    # delta_by_beta = np.divide(delta, beta)
    # delta_by_lgamma = np.divide(delta, lgamma)
    # delta_by_hgamma = np.divide(delta, hgamma)

    # theta_by_delta = np.divide(theta, delta)
    # theta_by_alpha = np.divide(theta, alpha)
    # theta_by_beta = np.divide(theta, beta)
    # theta_by_lgamma = np.divide(theta, lgamma)
    # theta_by_hgamma = np.divide(theta, hgamma)

    # alpha_by_delta = np.divide(alpha, delta)
    # alpha_by_theta = np.divide(alpha, theta)
    # alpha_by_beta = np.divide(alpha, beta)
    # alpha_by_lgamma = np.divide(alpha, lgamma)
    # alpha_by_hgamma = np.divide(alpha, hgamma)

    # beta_by_delta = np.divide(beta, delta)
    # beta_by_theta = np.divide(beta, theta)
    # beta_by_alpha = np.divide(beta, alpha)
    # beta_by_lgamma = np.divide(beta, lgamma)
    # beta_by_hgamma = np.divide(beta, hgamma)

    # lgamma_by_delta = np.divide(lgamma, delta)
    # lgamma_by_theta = np.divide(lgamma, theta)
    # lgamma_by_alpha = np.divide(lgamma, alpha)
    # lgamma_by_beta = np.divide(lgamma, beta)
    # lgamma_by_hgamma = np.divide(lgamma, hgamma)

    # hgamma_by_delta = np.divide(hgamma, delta)
    # hgamma_by_theta = np.divide(hgamma, theta)
    # hgamma_by_alpha = np.divide(hgamma, alpha)
    # hgamma_by_beta = np.divide(hgamma, beta)
    # hgamma_by_lgamma = np.divide(hgamma, lgamma)

    # delta
    delta_by_theta = np.divide(delta_n, theta)
    delta_by_alpha = np.divide(delta_n, alpha)
    delta_by_beta = np.divide(delta_n, beta)
    delta_by_lgamma = np.divide(delta_n, lgamma)
    delta_by_hgamma = np.divide(delta_n, hgamma)

    theta_by_delta = np.divide(theta_n, delta)
    theta_by_alpha = np.divide(theta_n, alpha)
    theta_by_beta = np.divide(theta_n, beta)
    theta_by_lgamma = np.divide(theta_n, lgamma)
    theta_by_hgamma = np.divide(theta_n, hgamma)

    alpha_by_delta = np.divide(alpha_n, delta)
    alpha_by_theta = np.divide(alpha_n, theta)
    alpha_by_beta = np.divide(alpha_n, beta)
    alpha_by_lgamma = np.divide(alpha_n, lgamma)
    alpha_by_hgamma = np.divide(alpha_n, hgamma)

    beta_by_delta = np.divide(beta_n, delta)
    beta_by_theta = np.divide(beta_n, theta)
    beta_by_alpha = np.divide(beta_n, alpha)
    beta_by_lgamma = np.divide(beta_n, lgamma)
    beta_by_hgamma = np.divide(beta_n, hgamma)

    lgamma_by_delta = np.divide(lgamma_n, delta)
    lgamma_by_theta = np.divide(lgamma_n, theta)
    lgamma_by_alpha = np.divide(lgamma_n, alpha)
    lgamma_by_beta = np.divide(lgamma_n, beta)
    lgamma_by_hgamma = np.divide(lgamma_n, hgamma)

    hgamma_by_delta = np.divide(hgamma_n, delta)
    hgamma_by_theta = np.divide(hgamma_n, theta)
    hgamma_by_alpha = np.divide(hgamma_n, alpha)
    hgamma_by_beta = np.divide(hgamma_n, beta)
    hgamma_by_lgamma = np.divide(hgamma_n, lgamma)

    # model = input('Use a random forest? y/n: ') == 'y'
    model = 1
    if model:
        # final_features = ['Animal_Name', 'animal_num', 'Time_Interval',
        final_features = [
                          'State', 'delta_pre', 'delta_pre2',
                          'delta_pre3', 'delta_post', 'delta_post2',
                          'delta_post3', 'EEGdelta', 'theta_pre',
                          'theta_pre2', 'theta_pre3',
                          'theta_post', 'theta_post2', 'theta_post3',
                          'EEGtheta', 'EEGalpha', 'EEGbeta',
                          'EEGgamma', 'EEGnarrow', 'nb_pre',
                          'delta/theta', 'EEGfire', 'EEGamp', 'EEGmax',
                          'EEGmean', 'EMG', 'Motion', 'raw_var']

        final_features_v2 =\
            ['State',
             'delta', 'theta', 'alpha', 'beta', 'lgamma', 'hgamma',
             'delta1s1', 'theta1s1', 'alpha1s1', 'beta1s1',
             'lgamma1s1', 'hgamma1s1',
             'delta1s2', 'theta1s2', 'alpha1s2', 'beta1s2',
             'lgamma1s2', 'hgamma1s2',
             'delta1s3', 'theta1s3', 'alpha1s3', 'beta1s3',
             'lgamma1s3', 'hgamma1s3',
             'delta1s4', 'theta1s4', 'alpha1s4', 'beta1s4',
             'lgamma1s4', 'hgamma1s4',

             'delta_by_all1s1', 'delta_by_all1s2',
             'delta_by_all1s3', 'delta_by_all1s4',
             'theta_by_all1s1', 'theta_by_all1s2',
             'theta_by_all1s3', 'theta_by_all1s4',
             'alpha_by_all1s1', 'alpha_by_all1s2',
             'alpha_by_all1s3', 'alpha_by_all1s4',
             'beta_by_all1s1', 'beta_by_all1s2',
             'beta_by_all1s3', 'beta_by_all1s4',
             'lgamma_by_all1s1', 'lgamma_by_all1s2',
             'lgamma_by_all1s3', 'lgamma_by_all1s4',
             'hgamma_by_all1s1', 'hgamma_by_all1s2',
             'hgamma_by_all1s3', 'hgamma_by_all1s4',

             'delta_by_theta',
             'delta_by_alpha',
             'delta_by_beta',
             'delta_by_lgamma',
             'delta_by_hgamma',

             'theta_by_delta',
             'theta_by_alpha',
             'theta_by_beta',
             'theta_by_lgamma',
             'theta_by_hgamma',

             'alpha_by_delta',
             'alpha_by_theta',
             'alpha_by_beta',
             'alpha_by_lgamma',
             'alpha_by_hgamma',

             'beta_by_delta',
             'beta_by_theta',
             'beta_by_alpha',
             'beta_by_lgamma',
             'beta_by_hgamma',

             'lgamma_by_delta',
             'lgamma_by_theta',
             'lgamma_by_alpha',
             'lgamma_by_beta',
             'lgamma_by_hgamma',

             'hgamma_by_delta',
             'hgamma_by_theta',
             'hgamma_by_alpha',
             'hgamma_by_beta',
             'hgamma_by_lgamma',

             'EEGdelta',
             'EEGalpha',
             'EEGbeta',
             'EEGgamma',
             'EEGnb',
             'EEGtheta',
             'EEGfire',
             'delt_thet',

             'binned_mot_med1',
             'binned_mot_med2',
             'binned_mot_med3',
             'binned_mot_med4',
             'raw_var',
             'binned_mot']

        nans = np.full(np.shape(animal_name), np.nan)

        os.chdir(model_dir)
        mv_file = movement_files[int(hr)-1]
        t_stamp = mv_file[mv_file.find('_tmove')-18:mv_file.find('_tmove')]
        print("t_stamp ", t_stamp)
        filename = LFP_dir + animal + '_SleepStates_' + t_stamp + '.npy'
        print("filename ", filename, flush=True)
        if (os.path.exists(filename) and os.path.isfile(filename)):
            mod_name = "load_scores"
            print("\n\n\n", "=" * 42)
            print("Loading previously scored file")
            print("scored filename is : ", filename)
            Predict_y = np.load(filename)
            print("Human scored values are \n", Predict_y)
            print("=" * 42, "\n\n\n")
        else:
            print("\n\n\n", "=" * 42)
            print("Using random forest to predict states")
            print("model name is: ", mod_name)

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
                FeatureList = [animal_num, delta_pre, delta_pre2, delta_pre3,
                               delta_post, delta_post2, delta_post3, EEGdelta,
                               theta_pre, theta_pre2, theta_pre3, theta_post,
                               theta_post2, theta_post3,
                               EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb,
                               nb_pre, delt_thet, EEGfire, EEGamp, EEGmax,
                               EEGmean, binned_mot, raw_var]
                FeatureList_v2 =\
                    [
                     delta, theta, alpha, beta, lgamma, hgamma,
                     delta1s1, theta1s1, alpha1s1,  beta1s1,
                     lgamma1s1, hgamma1s1,
                     delta1s2, theta1s2, alpha1s2,  beta1s2,
                     lgamma1s2, hgamma1s2,
                     delta1s3, theta1s3, alpha1s3,  beta1s3,
                     lgamma1s3, hgamma1s3,
                     delta1s4, theta1s4, alpha1s4,  beta1s4,
                     lgamma1s4, hgamma1s4,

                     delta_by_all1s1, delta_by_all1s2, delta_by_all1s3,
                     delta_by_all1s4,
                     theta_by_all1s1, theta_by_all1s2, theta_by_all1s3,
                     theta_by_all1s4,
                     alpha_by_all1s1, alpha_by_all1s2, alpha_by_all1s3,
                     alpha_by_all1s4,
                     beta_by_all1s1, beta_by_all1s2, beta_by_all1s3,
                     beta_by_all1s4,
                     lgamma_by_all1s1, lgamma_by_all1s2, lgamma_by_all1s3,
                     lgamma_by_all1s4,
                     hgamma_by_all1s1, hgamma_by_all1s2, hgamma_by_all1s3,
                     hgamma_by_all1s4,

                     delta_by_theta,
                     delta_by_alpha,
                     delta_by_beta,
                     delta_by_lgamma,
                     delta_by_hgamma,

                     theta_by_delta,
                     theta_by_alpha,
                     theta_by_beta,
                     theta_by_lgamma,
                     theta_by_hgamma,

                     alpha_by_delta,
                     alpha_by_theta,
                     alpha_by_beta,
                     alpha_by_lgamma,
                     alpha_by_hgamma,

                     beta_by_delta,
                     beta_by_theta,
                     beta_by_alpha,
                     beta_by_lgamma,
                     beta_by_hgamma,

                     lgamma_by_delta,
                     lgamma_by_theta,
                     lgamma_by_alpha,
                     lgamma_by_beta,
                     lgamma_by_hgamma,

                     hgamma_by_delta,
                     hgamma_by_theta,
                     hgamma_by_alpha,
                     hgamma_by_beta,
                     hgamma_by_lgamma,

                     binned_mot_med1,
                     binned_mot_med2,
                     binned_mot_med3,
                     binned_mot_med4,
                     raw_var,
                     binned_mot]

            elif not pos and not emg:
                FeatureList = [animal_num, delta_pre, delta_pre2, delta_pre3,
                               delta_post, delta_post2, delta_post3, EEGdelta,
                               theta_pre, theta_pre2, theta_pre3, theta_post,
                               theta_post2, theta_post3,
                               EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb,
                               nb_pre, delt_thet, EEGfire, EEGamp, EEGmax,
                               EEGmean, nans, nans]

            elif not pos and emg:
                FeatureList = [animal_num, delta_pre, delta_pre2, delta_pre3,
                               delta_post, delta_post2, delta_post3, EEGdelta,
                               theta_pre, theta_pre2, theta_pre3, theta_post,
                               theta_post2, theta_post3,
                               EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb,
                               nb_pre, delt_thet, EEGfire, EEGamp, EEGmax,
                               EEGmean, EMG, nans]
            elif pos and emg:
                FeatureList = [animal_num, delta_pre, delta_pre2, delta_pre3,
                               delta_post, delta_post2, delta_post3, EEGdelta,
                               theta_pre, theta_pre2, theta_pre3, theta_post,
                               theta_post2, theta_post3,
                               EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb,
                               nb_pre, delt_thet, EEGfire, EEGamp, EEGmax,
                               EEGmean, EMG, binned_mot, raw_var]

            FeatureList_smoothed = []
            for f in FeatureList:
                FeatureList_smoothed.append(signal.medfilt(f, 5))
            Features = np.column_stack((FeatureList_smoothed))

            Features = np.nan_to_num(Features)

            # print("Type.......", type(Features))
            # temp_inf_test = np.isinf(Features)
            # print(np.where(temp_inf_test == 1))
            # temp_nan_test = np.isnan(Features)
            # print(np.where(temp_nan_test == 1))

            Predict_y = clf.predict(Features)
            # In random forest
            # 0 is wake but we save 1 as wake to numpy
            # 5 is REM but we save 3 as REM to numpy
            # so convert before proceeding further
            Predict_y[Predict_y == 0] = 1
            Predict_y[Predict_y == 5] = 3
            # Predict_y = SW_utils.fix_states(Predict_y)
            print("Random forest predicted values are \n", Predict_y)
            print("=" * 42, "\n\n\n")

        # fix = input('Do you want to fix the models states?: y/n')=='y'
        fix = 1
        plt.close('all')
        if fix:
            SW_utils.print_instructions()
            start = 0
            end = int(fs * 3 * epochlen)
            realtime = np.arange(np.size(downdatlfp)) / fs
            fig2, (ax4, ax5, ax6, ax7) = plt.subplots(nrows=4, ncols=1,
                                                      figsize=(11, 6))
            line1, line2, line3 = \
                SW_utils.pull_up_raw_trace(0, ax4, ax5, ax6, ax7, emg,
                                           start, end, realtime, downdatlfp,
                                           fs, mod_name, LFP_YLIM, delt, thet,
                                           epochlen, EMGamp, ratio2, fig=fig2)

            if mod_name == "load_scores":
                if pos:
                    if lemg:
                        # this should probably be a different figure without
                        # the confidence line?
                        fig, ax1, ax2, ax3 = \
                            SW_utils.create_prediction_figure(LFP_dir, hr,
                                                              Predict_y, None,
                                                              None, pos, med,
                                                              video_key,
                                                              newemg=EMGamp)
                    else:
                        # this should probably be a different figure without
                        # the confidence line?
                        fig, ax1, ax2, ax3 = \
                            SW_utils.create_prediction_figure(LFP_dir, hr,
                                                              Predict_y, None,
                                                              None, pos, med,
                                                              video_key)

                else:
                    fig, ax1, ax2, ax3 = \
                        SW_utils.create_prediction_figure(LFP_dir, hr,
                                                          Predict_y, None,
                                                          None, pos)
            else:
                if pos:
                    if lemg:
                        # this should probably be a different figure without
                        # the confidence line?
                        fig, ax1, ax2, ax3 = \
                            SW_utils.create_prediction_figure(LFP_dir, hr,
                                                              Predict_y, clf,
                                                              Features, pos,
                                                              med, video_key,
                                                              newemg=EMGamp)
                    else:
                        # this should probably be a different figure without
                        # the confidence line?
                        fig, ax1, ax2, ax3 = \
                            SW_utils.create_prediction_figure(LFP_dir, hr,
                                                              Predict_y, clf,
                                                              Features, pos,
                                                              med, video_key)

                else:
                    fig, ax1, ax2, ax3 = \
                        SW_utils.create_prediction_figure(LFP_dir, hr,
                                                          Predict_y, clf,
                                                          Features, pos)

            plt.ion()
            State = copy.deepcopy(Predict_y)
            cursor = Cursor(ax1, ax2, ax3)

            # cID = \
            # fig.canvas.mpl_connect('button_press_event', cursor.on_click)
            # cID2 = fig.canvas.mpl_connect('axes_enter_event', cursor.in_axes)
            # cID3 = fig.canvas.mpl_connect('key_press_event', cursor.on_press)
            fig.canvas.mpl_connect('button_press_event', cursor.on_click)
            # fig.canvas.mpl_connect('axes_enter_event', cursor.in_axes)
            fig.canvas.mpl_connect('key_press_event', cursor.on_press)

            print("sh State ", State.shape)
            print("sh delta ", delta.shape)
            data_tosave = \
                np.vstack([State,
                           delta, theta, alpha, beta, lgamma, hgamma,

                           delta1s1, theta1s1, alpha1s1,  beta1s1,
                           lgamma1s1, hgamma1s1,
                           delta1s2, theta1s2, alpha1s2,  beta1s2,
                           lgamma1s2, hgamma1s2,
                           delta1s3, theta1s3, alpha1s3,  beta1s3,
                           lgamma1s3, hgamma1s3,
                           delta1s4, theta1s4, alpha1s4,  beta1s4,
                           lgamma1s4, hgamma1s4,

                           delta_by_all1s1, delta_by_all1s2, delta_by_all1s3,
                           delta_by_all1s4,
                           theta_by_all1s1, theta_by_all1s2, theta_by_all1s3,
                           theta_by_all1s4,
                           alpha_by_all1s1, alpha_by_all1s2, alpha_by_all1s3,
                           alpha_by_all1s4,
                           beta_by_all1s1, beta_by_all1s2, beta_by_all1s3,
                           beta_by_all1s4,
                           lgamma_by_all1s1, lgamma_by_all1s2,
                           lgamma_by_all1s3,
                           lgamma_by_all1s4,
                           hgamma_by_all1s1, hgamma_by_all1s2,
                           hgamma_by_all1s3,
                           hgamma_by_all1s4,

                           delta_by_theta,
                           delta_by_alpha,
                           delta_by_beta,
                           delta_by_lgamma,
                           delta_by_hgamma,

                           theta_by_delta,
                           theta_by_alpha,
                           theta_by_beta,
                           theta_by_lgamma,
                           theta_by_hgamma,

                           alpha_by_delta,
                           alpha_by_theta,
                           alpha_by_beta,
                           alpha_by_lgamma,
                           alpha_by_hgamma,

                           beta_by_delta,
                           beta_by_theta,
                           beta_by_alpha,
                           beta_by_lgamma,
                           beta_by_hgamma,

                           lgamma_by_delta,
                           lgamma_by_theta,
                           lgamma_by_alpha,
                           lgamma_by_beta,
                           lgamma_by_hgamma,

                           hgamma_by_delta,
                           hgamma_by_theta,
                           hgamma_by_alpha,
                           hgamma_by_beta,
                           hgamma_by_lgamma,

                           EEGdelta,
                           EEGalpha,
                           EEGbeta,
                           EEGgamma,
                           EEGnb,
                           EEGtheta,
                           EEGfire,
                           delt_thet,

                           binned_mot_med1,
                           binned_mot_med2,
                           binned_mot_med3,
                           binned_mot_med4,
                           raw_var,
                           binned_mot])
            print("sh data_tosave ", data_tosave.shape)
            print("len final_features_v2 ", len(final_features_v2))
            df_tosave = \
                pd.DataFrame(data_tosave.T,
                             columns=final_features_v2)
            # dir_to_save =\
            #     '/hlabhome/kiranbn/git/Sleep_Wake_Scoring_p/datanewmodel/'
            dir_to_save =\
                '/media/HlabShare/ckbn/sleep_score_james_paper/data/'
            fl_to_save = op.join(dir_to_save, 'data_tosave.csv')
            if not op.isfile(fl_to_save):
                df_tosave.to_csv(fl_to_save,
                                 header=True,
                                 index=False)
            else:
                df_tosave.to_csv(fl_to_save,
                                 mode='a',
                                 index=False,
                                 header=False)
            plt.close('all')
            return 1

            plt.show()
            DONE = False
            while not DONE:
                plt.waitforbuttonpress()
                if cursor.change_bins:
                    bins = np.sort(cursor.bins)
                    start_bin = cursor.bins[0]
                    end_bin = cursor.bins[1] + 1
                    # print("start_bin ", start_bin, " end_bin ", end_bin)
                    print(f'changing bins: {start_bin} to {end_bin}')
                    SW_utils.clear_bins(bins, ax2)
                    fig.canvas.draw()
                    # new_state = int(input('What state should these be?: '))

                    # Loop until user enters valid integer
                    while True:
                        try:
                            new_state = \
                                int(input('What state should these be?: '))
                            if new_state not in valid_sw_states:
                                raise\
                                    ValueError('Not a valid value')
                        except Exception as e:
                            print("Error ", e)
                            print('''\
                                   Valid values are 1, 2, 3, 4 and 5
                                   # 1 – Active Wake, Green
                                   # 2 – NREM, Blue
                                   # 3 – REM, red
                                   # 4 micro-arousal (not used often)
                                   # 5 – Quiet Wake, White\n''')
                            continue
                        if new_state in valid_sw_states:
                            break

                    # new_state = \
                    # int(input('What state should these be?: '))
                    SW_utils.correct_bins(start_bin, end_bin, ax2, new_state)
                    fig.canvas.draw()
                    State[start_bin:end_bin] = new_state
                    cursor.bins = []
                    cursor.change_bins = False
                if cursor.movie_bin is not None:
                    if cursor.movie_mode and cursor.movie_bin > 0:
                        if vid:
                            start = int(cursor.movie_bin * 60 * fs)
                            end = int(((cursor.movie_bin * 60) + 12) * fs)
                            # i = 0
                            # SW_utils.update_raw_trace(line1, line2, line3,
                            #                           ax4,
                            #                           fig, start, end, i,
                            #                           downdatlfp, delt, thet,
                            #                           fs, epochlen, emg,
                            #                           ratio2, EMGamp)

                            realtime = np.arange(np.size(downdatlfp)) / fs
                            SW_utils.pull_up_raw_trace(0, ax4, ax5, ax6, ax7,
                                                       emg, start, end,
                                                       realtime, downdatlfp,
                                                       fs, mod_name,
                                                       LFP_YLIM, delt, thet,
                                                       epochlen, EMGamp,
                                                       ratio2, fig=fig2)
                            # fig2.canvas.draw()
                            # fig2.tight_layout()
                            SW_utils.pull_up_movie(start, end, vid_sample,
                                                   video_key, motion_dir,
                                                   fs, epochlen, ratio2, dt)
                            cursor.movie_bin = 0

                        else:
                            print("no video, please extract motion using dlc")
                else:
                    print("click inside the motion plot")
                if cursor.DONE:
                    DONE = True

            print('successfully left GUI')
            cv2.destroyAllWindows()
            plt.close('all')
            save_states = \
                input('Would you like to save these sleep states?:y/n ') == 'y'
            if save_states:
                # Update human scored Sleep states file
                if mod_name == "load_scores":
                    mv_file = movement_files[int(hr) - 1]
                    t_stamp = \
                        mv_file[mv_file.find('_tmove') - 18:
                                mv_file.find('_tmove')]
                    loverwrite = \
                        input('Overwrite these sleep states?: y/n ') == 'y'
                    if loverwrite:
                        filename = LFP_dir + animal + '_SleepStates_' +\
                            t_stamp + '.npy'
                        print("Saving ", filename, flush=True)
                        np.save(filename, State)
                    else:
                        username = input('Enter username initials/condition?:')
                        print("username ", username)
                        filename = LFP_dir + animal + '_SleepStates_' +\
                            t_stamp + '_' + str(username) + '.npy'
                        print("Saving ", filename, flush=True)
                        np.save(filename, State)
                # Save the human scored file generated by random forest
                else:
                    mv_file = movement_files[int(hr) - 1]
                    t_stamp = \
                        mv_file[mv_file.find('_tmove') - 18:
                                mv_file.find('_tmove')]
                    filename = \
                        LFP_dir + animal + '_SleepStates_' +\
                        t_stamp + '.npy'
                    print("Saving ", filename, flush=True)
                    np.save(filename, State)

            update = \
                input('Would you like to update the model?: y/n ') == 'y'
            # Reorder to random forest's values where
            # 0: wake
            # 2 : NREM
            # 5 : REM
            if update:
                # change mod_name
                if mod_name == "load_scores":
                    mod_name = MOD_NAME_JSON
                print("mod_name ", mod_name)

                State[State == 1] = 0
                State[State == 2] = 2
                State[State == 3] = 5
                ymot = input('Use motion?: y/n ') == 'y'
                yemg = input('Use EMG?: y/n ') == 'y'
                # time_int = [video_key[1, i][0:26] for i in
                #             np.arange(0, np.size(video_key[1, :]),
                #                       int(np.size(video_key[1, :]) /
                #                           np.size(animal_name)))]
                # data = np.vstack(
                #     [animal_name, animal_num, time_int, State, delta_pre,
                data = np.vstack(
                    [State, delta_pre,
                     delta_pre2, delta_pre3, delta_post,
                     delta_post2, delta_post3, EEGdelta, theta_pre, theta_pre2,
                     theta_pre3, theta_post, theta_post2,
                     theta_post3,
                     EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre,
                     delt_thet, EEGfire, EEGamp, EEGmax,
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
                df_additions = \
                    pd.DataFrame(columns=final_features, data=data.T)
                # change nans by mean
                df_additions = df_additions.fillna(df_additions.mean())
                # change to 0 if entire column is nan
                df_additions = df_additions.fillna(0)
                # for x in df_additions:
                #     if df_additions[x].dtypes == "int64":
                #         df_additions[x] = df_additions[x].astype(float)

                Sleep_Model = \
                    SW_utils.update_sleep_model(model_dir, mod_name,
                                                df_additions)
                jobname, x_features = \
                    SW_utils.load_joblib(final_features, ymot, yemg, mod_name)
                if yemg:
                    Sleep_Model = \
                        Sleep_Model.drop(index=np.where(Sleep_Model['EMG']
                                                        .isin(['nan']))[0])
                SW_utils.retrain_model(Sleep_Model, x_features,
                                       model_dir, jobname)


def load_data_for_sw_v2(filename_sw, hr):
    '''
     load_data_for_sw(filename_sw)

    LFP_dir, motion_dir, model_dir, animal, mod_name,
    epochlen, fs, emg, pos, vid
    '''

    # constant as 500 is used in sorting code
    RECBLOCK_STRUCTURE_FS = 500

    with open(filename_sw, 'r') as f:
        d = json.load(f)

    LFP_dir = str(d['LFP_dir'])
    motion_dir = str(d['motion_dir'])
    model_dir = str(d['model_dir'])
    animal = str(d['animal'])
    mod_name = str(d['mod_name'])
    epochlen = int(d['epochlen'])
    fs = int(d['fs'])
    emg = int(d['emg'])
    pos = int(d['pos'])
    vid = int(d['vid'])
    # fr = int(d['video_fr'])
    accelerometer = str(d['accelerometer'])
    if accelerometer == "None":
        print("None accelerometer ", accelerometer)
        laccelerometer = 0
    else:
        laccelerometer = 1

    try:
        recblock_structure = str(d['recblock_structure'])
    except KeyError as e:
        print("recblock_structure is not used", e)
        recblock_structure = None

    # if recblock_structure append to LFP_dir
    if recblock_structure is not None:
        #  create a base name based on recblock_structure
        base_dir_name = \
            (recblock_structure.replace(op.sep, '_').split('*')[0]
                .replace('_', ''))
        LFP_dir = op.join(LFP_dir, base_dir_name + op.sep)
        print("LFP_dir ", LFP_dir)

    # if recblock_structure is not None:
        if fs != RECBLOCK_STRUCTURE_FS:
            raise ValueError('Please check fs expected 500')

    os.chdir(LFP_dir)
    start_swscoring_v2(LFP_dir, motion_dir, model_dir,
                       animal, mod_name,
                       epochlen, fs, emg, pos, vid,
                       laccelerometer=laccelerometer,
                       hr=hr)
