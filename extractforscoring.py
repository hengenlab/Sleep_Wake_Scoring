import os
from scipy.signal import savgol_filter
import numpy as np
import scipy.signal as signal
from scipy.integrate import simps
import matplotlib.patches as patch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import neuraltoolkit as ntk
import seaborn as sns
import sys
import time as timer
import glob
import DLCMovement_input
import psutil
import math
import sys
import json
from findPulse import findPulse
from matplotlib import cm
from sys import platform
from Sleep_Wake_Scoring import LFP_check as lfpcheck
import gc

def extract_lfp(filename_sw):
    if platform == "darwin":

        plt.switch_backend('TkAgg')
    else:

        plt.switch_backend('Agg')

    with open(filename_sw, 'r') as f:
           d = json.load(f)

    rawdat_dir = str(d['rawdat_dir'])
    motion_dir = str(d['motion_dir'])
    model_dir = str(d['model_dir'])
    LFP_dir = str(d['LFP_dir'])
    digi_dir = str(d['digi_dir'])
    animal = str(d['animal'])
    mod_name = str(d['mod_name'])
    epochlen = int(d['epochlen'])
    fs = int(d['fs'])
    emg = int(d['emg'])
    pos = int(d['pos'])
    vid = int(d['vid'])
    num = int(d['num'])
    num_labels = int(d['num_labels'])
    cort = int(d['cort'])
    EMGinput = int(d['EMGinput'])
    numchan = int(d['numchan'])
    HS = d['HS']
    LFP_check = int(d['LFP_check'])
    probenum = int(d['probenum'])
    nprobes = int(d['nprobes'])

    print(digi_dir)
    print(motion_dir)
    print(rawdat_dir)

 
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

    reclen = 3600

    silicon_flag = 0
    print('probe number is {}'.format(probenum))
    chan_map = ntk.find_channel_map(HS[probenum],64)

    if cort is not 1:
        try:
            selected_chans = np.load(LFP_dir+'LFP_chancheck/selected_channels.npy')
            # LFP_check = input('Have you checked the average LFP that you are about to use? (y/n)')
            if LFP_check is not 1:
                # hour = int(input('what hour did you use?'))
                # sw.confirm_channels(selected_chans, raw_datdir, HS, hour)
                print('Go find the individual and average spectrograms of your LFP in ' + LFP_dir+'LFP_chancheck')
                sys.exit()
        except FileNotFoundError:
            LFP_check = input('You have not selected LFP channels, would you like to do that now? (y/n)')
            if LFP_check == 'y':
                hour = int(input('what hour will you use?'))
                #good_chans = [9, 14, 25, 32, 49, 57, 48]
                lfpcheck.selectLFPchan(filename_sw, hour)
                sys.exit('Exiting program now. Please run plot_LFP on local computer to choose cells')
            if LFP_check == 'n':
                sys.exit('Ok, I am exiting then')

    else:
        EEG = int(input('Enter EEG channel: ')) - 1
        EEG = np.where(chan_map==EEG)
        EEG = EEG[0][0]
    if EMGinput!=-1:
        EMG = np.where(chan_map==EMGinput)
        EMG = EMG[0][0]
    else:
        EMG = 0
    tic = timer.time()

    #time, dat = ntk.ntk_ecube.load_raw_binary(files[0], 64)
    try:
        average_EEG = list(np.load('Average_EEG_perhr.npy'))
    except FileNotFoundError:
        average_EEG = []
    try:
        var_EEG = list(np.load('Var_EEG_perhr.npy'))
    except FileNotFoundError:
        var_EEG = []

    for fil in filesindex:
        print('STARTING LOOP: '+str(fil))
        print('THIS IS THE STARTING USAGE: ' + str(psutil.virtual_memory()))
        start_label = files[fil][30:-4]
        end_label = files[fil+12][30:-4]
        # THIS CAN BE CONSOLIDATED INTO A FEW LINES LATER. THIS IS WHEN SAM IS LEARNING...
        # start importing your data

        # select the 12 files that you will work on at a time. this is to prevent
        # overloading RAM; since the default is to write 5 minute files, this will
        # effectively load an hour's worth of data at a time
        load_files = files[fil:fil+12]

        time     = []
        dat     = []
        dat2    = []
        full_selected_eeg     = np.zeros([np.size(selected_chans), 1])
        emg     = []


        print('Working on hour ' + str(int((fil+12)/12)))
        if cort == 'n':
            eeg = np.zeros([np.size(selected_chans),1])

        for a in np.arange(0,np.size(load_files)):
            if cort is not 1:
                print('merging file {}'.format(a))
                if EMGinput != -1:
                    dat = ntk.ntk_ecube.load_a_ch(load_files[a], numchan, EMG)
                    emg         = np.concatenate((emg,dat), axis=0)

                t,dgc = ntk.load_raw_gain_chmap_1probe(load_files[a],numchan, HS, nprobes, te = -1, probenum = probenum,probechans = 64)
                selected_eeg = dgc[selected_chans,:]

                full_selected_eeg = np.concatenate([full_selected_eeg, selected_eeg], axis = 1)

            else:
                print('Importing data from binary file...')
                time, dat     = ntk.ntk_ecube.load_raw_binary(load_files[a], 64)
                dat         = ntk.ntk_channelmap.channel_map_data(dat, 64, 'hs64')

                print('merging file {}'.format(a))
                eeg         = np.concatenate((eeg,dat[EEG]),axis=0)
                if EMGinput != -1:
                    emg         = np.concatenate((emg,dat[EMG]),axis=0)
        if cort is not 1:
            eeg = full_selected_eeg[:,1:]

        #    print('This is usage at step 3: ' + str(psutil.virtual_memory()))
        #print('This is usage at step 4: ' + str(psutil.virtual_memory()))
        #filters raw data
        print('Filtering and downsampling...')
        fs = 25000 # sample frequency
        nyq = 0.5*fs # nyquist
        N  = 3    # Filter order
        Wn = [0.5/nyq,400/nyq] # Cutoff frequencies
        B, A = signal.butter(N, Wn, btype='bandpass',output='ba')
        datf = signal.filtfilt(B,A, np.mean(eeg, axis = 0))    # key step
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
            #    print (i)
            EMGfor = (EMGfor - np.average(EMGfor))/np.std(EMGfor)
            #np.save('EMGfor' + str(int((fil+12)/12)) + '.npy',EMGfor)
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

        average_EEG.append(np.mean(downdatlfp))
        var_EEG.append(np.var(downdatlfp))

        np.save(LFP_dir + 'Average_EEG_perhr.npy', average_EEG)
        np.save(LFP_dir + 'Var_EEG_perhr.npy', var_EEG)

        np.save(LFP_dir + 'EEGhr' + str(int((fil+12)/12)),downdatlfp)

        print('Calculating bandpower...')
        #print('This is usage at step 5: ' + str(psutil.virtual_memory()))
        fsd = 200
        f, t_spec, x_spec = signal.spectrogram(downdatlfp, fs=fsd, window='hanning', nperseg=1000, noverlap=1000-1, mode='psd')
        # f, t_spec, x_spec = signal.spectrogram(downdatlfp, fs=fsd, window='hanning', nfft=400, detrend=False, noverlap=200, mode='psd')
        del(downdatlfp)
        del(datf)
        # x_spec[x_spec > 500] = 0
        fmax = 64
        fmin = 1
        x_mesh, y_mesh = np.meshgrid(t_spec, f[(f<fmax) & (f>fmin)])
        plt.figure(figsize=(16,2))
        p1 = plt.pcolormesh(x_mesh, y_mesh, np.log10(x_spec[(f<fmax) & (f>fmin)]), cmap='jet')
        #print('This is usage at step 5: ' + str(psutil.virtual_memory()))
        #plt.savefig('spect2.jpg')
        del(p1)
        del(x_mesh)
        del(y_mesh)
        fsemg = 4
        if EMGinput >= 0:
            realtime = np.arange(np.size(EMGamp))/fsemg
            plt.plot(realtime,(EMGamp - np.nanmean(EMGamp))/np.nanstd(EMGamp))
            np.save(LFP_dir + 'EMGhr' + str(int((fil+12)/12)),EMGamp)
        plt.ylim(1,64)
        plt.xlim(0,3600)
        plt.yscale('log')
        plt.title('Time Period: ' + start_label + '-' + end_label)
        plt.savefig(LFP_dir + 'specthr' + str(int((fil+12)/12)) + '.jpg')
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

        print('THIS IS THE USAGE AT THE END OF LOOP: ' + str(psutil.virtual_memory()))
        np.save(LFP_dir + 'delt' + str(int((fil+12)/12)) + '.npy',delt)
        np.save(LFP_dir + 'thet' + str(int((fil+12)/12)) + '.npy',thet)

        del(eeg)
        del(delt)
        del(thet)
        del(thetw)
        del(thetn)
        del(x_spec)
        del(f)
        del(t_spec)
        gc.collect()

    average_EEG = np.asarray(average_EEG)
    var_EEG = np.asarray(var_EEG)
    np.save(LFP_dir + 'Average_EEG_perhr.npy', average_EEG)
    np.save(LFP_dir + 'Var_EEG_perhr.npy', var_EEG)
