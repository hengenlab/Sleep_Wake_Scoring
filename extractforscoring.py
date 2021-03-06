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
# from samb_work import videotimestamp
import videotimestamp
# from lizzie_work import DLCMovement_input
import DLCMovement_input
import psutil
import math
#import cv2
import sys
#import Sleep_Wake_Scoring as SWS
import LFP_utils as SWS
import json
from findPulse import findPulse

def check1(h5files):
# Checks to make sure that all of the h5 files are the same size
    sizes = [os.stat(i).st_size for i in h5files]
    if np.size(np.unique(sizes))>1:
        sys.exit('Not all of the h5 files are the same size')
def check2(files):
    str_idx = files[0].find('e3v') + 17
    timestamps = [files[i][str_idx:str_idx+9] for i in np.arange(np.size(files))]
    chk = []
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
    str_idx = h5files[0].find('e3v') + 17
    timestamps_h5 = [h5files[i][str_idx:str_idx+9] for i in np.arange(np.size(h5files))]
    timestamps_vid = [vidfiles[i][str_idx:str_idx+9] for i in np.arange(np.size(vidfiles))]
    if timestamps_h5 != timestamps_vid:
        sys.exit('h5 files and video files not aligned')
# Checks to make sure that all of the h5 files are continuous
# digi_dir = '/media/bs004r/D1/2019-03-29_10-24-20_d2_c2/'
# motion_dir = '/media/bs004r/EAB00040/labeled_videos/03_29/'


def extract_lfp(filename_sw):
    with open(filename_sw, 'r') as f:
           d = json.load(f)

    rawdat_dir = str(d['rawdat_dir'])
    motion_dir = str(d['motion_dir'])
    model_dir = str(d['model_dir'])
    digi_dir = str(d['digi_dir'])
    animal = str(d['animal'])
    mod_name = str(d['mod_name'])
    epochlen = int(d['epochlen'])
    fs = int(d['fs'])
    emg = int(d['emg'])
    pos = int(d['pos'])
    vid = int(d['vid'])
    move_flag = int(d['move_flag'])
    num = int(d['num'])
    num_labels = int(d['num_labels'])
    cort = int(d['cort'])
    EMGinput = int(d['EMGinput'])
    numchan = int(d['numchan'])
    HS = str(d['HS'])
    LFP_check = int(d['LFP_check'])



    # digi_dir = '/media/bs003r/Digital_Files/2019-06-05_12-56-31_d2_c5/'
    # rawdat_dir = '/media/bs004r/EAB00047/EAB00047_2019-06-05_11-10-44_p8_c4/'
    # motion_dir = '/media/bs004r/EAB00047/EAB00047_2019-06-05_11-10-44_p8_c4_video/'
    # digi_dir = '/media/rawdata/HellWeek/Digital/Cam_2018-10-19_18-21-31/'
    # rawdat_dir = '/media/rawdata/HellWeek/EAB00023/EAB23_2018-10-19_18-28-50_p10c5_grounded2/'
    # motion_dir = '/media/rawdata/HellWeek/EAB00023/labeled_videos/'

    print(digi_dir)
    print(motion_dir)
    print(rawdat_dir)
    # os.chdir(digi_dir)


    # move_flag = input('Have you already created aligned movement arrays (answer y if not using movement)? (y/n): ')
    # num = int(input('What hour are you starting on? (Starting at 0): '))

    os.chdir(rawdat_dir)
    files = sorted(glob.glob('*.bin'))


    os.chdir(digi_dir)
    digi_files = sorted(glob.glob('*.bin'))

    if move_flag == 0:
        stmp = findPulse(digi_dir,digi_files[0])
        h5 = sorted(glob.glob(motion_dir+'*.h5'))
        vidfiles = sorted(glob.glob(motion_dir+'*labeled.mp4'))
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
            string_idx = videofilename.find('e3v')
            which_vid.append(np.full((1,int(v.length)), videofilename[string_idx:])[0])
            leng.append(v.length)
            frame.append(np.arange(int(v.length)))
        leng = np.array(leng)
        which_vid = np.concatenate(which_vid)
        frame = np.concatenate(frame)
        #posi = np.cumsum(leng)
        #frame = np.arange(np.sum(leng))

        os.chdir(rawdat_dir)
        time, dat = ntk.load_raw_binary(files[0],64)
        offset = (stmp-time[0])
        alignedtime = (1000*1000*1000)*np.arange(np.sum(leng))/15 + offset
        # Time in column 2 is in nanoseconds and time in column 3 is in hours
        mot_vect = []
        basenames = []

        #labels = []
        #ckbn num_labels = int(input('How many labels did you use with DLC?'))
        # if num_labels > 1:
        #     for n in np.arange(num_labels):
        #         labels.append(input('What is label #' + str(n+1)))

        for i in np.arange(np.size(h5)):
            b = h5[i]
            basename = DLCMovement_input.get_movement(b, savedir = motion_dir, num_labels = num_labels, labels = False)
            vect = np.load(motion_dir+basename+'_full_movement_trace.npy')
            if np.size(vect)>leng[i]:
                #print('removing one nan')
                vect = vect[0:-1]
            #print(np.size(vect))
            mot_vect.append(vect)
            basenames.append(basename)
        mot_vect = np.concatenate(mot_vect)
        dt = alignedtime[1]-alignedtime[0]

        size_diff = np.size(mot_vect) - np.size(frame)

        if size_diff > 0:
            if all(np.isnan(mot_vect[-(size_diff):])):
                print('deleting extra nans from mot_vect')
                mot_vect= np.delete(mot_vect, np.arange((np.size(mot_vect)-size_diff), np.size(mot_vect)))

        if offset<0:
            n_phantom_frames = 0
        else:
            n_phantom_frames = int(math.floor((offset/dt)))

        phantom_frames = np.zeros(n_phantom_frames)
        phantom_frames[:] = np.nan
        novid_time = np.arange(dt, alignedtime[0], dt)

        corrected_motvect = np.concatenate([phantom_frames, mot_vect])
        corrected_frames  = np.concatenate([phantom_frames, frame])
        full_alignedtime = np.concatenate([novid_time, alignedtime])
        which_vid_full = np.concatenate([np.full((1, np.size(phantom_frames)), 'no video yet')[0], which_vid])


        aligner = np.column_stack((full_alignedtime,full_alignedtime/(1000*1000*1000*3600), corrected_motvect))
        #video_aligner = np.column_stack((corrected_frames, which_vid))
        neg_vals = []
        for gg in np.arange(np.shape(aligner)[0]):
            if aligner[gg,0] < 0:
                neg_vals.append(gg)

        aligner = np.delete(aligner, neg_vals, 0)
        which_vid_full = np.delete(which_vid_full, neg_vals, 0)
        corrected_frames = np.delete(corrected_frames, neg_vals, 0)

        reorganized_mot = []
        nhours = int(aligner[-1,1])
        bns = [i[i.find('-')+1:i.find('-')+19] for i in basenames]
        bns = np.unique(bns)
        for h in np.arange(num, nhours):
            tmp_idx = np.where((aligner[:,1]>(h)) & (aligner[:,1]<(h+1)))[0]
            time_move = (np.vstack((aligner[tmp_idx, 2], aligner[tmp_idx,1])))
            video_key = (np.vstack((aligner[tmp_idx, 0], which_vid_full[tmp_idx], corrected_frames[tmp_idx])))
            np.save(motion_dir+bns[h]+'_tmove.npy', time_move)
            np.save(motion_dir+bns[h]+'_vidkey.npy', video_key)

    os.chdir(rawdat_dir)
    filesindex = np.arange((num*12),np.size(files),12)
    if len(filesindex) == 0:
        raise ValueError('No files in this directory, check num')

    # cort = input('Cortical screw? (y/n): ')
    # EMGinput = int(input('Enter EMG channel (0 if using motion): ')) - 1
    # numchan = int(input('How many channels are on the headstage?'))
    # HS = input('Enter array type (hs64, eibless64, silicon_probex, PCB_tetrode): ')
    #reclen = int(input('Enter recording length in seconds: ')) #recording length in seconds
    reclen = 3600

    silicon_flag = 0
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
    elif HS == 'PCB_tetrode':
         chan_map = np.array([2, 41, 50, 62, 6, 39, 42, 47, 34, 44, 51, 56,
                              38, 48, 59, 64, 35, 53, 5, 37, 34, 57, 40, 43,
                              45, 61, 46, 49, 36, 33, 52, 55, 15, 5, 58, 60,
                              18, 9, 63, 1, 32, 14, 4, 7, 26, 20, 10, 13, 19,
                              22, 16, 8, 28, 25, 12, 17, 23, 29, 27, 21, 11, 31, 30, 24]) - 1



    elif HS == 'silicon_probe1':
        chan_map = np.arange(0, 64)
        silicon_flag = 1
    elif HS == 'silicon_probe2':
        chan_map = np.arange(64, 129)
        silicon_flag = 1
    elif HS == 'silicon_probe3':
        chan_map = np.arange(129, 193)
        silicon_flag = 1
    elif HS == 'silicon_probe4':
        chan_map = np.arange(193, 257)
        silicon_flag = 1
    elif HS == 'silicon_probe5':
        chan_map = np.arange(257, 321)
        silicon_flag = 1
    elif HS == 'silicon_probe6':
        chan_map = np.arange(321, 385)
        silicon_flag = 1

    if cort is not 1:
        try:
            selected_chans = np.load(rawdat_dir+'LFP_chancheck/selected_channels.npy')
            # LFP_check = input('Have you checked the average LFP that you are about to use? (y/n)')
            if LFP_check is not 1:
                hour = int(input('what hour did you use?'))
                SWS.confirm_channels(selected_chans, raw_datdir, HS, hour)
                print('Go find the individual and average spectrograms of your LFP in ' + rawdat_dir+'LFP_chancheck')
                sys.exit()
        except FileNotFoundError:
            LFP_check = input('You have not selected LFP channels, would you like to do that now? (y/n)')
            if LFP_check == 'y':
                hour = int(input('what hour will you use?'))
                #good_chans = [9, 14, 25, 32, 49, 57, 48]
                SWS.checkLFPchan(rawdat_dir, HS, hour)
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
                for n,c in enumerate(selected_chans):
                    print('Importing data from binary file...')
                    if silicon_flag:
                        idx = chan_map[c]
                        dat = ntk.ntk_ecube.load_a_ch(load_files[a], numchan, idx)
                    else:
                        idx = np.where(chan_map == c)[0][0]
                        dat     = ntk.ntk_ecube.load_a_ch(load_files[a], numchan, idx)
                    if n == 0:
                        temp = np.zeros(np.size(dat))
                        eeg = np.vstack([temp,dat])
                    else:
                        eeg = np.vstack([eeg, dat])
                selected_eeg = np.delete(eeg,0, axis = 0)

                # selected_eeg = np.zeros([np.size(selected_chans), np.size(dat[0])])
                # for a,ch in enumerate(selected_chans):
                #     selected_eeg[a] = dat[ch]
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
        datf = signal.filtfilt(B,A, np.mean(eeg, axis = 0))
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

        np.save('Average_EEG_perhr.npy', average_EEG)
        np.save('Var_EEG_perhr.npy', var_EEG)

        np.save('EEGhr' + str(int((fil+12)/12)),downdatlfp)

        print('Calculating bandpower...')
        #print('This is usage at step 5: ' + str(psutil.virtual_memory()))
        fsd = 200
        f, t_spec, x_spec = signal.spectrogram(downdatlfp, fs=fsd, window='hanning', nperseg=1000, noverlap=1000-1, mode='psd')
        del (downdatlfp)
        del(datf)
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
            np.save('EMGhr' + str(int((fil+12)/12)),EMGamp)
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

        print('THIS IS THE USAGE AT THE END OF LOOP: ' + str(psutil.virtual_memory()))
        np.save('delt' + str(int((fil+12)/12)) + '.npy',delt)
        np.save('thet' + str(int((fil+12)/12)) + '.npy',thet)


        del(delt)
        del(thet)
        del(thetw)
        del(thetn)
        del(x_spec)
        del(f)
        del(t_spec)

    average_EEG = np.asarray(average_EEG)
    var_EEG = np.asarray(var_EEG)
    np.save('Average_EEG_perhr.npy', average_EEG)
    np.save('Var_EEG_perhr.npy', var_EEG)
