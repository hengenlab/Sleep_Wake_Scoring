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
from sklearn.decomposition import PCA
import videotimestamp
import DLCMovement_input
import psutil
import math
import sys

def checkLFPchan(rawdat_dir, hstype, hour, start_chan = 0, fs = 25000, num_chans = 64):
    silicon_flag = 0
    os.chdir(rawdat_dir)
    files = sorted(glob.glob('*.bin'))
    if hstype == 'hs64':
        chan_map = np.array([26, 30, 6,  2,  18, 22, 14, 10, 12, 16, 8,  4,
                             28, 32, 24, 20, 48, 44, 36, 40, 64, 60, 52, 56,
                             54, 50, 42, 46, 62, 58, 34, 38, 39, 35, 59, 63,
                             47, 43, 51, 55, 53, 49, 57, 61, 37, 33, 41, 45,
                             17, 21, 29, 25, 1,  5,  13, 9,  11, 15, 23, 19,
                              3,  7,  31, 27]) - 1
    elif hstype == 'eibless64':
        chan_map = np.array([1,  5,  9,  13, 3,  7,  11, 15, 17, 21, 25, 29,
                             19, 23, 27, 31, 33, 37, 41, 45, 35, 39, 43, 47,
                             49, 53, 57, 61, 51, 55, 59, 63, 2,  6,  10, 14,
                             4,  8,  12, 16, 18, 22, 26, 30, 20, 24, 28, 32,
                             34, 38, 42, 46, 36, 40, 44, 48, 50, 54, 58, 62,
                             52, 56, 60, 64]) - 1
    elif hstype == 'PCB_tetrode':
        chan_map = np.array([2, 41, 50, 62, 6, 39, 42, 47, 34, 44, 51, 56, 
                            38, 48, 59, 64, 35, 53, 3, 37, 54, 57, 40, 43, 
                            45, 61, 46, 49, 36, 33, 52, 55, 15, 5, 58, 60, 
                            18, 9, 63, 1, 32, 14, 4, 7, 26, 20, 10, 13, 19, 
                            22, 16, 8, 28, 25, 12, 17, 23, 29, 27, 21, 11, 31, 30, 24]) - 1


    elif hstype == 'silicon_probe1':
        chan_map = np.arange(0, 64)
        silicon_flag = 1
    elif hstype == 'silicon_probe2':
        chan_map = np.arange(64, 129)
        silicon_flag = 1
    elif hstype == 'silicon_probe3':
        chan_map = np.arange(129, 193)
        silicon_flag = 1
    elif hstype == 'silicon_probe4':
        chan_map = np.arange(193, 257)
        silicon_flag = 1
    elif hstype == 'silicon_probe5':
        chan_map = np.arange(257, 321)
        silicon_flag = 1
    elif hstype == 'silicon_probe6':
        chan_map = np.arange(321, 385)
        silicon_flag = 1


    fil = hour*12
    load_files = files[fil:fil+12]

    time     = []
    dat     = []
    dat2    = []
    emg     = []
    
    print('Importing first data from binary file...')
    time, dat     = ntk.ntk_ecube.load_raw_binary(load_files[0], num_chans)
    if silicon_flag:
        eeg = dat[chan_map[0]:chan_map[-1]+1,:]
    else:
        eeg = ntk.ntk_channelmap.channel_map_data(dat, num_chans, hstype)

    for a in np.arange(1,np.size(load_files)):
        print('Importing next data from binary file...')
        time, dat     = ntk.ntk_ecube.load_raw_binary(load_files[a], num_chans)
        if silicon_flag:
            dat = dat[chan_map[0]:chan_map[-1]+1,:]
        else:
            dat = ntk.ntk_channelmap.channel_map_data(dat, num_chans, hstype)

        print('merging file {}'.format(a))
        eeg = np.concatenate([eeg, dat], axis = 1)
    try:
        os.chdir('LFP_chancheck')
    except FileNotFoundError:
        os.mkdir('LFP_chancheck')

    os.chdir(rawdat_dir)

    for chan in np.arange(start_chan, np.size(chan_map)):
        nyq = 0.5*fs # nyquist
        N  = 3    # Filter order
        Wn = [0.5/nyq,400/nyq] # Cutoff frequencies
        B, A = signal.butter(N, Wn, btype='bandpass',output='ba')
        datf = signal.filtfilt(B,A, eeg[chan])
        Wn = [10/nyq] # Cutoff frequencies
        B, A = signal.butter(N, Wn, btype='highpass',output='ba')

        reclen = 3600
        finalfs = 200
        R = fs/finalfs
        print(int(R*np.size(datf)))
        #downdatlfp = np.zeros(int(R*np.size(datf)))
        downsamp = np.zeros(fs*reclen)
        downsamp = datf[0:fs*reclen]
        del(datf)
        disp = fs*reclen - np.size(downsamp)
        if disp > 0:
            downsamp = np.pad(downsamp, (0,disp), 'constant')
        downsamp = downsamp.reshape(-1,int(R))
        downdatlfp = downsamp[:,1]
        dat = []

        fsd = 200
        f, t_spec, x_spec = signal.spectrogram(downdatlfp, fs=fsd, window='hanning', nperseg=1000, noverlap=1000-1, mode='psd')
        del (downdatlfp)
        fmax = 64
        fmin = 1
        x_mesh, y_mesh = np.meshgrid(t_spec, f[(f<fmax) & (f>fmin)])
        plt.figure(figsize=(16,2))
        p1 = plt.pcolormesh(x_mesh, y_mesh, np.log10(x_spec[(f<fmax) & (f>fmin)]), cmap='jet')

        plt.ylim(1,64)
        plt.xlim(0,3600)
        plt.yscale('log')
        plt.savefig(rawdat_dir + 'LFP_chancheck/spect_ch' + str(chan)+'.jpg')
        plt.close('all')
        #print('This is usage at step 5: ' + str(psutil.virtual_memory()))
        #plt.savefig('spect2.jpg')
        del(p1)
        del(x_mesh)
        del(y_mesh)
        del(f)
        del(t_spec)
        del(x_spec)
        #del(downdatlfp)


    

def plot_LFP(spect_dir):
    plt.ion()
    fig1,ax1 = plt.subplots(nrows = 8, ncols = 2, figsize = [10,10])

    for a,c in enumerate(np.arange(0,16)):
        row = np.concatenate([np.arange(0, 8), np.arange(0, 8)])
        col = np.concatenate([np.full(8, 0), np.full(8, 1)])

        img=mpimg.imread(spect_dir+'spect_ch'+ str(c) + '.jpg')
        imgplot = ax1[row[a],col[a]].imshow(img,aspect='auto')
        ax1[row[a],col[a]].set_ylabel(str(c+1))
        ax1[row[a],col[a]].set_xlim(199,1441)
        ax1[row[a],col[a]].set_ylim(178,24)
    fig1.tight_layout()

    fig2,ax2 = plt.subplots(nrows = 8, ncols = 2, figsize = [10,10])
    for a,c in enumerate(np.arange(16,32)):
        row = np.concatenate([np.arange(0, 8), np.arange(0, 8)])
        col = np.concatenate([np.full(8, 0), np.full(8, 1)])

        img=mpimg.imread(spect_dir+'spect_ch'+ str(c) + '.jpg')
        imgplot = ax2[row[a],col[a]].imshow(img,aspect='auto')

        ax2[row[a],col[a]].set_ylabel(str(c+1))
        ax2[row[a],col[a]].set_xlim(199,1441)
        ax2[row[a],col[a]].set_ylim(178,24)
    fig2.tight_layout()

    fig3,ax3 = plt.subplots(nrows = 8, ncols = 2, figsize = [10,10])
    for a,c in enumerate(np.arange(32,48)):
        row = np.concatenate([np.arange(0, 8), np.arange(0, 8)])
        col = np.concatenate([np.full(8, 0), np.full(8, 1)])

        img=mpimg.imread(spect_dir+'spect_ch'+ str(c) + '.jpg')
        imgplot = ax3[row[a],col[a]].imshow(img,aspect='auto')

        ax3[row[a],col[a]].set_ylabel(str(c+1))
        ax3[row[a],col[a]].set_xlim(199,1441)
        ax3[row[a],col[a]].set_ylim(178,24)
    fig3.tight_layout()

    fig4,ax4 = plt.subplots(nrows = 8, ncols = 2, figsize = [10,10])
    for a,c in enumerate(np.arange(48,64)):
        row = np.concatenate([np.arange(0, 8), np.arange(0, 8)])
        col = np.concatenate([np.full(8, 0), np.full(8, 1)])

        img=mpimg.imread(spect_dir+'spect_ch'+ str(c) + '.jpg')
        imgplot = ax4[row[a],col[a]].imshow(img,aspect='auto')

        ax4[row[a],col[a]].set_ylabel(str(c+1))
        ax4[row[a],col[a]].set_xlim(199,1441)
        ax4[row[a],col[a]].set_ylim(178,24)
    fig4.tight_layout()

    good_list = []
    enter = False

    while enter != 'n':
        enter = input('Enter desired channel number (enter "n" if done)')
        if enter!= 'n':
            n = int(enter)
            good_list.append(n)

    np.save(spect_dir + 'selected_channels.npy', good_list)
    return good_list

def confirm_channels(chans, raw_datdir, hstype, hour):

    spect_dir = raw_datdir+'LFP_chancheck/'
    os.chdir(raw_datdir)
    files = sorted(glob.glob('*.bin'))
    if hstype == 'hs64':
        chan_map = np.array([26, 30, 6,  2,  18, 22, 14, 10, 12, 16, 8,  4,
                             28, 32, 24, 20, 48, 44, 36, 40, 64, 60, 52, 56,
                             54, 50, 42, 46, 62, 58, 34, 38, 39, 35, 59, 63,
                             47, 43, 51, 55, 53, 49, 57, 61, 37, 33, 41, 45,
                             17, 21, 29, 25, 1,  5,  13, 9,  11, 15, 23, 19,
                              3,  7,  31, 27]) - 1
    elif hstype == 'eibless64':
        chan_map = np.array([1,  5,  9,  13, 3,  7,  11, 15, 17, 21, 25, 29,
                             19, 23, 27, 31, 33, 37, 41, 45, 35, 39, 43, 47,
                             49, 53, 57, 61, 51, 55, 59, 63, 2,  6,  10, 14,
                             4,  8,  12, 16, 18, 22, 26, 30, 20, 24, 28, 32,
                             34, 38, 42, 46, 36, 40, 44, 48, 50, 54, 58, 62,
                             52, 56, 60, 64]) - 1
    fil = hour*12
    load_files = files[fil:fil+12]

    time     = []
    dat     = []
    dat2    = []
    emg     = []
    
    print('Importing first data from binary file...')
    time, dat     = ntk.ntk_ecube.load_raw_binary(load_files[0], 64)
    eeg = ntk.ntk_channelmap.channel_map_data(dat, 64, hstype)

    for a in np.arange(1,np.size(load_files)):
        print('Importing next data from binary file...')
        time, dat     = ntk.ntk_ecube.load_raw_binary(load_files[a], 64)
        dat = ntk.ntk_channelmap.channel_map_data(dat, 64, hstype)
        print('merging file {}'.format(a))

        eeg = np.concatenate([eeg, dat], axis = 1)
    selected_eeg = np.zeros([np.size(chans), np.size(eeg[0])])
    for a,ch in enumerate(chans):    
        selected_eeg[a] = eeg[ch]

    #np.save('LFP_chancheck/selected_eeg.npy', selected_eeg)
    fig1, ax1 = plt.subplots(nrows = np.size(chans), ncols = 1, figsize = [12,12])
    fs = 25000

    for a,c in enumerate(chans):
        img=mpimg.imread(spect_dir+'spect_ch'+ str(c) + '.jpg')
        imgplot = ax1[a].imshow(img,aspect='auto')
        fig1.suptitle('LFP on Selected Channels')

        ax1[a].set_ylabel(str(c+1))
        ax1[a].set_xlim(199,1441)
        ax1[a].set_ylim(178,24)

    fig2, ax2 = plt.subplots(nrows = 1, ncols = 1, figsize = [16,2])
    nyq = 0.5*fs # nyquist
    N  = 3    # Filter order
    Wn = [0.5/nyq,400/nyq] # Cutoff frequencies
    B, A = signal.butter(N, Wn, btype='bandpass',output='ba')
    datf = signal.filtfilt(B,A, np.mean(selected_eeg, axis = 0))
    Wn = [10/nyq] # Cutoff frequencies
    B, A = signal.butter(N, Wn, btype='highpass',output='ba')

    reclen = 3600
    finalfs = 200
    R = fs/finalfs
    print(int(R*np.size(datf)))
    #downdatlfp = np.zeros(int(R*np.size(datf)))
    downsamp = np.zeros(fs*reclen)
    downsamp = datf[0:fs*reclen]
    del(datf)
    disp = fs*reclen - np.size(downsamp)
    if disp > 0:
        downsamp = np.pad(downsamp, (0,disp), 'constant')
    downsamp = downsamp.reshape(-1,int(R))
    downdatlfp = downsamp[:,1]
    dat = []

    fsd = 200
    f, t_spec, x_spec = signal.spectrogram(downdatlfp, fs=fsd, window='hanning', nperseg=1000, noverlap=1000-1, mode='psd')
    del (downdatlfp)
    fmax = 64
    fmin = 1
    x_mesh, y_mesh = np.meshgrid(t_spec, f[(f<fmax) & (f>fmin)])
    p1 = ax2.pcolormesh(x_mesh, y_mesh, np.log10(x_spec[(f<fmax) & (f>fmin)]), cmap='jet')
    #print('This is usage at step 5: ' + str(psutil.virtual_memory()))


    plt.ylim(1,64)
    plt.xlim(0,3600)
    plt.yscale('log')
    fig1.savefig('LFP_chancheck/selected_chans.jpg')
    fig2.savefig('LFP_chancheck/average_chans.jpg')












