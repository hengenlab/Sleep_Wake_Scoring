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
#import videotimestamp
import DLCMovement_input
import psutil
import math
import sys

def checkLFPchan(rawdat_dir, hstype, hour, start_chan = 0, fs = 25000, nprobes =1, num_chans = 64, probenum=0):
    
    print('probe number is {}'.format(probenum))
    os.chdir(rawdat_dir)
    files = sorted(glob.glob('H*.bin'))
    chan_map = ntk.find_channel_map(hstype[probenum],64) 

    fil = hour*12
    load_files = files[fil:fil+12]

    time     = []
    dat     = []
    dat2    = []
    emg     = []
    
    print('Importing first data from binary file...')
    time, eeg = ntk.ntk_ecube.load_raw_gain_chmap_1probe(load_files[0],num_chans,hstype,nprobes= nprobes,lraw=1,te = -1, probenum = probenum)

    for a in np.arange(1,np.size(load_files)):
        print('Importing next data from binary file...')
        time, dat     = ntk.ntk_ecube.load_raw_gain_chmap_1probe(load_files[a],num_chans,hstype,nprobes= nprobes,lraw=1,te = -1, probenum = probenum)
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
        # ntk.butter_bandpass(dat,highpass,lowpass,fs,3)
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
        # x_spec[x_spec > 500] = 0
        print('remove noise')
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
