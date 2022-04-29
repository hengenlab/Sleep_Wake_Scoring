import os
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import neuraltoolkit as ntk
import glob
import gc
import json
from sys import platform
import warnings


def selectLFPchan(filename_sw, hour):

    '''
    selectLFPchan is used to select best channels
    Creates spectrogram file
    LFP_chancheck/spect_ch*.jpg
    inside LFP_dir defined in filename_sw json input file.
    Based on these spectrograms select channel for LFP extraction


    filename_sw : json input file, please check json_input_files directory
    hour : 5, for example hour 5.
           Each hour is 12 files as recordings are 5min each.
           Choose a representative hour with both NREM, REM and wake.
    '''

    warnings.filterwarnings('ignore', '.*do not.*', )
    warnings.warn("selectLFPchan: Please use ntk.selectlfpchans",
                  FutureWarning)

    if platform == "darwin":
        plt.switch_backend('TkAgg')
    else:
        plt.switch_backend('Agg')

    with open(filename_sw, 'r') as f:
        d = json.load(f)

    rawdat_dir = str(d['rawdat_dir'])
    LFP_dir = str(d['LFP_dir'])
    fs = 25000
    start_chan = 0
    # EMGinput = int(d['EMGinput'])
    num_chans = int(d['numchan'])
    hstype = d['HS']
    probenum = int(d['probenum'])
    nprobes = int(d['nprobes'])

    if(rawdat_dir[-4:] == '.txt'):
        file1 = open(rawdat_dir, 'r')
        lines = file1.readlines()
        files = [line.strip('\n') for line in lines]
    else:
        os.chdir(rawdat_dir)
        files = sorted(glob.glob('H*.bin'))

    chan_map = ntk.find_channel_map(hstype[probenum], 64)

    fil = hour*12
    load_files = files[fil:fil+12]

    time = []
    dat = []
    # dat2 = []
    # emg = []

    print('Importing first data from binary file...')
    time, eeg = \
        ntk.load_raw_gain_chmap_1probe(load_files[0],
                                       num_chans,
                                       hstype,
                                       nprobes=nprobes,
                                       lraw=1,
                                       te=-1,
                                       probenum=probenum)

    for a in np.arange(1, np.size(load_files)):
        print('Importing next data from binary file...')
        time, dat = ntk.load_raw_gain_chmap_1probe(load_files[a],
                                                   num_chans,
                                                   hstype,
                                                   nprobes=nprobes,
                                                   lraw=1,
                                                   te=-1,
                                                   probenum=probenum)
        print('merging file {}'.format(a))
        eeg = np.concatenate([eeg, dat], axis=1)

    os.chdir(LFP_dir)
    try:
        os.chdir('LFP_chancheck')
    except FileNotFoundError:
        os.mkdir('LFP_chancheck')

    for chan in np.arange(start_chan, np.size(chan_map)):
        nyq = 0.5*fs  # nyquist
        N = 3    # Filter order
        Wn = [0.5/nyq, 400/nyq]  # Cutoff frequencies
        B, A = signal.butter(N, Wn, btype='bandpass', output='ba')
        datf = signal.filtfilt(B, A, eeg[chan])
        Wn = [10/nyq]  # Cutoff frequencies
        B, A = signal.butter(N, Wn, btype='highpass', output='ba')
        # ntk.butter_bandpass(dat,highpass,lowpass,fs,3)
        reclen = 3600
        finalfs = 200
        R = fs/finalfs
        print(int(R*np.size(datf)))
        # downdatlfp = np.zeros(int(R*np.size(datf)))
        downsamp = np.zeros(fs*reclen)
        downsamp = datf[0:fs*reclen]
        del(datf)
        disp = fs*reclen - np.size(downsamp)
        if disp > 0:
            downsamp = np.pad(downsamp, (0, disp), 'constant')
        downsamp = downsamp.reshape(-1, int(R))
        downdatlfp = downsamp[:, 1]
        dat = []

        fsd = 200
        f, t_spec, x_spec = \
            signal.spectrogram(downdatlfp, fs=fsd, window='hanning',
                               nperseg=1000, noverlap=1000-1, mode='psd')
        # f, t_spec, x_spec = signal.spectrogram(downdatlfp, fs=fsd,
        # window='hanning', nfft=400, detrend=False, noverlap=1000-1,
        # mode='psd')
        # f, t_spec, x_spec = signal.spectrogram(downdatlfp, fs=fsd,
        # window='hanning', nfft=400, detrend=False, noverlap=200, mode='psd')
        del (downdatlfp)
        # x_spec[x_spec > 500] = 0
        print('remove noise')
        fmax = 64
        fmin = 1
        x_mesh, y_mesh = np.meshgrid(t_spec, f[(f < fmax) & (f > fmin)])
        plt.figure(figsize=(16, 2))
        p1 = plt.pcolormesh(x_mesh, y_mesh,
                            np.log10(x_spec[(f < fmax) & (f > fmin)]),
                            cmap='jet')

        plt.ylim(1, 64)
        plt.xlim(0, 3600)
        plt.yscale('log')
        plt.savefig(LFP_dir + 'LFP_chancheck/spect_ch' + str(chan)+'.jpg')
        plt.close('all')
        # print('This is usage at step 5: ' + str(psutil.virtual_memory()))
        # plt.savefig('spect2.jpg')
        del(p1)
        del(x_mesh)
        del(y_mesh)
        del(f)
        del(t_spec)
        del(x_spec)
        gc.collect()
