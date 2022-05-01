import numpy as np
import os
import neuraltoolkit as ntk

# inputs_version1   ## need new json files
LFP_dir = \
    '/media/HlabShare/Sleep_Scoring/KDR00035/kdr35_02042022/0_9/probe1/co/'
filename = 'H_2022-02-04_08-12-28_2022-02-04_17-07-29_lfp_group0.npy'
# different start hour for consecutive blocks; for example: 0-12; 12-24; 24-36
start_hour = 0
# number of hours in each block
total_hour = 9
lfp_freq = 500
#


os.chdir(LFP_dir)
lfp_all = np.load(filename)
reclen = 3600

try:
    average_EEG = list(np.load('Average_EEG_perhr.npy'))
except FileNotFoundError:
    average_EEG = []
try:
    var_EEG = list(np.load('Var_EEG_perhr.npy'))
except FileNotFoundError:
    var_EEG = []

for idx, hour in enumerate(np.arange(start_hour, start_hour+total_hour, 1)):

    start = idx*3600*lfp_freq
    end = int((idx+1)*3600*lfp_freq)
    eeg = lfp_all[:, start:end]
    downdatlfp = np.mean(eeg, 0)
    np.save(LFP_dir + 'EEGhr' + str(hour), downdatlfp)

    average_EEG.append(np.mean(downdatlfp))
    var_EEG.append(np.var(downdatlfp))

    ntk.ntk_spectrogram(downdatlfp, fs=lfp_freq, nperseg=None,
                        noverlap=None, f_low=1, f_high=64,
                        lsavedir=LFP_dir, hour=hour, chan=None,
                        reclen=3600, lsavedeltathetha=1, probenum=None)

average_EEG = np.asarray(average_EEG)
var_EEG = np.asarray(var_EEG)
np.save(LFP_dir + 'Average_EEG_perhr.npy', average_EEG)
np.save(LFP_dir + 'Var_EEG_perhr.npy', var_EEG)
