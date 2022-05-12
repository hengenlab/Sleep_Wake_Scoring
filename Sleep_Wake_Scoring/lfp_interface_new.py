import numpy as np
import os
import neuraltoolkit as ntk
import json
import glob
import os.path as op


def extract_delta_theta_from_lfp(filename_sw):
    '''
    Generate spectrogram and save delta, theta

    extract_delta_theta_from_lfp(filename_sw)
    filename_sw : json input file, please check json_input_files directory
    '''

    # contants
    reclen = 3600
    lfp_freq = 500
    # 8 * lfp_freq
    min_lfp_needed = 8

    with open(filename_sw, 'r') as f:
        d = json.load(f)

    LFP_dir = str(d['LFP_dir'])
    recblock_structure = str(d['recblock_structure'])
    print("recblock_structure ", recblock_structure)
    fl_list = ntk.natural_sort(glob.glob(LFP_dir + recblock_structure))
    for indx, fl in enumerate(fl_list):
        print(indx, " ", fl)
    lorder = input('Is files in correct order?: y/n') == 'y'
    if lorder:
        print("creating spectrograms with this file order")
    else:
        fl_list = sorted(glob.glob(LFP_dir + recblock_structure))
        for indx, fl in enumerate(fl_list):
            print(indx, " ", fl)
        lorder = input('Is files in correct order now?: y/n') == 'y'
        if lorder:
            print("creating spectrograms with this file order")
        else:
            raise ValueError('Files are not in order')

    # check LFPdir
    if not os.path.exists(LFP_dir) and not os.path.isdir(LFP_dir):
        raise ValueError('LFP_dir not found')

    # Load all lfp files and append # whatif it is too big
    lfp_all = None
    if len(fl_list) > 1:
        for indx, fl in enumerate(fl_list[1:]):
            if indx == 0:
                lfp_tmp1 = np.load(fl_list[indx], allow_pickle=True)
                lfp_tmp2 = np.load(fl_list[indx+1], allow_pickle=True)
                lfp_all = np.column_stack((lfp_tmp1, lfp_tmp2))
                del lfp_tmp1
                del lfp_tmp2
            elif indx > 0:
                lfp_tmp1 = np.load(fl_list[indx+1], allow_pickle=True)
                lfp_all = np.column_stack((lfp_all, lfp_tmp1))
                del lfp_tmp1
            print("sh lfp_all ", lfp_all.shape, flush=True)
    else:
        lfp_all = np.load(fl_list[0], allow_pickle=True)
        print("sh lfp_all ", lfp_all.shape, flush=True)

    total_hours = int(np.ceil(lfp_all.shape[1]/(lfp_freq * reclen)))

    # change to LFP dir
    os.chdir(LFP_dir)

    #  create a base name based on recblock_structure
    base_dir_name = \
        recblock_structure.replace(op.sep, '_').split('*')[0].replace('_', '')
    base_dir_name = op.join(LFP_dir, base_dir_name)
    if os.path.exists(base_dir_name) and os.path.isdir(base_dir_name):
        os.chdir(base_dir_name)
    else:
        os.mkdir(base_dir_name)

    try:
        average_EEG = list(np.load('Average_EEG_perhr.npy'))
    except FileNotFoundError:
        average_EEG = []
    try:
        var_EEG = list(np.load('Var_EEG_perhr.npy'))
    except FileNotFoundError:
        var_EEG = []

    for indx, hour in enumerate(range(total_hours)):
        print(indx, " ", hour)
        start = indx*reclen*lfp_freq
        end = int((indx+1)*reclen*lfp_freq)
        eeg = lfp_all[:, start:end]
        print("sh eeg ", eeg.shape, flush=True)
        if eeg.shape[1] < (lfp_freq * min_lfp_needed):
            break

        # mean across channels
        downdatlfp = np.mean(eeg, 0)
        np.save(op.join(base_dir_name, 'EEGhr' + str(hour)), downdatlfp)

        # calculate mean of mean of all channels!!!
        average_EEG.append(np.mean(downdatlfp))
        # calculate variance of mean of all channels!!!
        var_EEG.append(np.var(downdatlfp))

        # generate spectrogram, save delta and theta
        ntk.ntk_spectrogram(downdatlfp, fs=lfp_freq, nperseg=None,
                            noverlap=None, f_low=1, f_high=64,
                            lsavedir=base_dir_name, hour=hour+1, chan=None,
                            reclen=reclen, lsavedeltathetha=1, probenum=None)

    average_EEG = np.asarray(average_EEG)
    var_EEG = np.asarray(var_EEG)
    np.save(op.join(base_dir_name, 'Average_EEG_perhr.npy'), average_EEG)
    np.save(op.join(base_dir_name, 'Var_EEG_perhr.npy'), var_EEG)
