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
    lemg = int(d['emg'])
    EMGinput = str(d['EMGinput'])
    recblock_structure = str(d['recblock_structure'])
    accelerometer = str(d['accelerometer'])
    if accelerometer == "None":
        print("None accelerometer ", accelerometer)
        laccelerometer = 0
    else:
        laccelerometer = 1

    # check LFPdir
    if not os.path.exists(LFP_dir) and not os.path.isdir(LFP_dir):
        raise ValueError('LFP_dir not found')

    if lemg:
        # check EMGinput
        if not os.path.exists(EMGinput) and not os.path.isdir(EMGinput):
            raise ValueError('EMGinput not found')

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

    if lemg:
        emg_fl_list = \
            ntk.natural_sort(glob.glob(EMGinput + recblock_structure))
        for indx, emg_fl in enumerate(emg_fl_list):
            print(indx, " ", emg_fl)
        lorder = input('Is files in correct order?: y/n') == 'y'
        if lorder:
            print("creating emg with this file order")
        else:
            emg_fl_list = sorted(glob.glob(EMGinput + recblock_structure))
            for indx, emg_fl in enumerate(emg_fl_list):
                print(indx, " ", emg_fl)
            lorder = input('Is files in correct order now?: y/n') == 'y'
            if lorder:
                print("creating emg with this file order")
            else:
                raise ValueError('Files are not in order')

    if laccelerometer:
        accelerometer_fl_list = \
            ntk.natural_sort(glob.glob(accelerometer))
        for indx, accelerometer_fl in enumerate(accelerometer_fl_list):
            print(indx, " ", accelerometer_fl)
        lorder = input('Is files in correct order?: y/n') == 'y'
        if lorder:
            print("creating accelerometer with this file order")
        else:
            accelerometer_fl_list = sorted(glob.glob(accelerometer))
            for indx, accelerometer_fl in enumerate(accelerometer_fl_list):
                print(indx, " ", accelerometer_fl)
            lorder = input('Is files in correct order now?: y/n') == 'y'
            if lorder:
                print("creating accelerometer with this file order")
            else:
                raise ValueError('Files are not in order')

    # Load all lfp files and append # whatif it is too big
    # change to LFP dir
    os.chdir(LFP_dir)
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

    # Load all emg files and append # whatif it is too big
    if lemg:
        # change to EMG dir
        os.chdir(EMGinput)
        emg_all = None
        if len(emg_fl_list) > 1:
            for indx, emg_fl in enumerate(emg_fl_list[1:]):
                if indx == 0:
                    lfp_tmp1 = np.load(emg_fl_list[indx], allow_pickle=True)
                    lfp_tmp2 = np.load(emg_fl_list[indx+1], allow_pickle=True)
                    emg_all = np.column_stack((lfp_tmp1, lfp_tmp2))
                    del lfp_tmp1
                    del lfp_tmp2
                elif indx > 0:
                    lfp_tmp1 = np.load(emg_fl_list[indx+1], allow_pickle=True)
                    emg_all = np.column_stack((emg_all, lfp_tmp1))
                    del lfp_tmp1
                print("sh emg_all ", emg_all.shape, flush=True)
        else:
            emg_all = np.load(emg_fl_list[0], allow_pickle=True)
            print("sh emg_all ", emg_all.shape, flush=True)

    # Load all accelerometer files and append # whatif it is too big
    if laccelerometer:
        accelerometer_all = None
        if len(accelerometer_fl_list) > 1:
            for indx, accelerometer_fl in enumerate(accelerometer_fl_list[1:]):
                if indx == 0:
                    lfp_tmp1 = np.load(accelerometer_fl_list[indx],
                                       allow_pickle=True)
                    lfp_tmp2 = np.load(accelerometer_fl_list[indx+1],
                                       allow_pickle=True)
                    accelerometer_all = np.column_stack((lfp_tmp1, lfp_tmp2))
                    del lfp_tmp1
                    del lfp_tmp2
                elif indx > 0:
                    lfp_tmp1 = np.load(accelerometer_fl_list[indx+1],
                                       allow_pickle=True)
                    accelerometer_all = np.column_stack((accelerometer_all,
                                                         lfp_tmp1))
                    del lfp_tmp1
                print("sh accelerometer_all ", accelerometer_all.shape,
                      flush=True)
        else:
            accelerometer_all = np.load(accelerometer_fl_list[0],
                                        allow_pickle=True)
            print("sh accelerometer_all ", accelerometer_all.shape, flush=True)

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
        if lemg:
            emg = emg_all[:, start:end]

        if laccelerometer:
            accelerometer_h = accelerometer_all[:, start:end]

        if eeg.shape[1] < (lfp_freq * min_lfp_needed):
            break

        # mean across channels
        downdatlfp = np.mean(eeg, 0)
        np.save(op.join(base_dir_name, 'EEGhr' + str(hour+1)), downdatlfp)

        # save emg
        if lemg:
            np.save(op.join(base_dir_name, 'EMGhr' + str(hour+1)), emg)

        # save accelerometer
        if laccelerometer:
            np.save(op.join(base_dir_name, 'ACC' + str(hour+1)),
                    accelerometer_h)

        # calculate mean of mean of all channels!!!
        average_EEG.append(np.mean(downdatlfp))
        # calculate variance of mean of all channels!!!
        var_EEG.append(np.var(downdatlfp))

        # generate spectrogram, save delta and theta
        ntk.ntk_spectrogram(downdatlfp, fs=lfp_freq, nperseg=None,
                            noverlap=None, f_low=1, f_high=64,
                            lsavedir=base_dir_name, hour=hour+1, chan=None,
                            reclen=reclen, lsavedeltathetha=1, probenum=None,
                            lmultitaper=1)

    average_EEG = np.asarray(average_EEG)
    var_EEG = np.asarray(var_EEG)
    np.save(op.join(base_dir_name, 'Average_EEG_perhr.npy'), average_EEG)
    np.save(op.join(base_dir_name, 'Var_EEG_perhr.npy'), var_EEG)
