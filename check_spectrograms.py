import glob
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json


def check_spectrograms(filename_sw):

    '''
    check spectrograms in LFP_chancheck/spec*.jpg
    LFP_dir defined in filename_sw json input file.
    Based on these spectrograms select channel for LFP extraction

    filename_sw : json input file, please check json_input_files directory
    '''

    with open(filename_sw, 'r') as f:
        d = json.load(f)

    LFP_dir = str(d['LFP_dir'])

    # add LFP_chancheck
    LFP_dir_spec = op.join(LFP_dir, 'LFP_chancheck' + op.sep)
    print("LFP_dir_spec ", LFP_dir_spec)

    # List all jpg files
    fl_list = np.sort(glob.glob(LFP_dir_spec + '*.jpg'))

    count = 0
    for i in range(len(fl_list)//4):
        fig1, ax1 = plt.subplots(nrows=4, ncols=1, figsize=[10, 10])
        for ii in range(4):
            img = mpimg.imread(fl_list[count])
            name_str = str(fl_list[count]).replace(str(LFP_dir_spec), '')\
                .replace('.jpg', '').replace('/', '').replace('spect', '')
            count = count + 1
            ax1[ii].imshow(img, aspect='auto')
            # ax1[ii].set_ylabel(str(count))
            ax1[ii].set_ylabel(str(name_str))
            ax1[ii].set_xlim(199, 1441)
            ax1[ii].set_ylim(178, 24)
            fig1.tight_layout()
    plt.show()

    print(" " * 14)
    print(" " * 14)
    print(" " * 14)
    print("========" * 14)
    print("========" * 14)
    print("Now save best channels")
    print("========" * 14)
    print("import Sleep_Wake_Scoring as sw")
    print("import numpy as np")
    print("best channels found from plots")
    print("for example 14, 21, 26, 32, 44")
    print("best_channels = np.array([10, 21, 26, 32, 44])")
    print("sw.manually_add_selected_channels({}, best_channels)"
          .format(LFP_dir_spec))
    print("========" * 14)
    print("========" * 14)
    print("========" * 14)
    print(" " * 14)
    print(" " * 14)
    print(" " * 14)
