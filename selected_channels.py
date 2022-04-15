import numpy as np
import json
import os


def manually_add_selected_channels(filename_sw, best_channels_for_lfp):

    '''
    Function to add selected_channels.npy manually for sleep scoring
    manually_add_selected_channels(filename_sw, best_channels_for_lfp)

    filename_sw : json input file, please check json_input_files directory
    best_channels_for_lfp : Best 5 channels for lfp

    Examples :
    import numpy as np
    best_channels = np.array([1, 2, 3, 4, 5])
    manually_add_selected_channels('/home/kbn/ABC00001.json', best_channels)

    '''

    # check shape of best_channels_for_lfp is 5
    if len(best_channels_for_lfp) != 5:
        raise ValueError('Please check best_channels_for_lfp, max length is 5')

    with open(filename_sw, 'r') as f:
        d = json.load(f)

    LFP_dir = str(d['LFP_dir'])
    if not os.path.isdir(LFP_dir):
        raise FileNotFoundError('Found No directory', LFP_dir)

    # Change to LFP_dir
    os.chdir(LFP_dir)

    # Check directory exists
    if(os.path.isdir(os.path.join(LFP_dir, 'LFP_chancheck'))):
        # Change dir to LFP_chancheck
        os.chdir('LFP_chancheck')
    else:
        # create folder LFP_chancheck
        os.mkdir('LFP_chancheck')
        print('Created LFP_chancheck directory')

        # Change dir to LFP_chancheck
        os.chdir('LFP_chancheck')

    # Exit if file exist already
    if os.path.exists(os.path.join('selected_channels.npy')):
        raise FileExistsError('File selected_channels.npy already exists')
    else:
        # save best_channels_for_lfp inside LFP_chancheck
        np.save('selected_channels.npy', best_channels_for_lfp)
        print('Saved LFP_chancheck/selected_channels.npy file')
