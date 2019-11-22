import numpy as np
import os


def manually_add_selected_channels(rawdir_name, best_channels_for_lfp):

        '''
        Function to add selected_channels.npy manually for sleep scoring
        manually_add_selected_channels(rawdir_name, best_channels_for_lfp)

        rawdir_name : Directory where rawdata exists
        best_channels_for_lfp : Best 5 channels for lfp

        Examples :
        import numpy as np
        best_channels = np.array([1, 2, 3, 4, 5])
        manually_add_selected_channels('/home/kbn/', best_channels)

        '''


        # check shape of best_channels_for_lfp is 5
        if len(best_channels_for_lfp) != 5:
            raise ValueError('Please check best_channels_for_lfp')

        # Change to rawdata dir
        os.chdir(rawdir_name)

        # Check directory exists
        if(os.path.isdir(os.path.join(rawdir_name, 'LFP_chancheck'))):
            raise FileExistsError('Folder LFP_chancheck exists')

        # create folder LFP_chancheck
        os.mkdir('LFP_chancheck')

        # Change dir to LFP_chancheck
        os.chdir('LFP_chancheck')

        # Exit if file exist already
        if os.path.exists(os.path.join('selected_channels.npy')):
            raise FileExistsError('filename already exists')

        # save best_channels_for_lfp inside LFP_chancheck
        np.save('selected_channels.npy', best_channels_for_lfp)

