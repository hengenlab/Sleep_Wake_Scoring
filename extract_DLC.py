import os
import numpy as np
import neuraltoolkit as ntk
import glob
# import DLCMovement_input
from .dlc_dist_from_features import dlc_dist_from_features
import math
import json
# from findPulse import findPulse
# from SW_utils import check_h5_file_size
from SW_utils import check_time_stamps
from SW_utils import check_time_period


def extract_DLC(filename_sw):

    with open(filename_sw, 'r') as f:
        d = json.load(f)

    rawdat_dir = str(d['rawdat_dir'])
    motion_dir = str(d['motion_dir'])
    # model_dir = str(d['model_dir'])
    digi_dir = str(d['digi_dir'])
    # animal = str(d['animal'])
    # mod_name = str(d['mod_name'])
    # epochlen = int(d['epochlen'])
    # fs = int(d['fs'])
    # emg = int(d['emg'])
    # pos = int(d['pos'])
    # vid = int(d['vid'])
    num = int(d['num'])
    # num_labels = int(d['num_labels'])
    # cort = int(d['cort'])
    # EMGinput = int(d['EMGinput'])
    # numchan = int(d['numchan'])
    # HS = d['HS']
    # LFP_check = int(d['LFP_check'])
    # probenum = int(d['probenum'])
    # nprobes = int(d['nprobes'])
    fps = int(d['video_fr'])
    digital = int(d['digital'])
    align_offset = d['offset']

    print(digi_dir)
    print(motion_dir)
    print(rawdat_dir)

    h5 = sorted(glob.glob(motion_dir+'*.h5'))
    vidfiles = sorted(glob.glob(motion_dir+'*labeled.mp4'))

    # check h5 files have same size
    # check_h5_file_size(h5)
    check_time_stamps(h5)
    check_time_stamps(vidfiles)
    check_time_period(h5, vidfiles)

    leng = []
    which_vid = []
    frame = []
    for a in np.arange(np.size(vidfiles)):
        videofilename = vidfiles[a]
        lstream = 0
        # get video attributes
        v = ntk.NTKVideos(videofilename, lstream)
        # string_idx = videofilename.find('e3v')
        which_vid.append(np.full((1, int(v.length)),
                         videofilename.split('/')[-1])[0])
        print(v.length)

        leng.append(v.length)
        frame.append(np.arange(int(v.length)))
    leng = np.array(leng)
    which_vid = np.concatenate(which_vid)
    frame = np.concatenate(frame)
    print(leng)   # video length

    # if digital channel number is given
    # Load first digital file then find timestamp
    # add to it v_start_index in nanoseconds = stmp
    # Substract stmp from headstage timestamp to find offset
    if digital:
        os.chdir(digi_dir)
        digi_files = sorted(glob.glob('D*.bin'))

        os.chdir(rawdat_dir)
        files = sorted(glob.glob('H*.bin'))

        # stmp = findPulse(digi_dir, digi_files[0])
        try:
            # get first files timestamp
            stmp_digital = \
                ntk.load_digital_binary_allchannels(digi_files[0],
                                                    t_only=1,
                                                    channel=digital-1)
            # get video start index
            v_start_index =\
                ntk.find_video_start_index(digi_dir, digital-1, nfiles=10,
                                           fs=25000, fps=fps,
                                           lnew=1, fig_indx=None)
        except Exception as e:
            print("Got error", e, " trying old digital files")
            os.chdir(digi_dir)
            # get first files timestamp
            stmp_digital = ntk.load_digital_binary(digi_files[0],
                                                   t_only=1,
                                                   lcheckdigi64=1)
            # get video start index
            v_start_index =\
                ntk.find_video_start_index(digi_dir, digital-1, nfiles=10,
                                           fs=25000, fps=fps,
                                           lnew=0, fig_indx=None)
        # Add to timestamp , v_start_index in nanoseconds
        print("digital timestamp ", stmp_digital)
        print("video start is  ", v_start_index, " samples")
        print("video start is  ", v_start_index/25000, " seconds")
        stmp = stmp_digital[0] + ((v_start_index/25000) * 1e9)
        print("stmp is:")
        print(stmp)

        os.chdir(rawdat_dir)

        # please note as we are using t_only=1,
        # number of channels (64), hstype (['hs64'])
        # and nprobes (1) does not matter as only timestamp
        # is read and returned.
        time_d = \
            ntk.load_raw_binary_gain_chmap(files[0],
                                           64,
                                           ['hs64'],
                                           nprobes=1,
                                           t_only=1)
        print(time_d[0])
        offset = stmp - time_d[0]
        print("offset is  ", offset, " nanoseconds")
        print("offset is  ", offset/1e9, " seconds")

    else:
        print("manually adjust the offset between video and neural recording")
        offset = align_offset * 1e9

    print(offset)
    alignedtime = (1000*1000*1000)*np.arange(np.int(np.sum(leng)))/fps + offset
    print(alignedtime)

    mot_vect = []
    basenames = []

    for i in np.arange(np.size(h5)):
        b = h5[i]
        # basename = \
        #     DLCMovement_input.get_movement(b, savedir=motion_dir,
        #                                    num_labels=num_labels,
        #                                    labels=False)
        basename = \
            dlc_dist_from_features(b, fps,
                                   hour_sec=3600,
                                   cutoff_val=0.90,
                                   lmedian=1,
                                   lplot=0)
        vect = np.load(motion_dir+basename+'_full_movement_trace.npy')
        if np.size(vect) > leng[i]:
            print('removing one nan')
            vect = vect[0:-1]
        # print(np.size(vect))
        print(vect.shape)
        mot_vect.append(vect)
        basenames.append(basename)

    mot_vect = np.concatenate(mot_vect)

    print("movie time:")
    print(np.sum(leng))
    print("vect time")
    print(mot_vect.shape)

    dt = alignedtime[1]-alignedtime[0]
    #        print(dt)

    size_diff = np.size(mot_vect) - np.size(frame)

    if size_diff > 0:
        print("size_difference")
        if all(np.isnan(mot_vect[-(size_diff):])):
            print('deleting extra nans from mot_vect')
            mot_vect = \
                np.delete(mot_vect,
                          np.arange((np.size(mot_vect)-size_diff),
                                    np.size(mot_vect)))

    if offset < 0:
        n_phantom_frames = 0
    else:
        print("offset > 0")
        n_phantom_frames = int(math.floor((offset/dt)))
        print(n_phantom_frames)

    phantom_frames = np.zeros(n_phantom_frames)
    phantom_frames[:] = np.nan
    novid_time = np.arange(dt, alignedtime[0], dt)

    print(novid_time.shape)
    print(mot_vect.shape)
    print(alignedtime.shape)

    corrected_motvect = np.concatenate([phantom_frames, mot_vect])
    corrected_frames = np.concatenate([phantom_frames, frame])
    full_alignedtime = np.concatenate([novid_time, alignedtime])
    which_vid_full = \
        np.concatenate([np.full((1, np.size(phantom_frames)),
                       'no video yet')[0], which_vid])

    print(full_alignedtime.shape)
    print(corrected_motvect.shape)

    aligner = np.column_stack((full_alignedtime,
                               full_alignedtime/(1000*1000*1000*3600),
                               corrected_motvect))
    # video_aligner = np.column_stack((corrected_frames, which_vid))
    neg_vals = []
    for gg in np.arange(np.shape(aligner)[0]):
        if aligner[gg, 0] < 0:
            neg_vals.append(gg)

    aligner = np.delete(aligner, neg_vals, 0)
    which_vid_full = np.delete(which_vid_full, neg_vals, 0)
    corrected_frames = np.delete(corrected_frames, neg_vals, 0)

    # reorganized_mot = []
    nhours = int(aligner[-1, 1])
    bns = [i[i.find('-')+1:i.find('-')+19] for i in basenames]
    bns = np.unique(bns)
    for h in np.arange(num, nhours):
        tmp_idx = np.where((aligner[:, 1] > (h)) & (aligner[:, 1] <= (h+1)))[0]
        time_move = (np.vstack((aligner[tmp_idx, 2], aligner[tmp_idx, 1])))
        video_key = (np.vstack((aligner[tmp_idx, 0], which_vid_full[tmp_idx],
                     corrected_frames[tmp_idx])))
        np.save(motion_dir+bns[h]+'_tmove.npy', time_move)
        np.save(motion_dir+bns[h]+'_vidkey.npy', video_key)
