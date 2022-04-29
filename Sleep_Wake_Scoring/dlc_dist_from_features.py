import neuraltoolkit as ntk
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
import glob


def dlc_dist_from_features(mfile, fps, hour_sec=3600,
                           cutoff_val=0.90,
                           lmedian=1,
                           lplot=0):
    '''
    mfile : motion file
    fps : video fps
    hour_sec : (default, 3600 hour in sec)
    cutoff_val : (default, 0.90)
    lmedian : (default, median across features)
    lplot : (default 0, no plots)

    returns :
        basename of mfile

    '''

    # get basename
    basename = op.splitext(op.split(mfile)[1])[0]

    # Get features
    dlc_positions, dlc_features = \
        ntk.dlc_get_position(mfile,
                             cutoff=cutoff_val)

    if lplot:
        # Plot features
        fig, ax = plt.subplots(num=1, figsize=(25, 25),
                               nrows=len(dlc_features),
                               sharex=True, sharey=True)
        for i in range(len(dlc_features)):
            ax[i].scatter(dlc_positions[:, 2*i], dlc_positions[:, 2*i+1],
                          c=np.arange(0, dlc_positions.shape[0], 1),
                          cmap=plt.cm.YlGn)
            ax[i].set_title(str(dlc_features[i]))

    # Calculate distance
    dlc_dist = None
    dlc_dist = []
    for i in range(len(dlc_features)):
        # math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        dist_feature = None
        dist_feature = \
            [np.sqrt((bx-ax)**2+(by-ay)**2) for (ax, bx, ay, by) in
             zip(dlc_positions[:, 2*i],
                 dlc_positions[1:, 2*i],
                 dlc_positions[:, 2*i+1],
                 dlc_positions[1:, 2*i+1])]
        # Add 0 to begining
        dist_feature.insert(0, 0)
        dlc_dist.append(dist_feature)
        dist_feature = None

    # Find best features
    for i in range(len(dlc_features)):
        print(dlc_features[i], " non Nan values ",
              np.count_nonzero(~np.isnan(dlc_dist[i])))

    if lplot:
        # Calculate median, then convert to seconds based on fps
        fig, ax = plt.subplots(num=2, figsize=(25, 25),
                               nrows=len(dlc_features) + 2,
                               sharex=False, sharey=False)
        for i in range(len(dlc_features)):
            ax[i].plot(dlc_dist[i])
            ax[i].set_title(str(dlc_features[i]))

    if lmedian:
        dlc_dist_median = np.nanmedian(np.asarray(dlc_dist), axis=0)
        full_move_filename = op.splitext(mfile)[0] + \
            '_full_movement_trace.npy'
        np.save(full_move_filename, dlc_dist_median)
        if lplot:
            ax[len(dlc_features)].plot(dlc_dist_median)
            ax[len(dlc_features)].set_title("Median dist")

        if (dlc_dist_median.shape[0] % fps) == 0:
            dlc_dist_median_persec = \
                dlc_dist_median.reshape(-1, fps)
        else:
            print("WARNING not multiple of fps ", fps, " got ",
                  dlc_dist_median.shape[0])
            dlc_dist_median_persec = \
                dlc_dist_median[0:int(hour_sec*fps)].reshape(hour_sec, fps)

        dlc_dist_median_persec_mean = np.nanmedian(dlc_dist_median_persec,
                                                   axis=1)
        if lplot:
            ax[len(dlc_features) + 1].plot(dlc_dist_median_persec_mean)
            ax[len(dlc_features) + 1].set_ylim([0, 10])
            ax[len(dlc_features) + 1].set_title("Median dist/conds")
    else:
        dlc_dist_features = np.asarray(dlc_dist)
        if lplot:
            ax[len(dlc_features)].plot(dlc_dist_features.T,
                                       label=dlc_features)
            ax[len(dlc_features)].set_title("dist features")
            ax[len(dlc_features)].legend()

        if (dlc_dist_features.shape[0] % fps) == 0:
            dlc_dist_features_persec = \
                dlc_dist_features.reshape(len(dlc_features), -1, fps)
        else:
            print("WARNING not multiple of fps ", fps, " got ",
                  dlc_dist_features.shape[0])
            dlc_dist_features_persec = \
                dlc_dist_features[:, 0:int(hour_sec*fps)].reshape(3,
                                                                  hour_sec,
                                                                  fps)
        dlc_dist_features_persec_mean = np.nanmedian(dlc_dist_features_persec,
                                                     axis=2)
        if lplot:
            ax[len(dlc_features) + 1].plot(dlc_dist_features_persec_mean.T,
                                           label=dlc_features)
            ax[len(dlc_features) + 1].set_title("Median dist features/seconds")
            ax[len(dlc_features) + 1].legend()

    if lplot:
        plt.show()

    return basename


if __name__ == "__main__":
    motion_dir = '/media/bs001r/James_AD_Project/KDR00014/10182021/'
    mfiles = glob.glob(motion_dir + op.sep + '*.h5')[0:1]
    for mfile in mfiles:
        fps = 15
        # hour_sec = 3600
        # cutoff_val = 0.90
        lplot = 1
        mfile = op.join(motion_dir, mfile)

        dlc_dist_from_features(mfile, fps,
                               hour_sec=3600,
                               cutoff_val=0.90,
                               lmedian=1,
                               lplot=lplot)
