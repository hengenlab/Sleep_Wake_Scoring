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
