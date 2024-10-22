# Sleep_Wake_Scoring

---

We use a multivariate approach to scoring sleep in high resolution (4 second bins); we use machine learning to adaptively incorporate user input and predict the arousal state of subsequent data. This allows us to score months worth of continuous acquisition in hours.


---

### Installation
#### Create conda environment
```
conda create -n Sleep_Wake_Scoring_env python=3  
conda activate Sleep_Wake_Scoring_env
conda install pip
```
#### Dowload Sleep_Wake_Scoring
```
git clone https://github.com/hengenlab/Sleep_Wake_Scoring.git 
```

#### Using pip
```
cd locationofSleep_Wake_Scoring/Sleep_Wake_Scoring/
For example cd /home/kbn/git/Sleep_Wake_Scoring/
pip install .
```
---
#### Usage

##### Create a json file with info
For example,  
```
{"rawdat_dir" : "/media/bs007r/XYF00003/XYF00003_2020-11-14_19-27-08/",
 "motion_dir" : "/media/bs007r/XYF00003/XYF0003_1114videos/",        # please save all DLC outputs to Hlabshare in the future
 "model_dir" : "/media/HlabShare/Sleep_Model/",
 "digi_dir" : "/media/bs007r/D1_rat/d2_2020-11-14_19-26-29/",
 "LFP_dir" : "/media/HlabShare/Sleep_Scoring/XYF03/1114/"      # please save all LFP and sleep-scoring output to Hlabshare in the future
 "recblock_structure": "/XYF00003_02022022/*/probe1/*/*lfp_group0.npy",  # please check sub section below recblock
 "animal": "XYF00003",
 "mod_name" : "rat_mouse",
 "epochlen" : 4,
 "fs" : 500,
 "emg" : 0,
 "pos": 1,
 "vid": 1,
 "num" : 0,           # time point you want to extract LFP  hour1 = 0 in python
 "num_labels": 4,     # number of DLC labels 
 "cort": 0,
 "accelerometer": "None", # or "/media/HS/XYF00003/Accel/*/*/*/*lfp_group0.npy",
 "EMGinput": -1,
 "numchan": 64,
 "HS": ["hs64"],
 "LFP_check" : 1,
 "probenum": 0,      # the probenum you want to extract LFP  probe1 = 0 in python  
 "nprobes" : 1,
 "video_fr":30,      # video 15Hz or 30Hz
 "digital":0,        # 1-64: Use digital files to align LFP and videos; 0: manually alignment
 "offset":7.4        # offset between LFP and video (always start the recording first then save the video), only works when digital == 0
}


Please check XYF03.json file.  
```

###### recblock
```
As many users are using different methods/directory structure to save lfp,
recblock_structure guides Sleep_Wake_Scoring to find lfp files (*lfp_group0.npy)
for each recording block.

The list of paths to lfp_group0.npy files are created by
concatenating LFP_dir with recblock_structure.

Please make sure the following things.
1.  Make sure that all lfp is extracted from the entire restart/"recording block".
As Sleep_Wake_Scoring depends on hour labels. 
So missing lfp files will create inconsistent hour labels.

2. Do not mix different restart/"recording blocks" together.

3. While generating spectrograms, delta and theta 
import Sleep_Wake_Scoring as sw
sw.extract_delta_theta_from_lfp('/home/kbn/ABC00001.json')
Answer the question
"Is files in correct order?: y/n"
accurately.

```

##### 1. Find best channels  
```
# Create spectrograms
import neuraltoolkit as ntk
rawdat_dir='/media/KDR00032/KDR00032_L1_W2_2022-01-24_09-08-46/'
# Standard /media/HlabShare/Sleep_Scoring/ABC00001/LFP_chancheck/'
outdir='/media/HlabShare/Sleep_Scoring/ABC00001/LFP_chancheck/'
hstype = ['APT_PCB', 'APT_PCB']
# hour: hour to generate spectrograms
# choose a representative hour with both NREM, REM and wake
hour = 0
# fs: sampling frequency (default 25000)
# nprobes : Number of probes (default 1)
# number_of_channels : total number of channels
# probenum : which probe to return (starts from zero)
# probechans : number of channels per probe (symmetric)
# lfp_lowpass : default 250

ntk.selectlfpchans(rawdat_dir, outdir, hstype, hour,
                   fs=25000, nprobes=2, number_of_channels=128,
                   probenum=0, probechans=64, lfp_lowpass=250)
ntk.selectlfpchans(rawdat_dir, outdir, hstype, hour,
                   fs=25000, nprobes=2, number_of_channels=128,
                   probenum=1, probechans=64, lfp_lowpass=250)
# Now go through the spectrograms and select best lfp channels in
# the best probe to extract lfp

```
```
# Plot spectrograms
import Sleep_Wake_Scoring as sw
sw.check_spectrograms('json_input_files/KDR00014.json')
```

```
# Add channels manually
import Sleep_Wake_Scoring as sw 
sw.manually_add_selected_channels(filename_sw, best_channels_for_lfp)
filename_sw : json input file, please check json_input_files directory
best_channels_for_lfp : Best 5 channels for lfp
```

```diff
+ Eventhough you have extracted LFP from 5 channels using sorter,
+ please only add really good channels indexes in best_channels as
+ it affects sleep scoring.
! Remember best channels may change between recording sessions,
! as we do really long term recordings.
```

```
Examples :
import Sleep_Wake_Scoring as sw 
import numpy as np
best_channels = np.array([1, 2, 3, 4, 5])
sw.manually_add_selected_channels('/home/kbn/ABC00001.json', best_channels)
``` 
   
or
  
```diff
- Use method above using ntk.selectlfpchans  
- import Sleep_Wake_Scoring as sw
- hour = 5  # choose a representative hour with both NREM, REM and wake
- filename_sw = 'XYF03.json'  
- sw.selectLFPchan(filename_sw, hour)
```
##### 2. Running LFP extract  
```
Use sorting code to extract LFP. For more details check README.md in spikesorter code. 
Keep everything same as sorting input for spkint_wrapper_input.json and mountainsort.json
except in spkint_wrapper_input.json
change
1. "spk_sorter": "m", to "spk_sorter": "lfp",
   
2.  "lfp" : 0,  to "lfp": [70, 86, 100, 121],
where 70, 86, 100 and 121 are the channels to be extracted.   
It is best to select single channel from  different tetrode groups   
tetrode recordings.  Channel number starts at 1 not 0.
```
##### 3. Generate spectrograms, delta and theta
This will also generate accelerometer files (ACChr*.npy) files. 
```
import Sleep_Wake_Scoring as sw
sw.extract_delta_theta_from_lfp('/home/kbn/ABC00001.json')
```
or
```diff
- Use method above using spikesorter  
- import Sleep_Wake_Scoring as sw 
- sw.extract_lfp('XYF03.json')
``` 

#### 4. Get movement traces from DLC analysis

```
# Please use Mouse_genericmodel-Kiran-2022-09-08_aug2 model for dlc analysis
# in ris until there is a better model.
import Sleep_Wake_Scoring as sw
sw.extract_DLC('XYF03.json')  
```

#### Finding offset for json file input
```
# datadir: data directory where digital file is located
# ch : channel where Watchtower signal is recorded,
#      remember number starts from 0
# nfiles: First how many files to check for pulse change
#      (default first 10 files)
# fs: Sampling rate of digital file (default 25000)
# fps: Frames per second of video file (default 15)
# lnew: default 1, new digital files.
#      0 for old digital files with only one channel
# fig_indx: Default None, if index is given it plot figure
datadir = '/home/kbn/ABC12345/ABC_L9_W2_/'
ch = 1   #  _L9_W2_  zero indexing
nfiles = 10
fs = 25000
fps = 15
fig_indx = 1
video_start_index =\
    ntk.find_video_start_index(datadir, ch, nfiles=nfiles,
                                   fs=fs, fps=fps,
                                   lnew=1, fig_indx=fig_indx)
# Offset is in seconds so divide by fs
print("offset ", video_start_index/fs)  
```

#### Finding offset automatically

```
In json input:
"digital": 4,        # 1-64: Use digital files to align LFP and videos; 0: manually alignment
Here 4 is the channel watchtower data is recorded to.

Channel number starts from 1.

Recorded data folder name has this information. 
For example if ABC12345_L3_W2_2021-12-06_08-57-21
is the recorded data folder name. Then channel 2 is used for watchtower.

For old style digital files, channel number is always 1.
```

##### 5. Running Sleep Wake Scoring Module  

```
import Sleep_Wake_Scoring as sw
sw.load_data_for_sw('XYF03.json')

What hour are you working on? (starts at 1): 1
loading delta and theta...
loading motion...
initializing motion...
loading video...
Generating EEG vectors...
Extracting delta bandpower...
Extracting theta bandpower...
Extracting alpha bandpower...
Extracting beta bandpower...
Extracting gamma bandpower...
Extracting narrow-band theta bandpower...

     
                            .--,       .--,  
                           ( (  \.---./  ) ) 
                            '.__/o   o\__.'
                               {=  ^  =}
                                >  -  <
        ____________________.""`-------`"".________________________
         
                              INSTRUCTIONS
                      
                     Welcome to Sleep Wake Scoring!

        The figure you're looking at consists of 3 plots:
        1. The spectrogram for the hour you're scoring
        2. The random forest model's predicted states/already scored states
        3. The binned motion for the hour

        TO CORRECT BINS:
        - click once on the middle figure to select the start of the bin you
          want to change
        - then click the last spot of the bin you want to change
        - now color change to turquoise of the selected region
        - switch to terminal and type the state you want that bin to become
            Valid values are 1, 2, 3, 4 and 5
              1 – Active Wake, Green
              2 – NREM, Blue
              3 – REM, red
              4 - micro-arousal, yellow (not used often)
              5 – Quiet Wake, White
              else - cyan (unknown states)

        VIDEO / RAW DATA:
        - if you hover over the motion figure you enter ~~ movie mode ~~
        - click on that figure where you want to pull up movie and the raw
          trace for the 4 seconds before, during, and after the point that you
          clicked

        CURSOR:
        - clicking figure (spectrogram) will created magenta dashed line across
          all 3 plots to check alignment.
        - clicking again removes old line and shows new line.

        EXITING SCORING:
        - think for a second and then, when you're sure, press 'd'
        - Would you like to save these sleep states?:y/n
        - Would you like to update the model?: y/n n
            - choose wisely

        NOTES:
        - all keys pressed should be lowercase. don't 'shift + d'. just 'd'.
        - the video window along with the raw trace figure will remain up and
          update when you click a new bin don't worry about closing them or
          quitting them, it will probably error if you do.
        - if something isn't working, make sure you're on Figure 2 and not the
          raw trace/terminal/video
        - plz don't toggle line while in motion axes, it messes up the axes
          limits, not sure why, working on it                  

```
#### Frequently asked question

- ##### Running Sleep Wake Scoring Module too slow
Fix  
Please copy  
Model directory "/media/HlabShare/Sleep_Model/" to your local machine and  
change *.json file  
model_dir" : "model_directory_local_location",  



- #### Append missing empty row to dlc output h5 file to make it an hour. This can be used if video is stopped just short of 1hour and you have neuralrecording for whole hour
```
# h5_file_name : File name of h5 file with path
# video_fps : sampling rate used for video
# target_rows: The target number of rows to add. If None (default),
#         the number of rows is determined by video_fps. If specified,
#         this value takes precedence over video_fps,
#         and the number of rows will be calculated accordingly.

ntk.append_emptyframes_todlc_h5file(h5_file_name, video_fps, target_rows=None)
# create copy of h5_file_name as h5_file_name_back
#    make a new h5_file_name with frames for 1 hour

Example:
import neuraltoolkit as ntk
h5_file_name = '/home/kiran/ZBR00101-20240111T165554-175447DLC_resnet50_Mouse_genericmodelSep8shuffle1_1030000.h5'
video_fps = 15
ntk.append_emptyframes_todlc_h5file(h5_file_name, video_fps)
Added 989 rows to /home/kiran/ZBR00101-20240111T165554-175447DLC_resnet50_Mouse_genericmodelSep8shuffle1_1030000.h5.

Then rerun
import Sleep_Wake_Scoring as sw
# recreate dlc files, tmove.npy, videkey.npy and _full_movement_trace.npy
sw.extract_DLC('/home/kbn/ABC00001.json')

# have fun sleep scoring
sw.load_data_for_sw('/home/kbn/ABC00001.json')
```


