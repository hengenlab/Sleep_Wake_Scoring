# Sleep_Wake_Scoring
This is where all of the final and working versions of code involved with sleep-wake scoring will go  

#### Installation  
```
conda create -n Sleep_Wake_Scoring python=3  
conda activate Sleep_Wake_Scoring  
chmod 700 install.sh  
./install.sh  # install dependencies
```  
#### Usage

##### Create a json file with info
For example,  
```
{"rawdat_dir" : "/media/bs007r/XYF00003/XYF00003_2020-11-14_19-27-08/",
 "motion_dir" : "/media/bs007r/XYF00003/XYF0003_1114videos/",        # please save all DLC outputs to Hlabshare in the future
 "model_dir" : "/media/HlabShare/Sleep_Model/",
 "digi_dir" : "/media/bs007r/D1_rat/d2_2020-11-14_19-26-29/",
 "LFP_dir" : "/media/HlabShare/Sleep_Scoring/XYF03/1114/"      # please save all LFP and sleep-scoring output to Hlabshare in the future
 "animal": "XYF00003",
 "mod_name" : "rat_mouse",
 "epochlen" : 4,
 "fs" : 200,
 "emg" : 0,
 "pos": 1,
 "vid": 1,
 "num" : 0,           # time point you want to extract LFP  hour1 = 0 in python
 "num_labels": 4,     # number of DLC labels 
 "cort": 0,
 "EMGinput": -1,
 "numchan": 64,
 "HS": ["hs64"],
 "LFP_check" : 1,
 "probenum": 0,      # the probenum you want to extract LFP  probe1 = 0 in python  
 "nprobes" : 1,
 "video_fr":30,      # video 15Hz or 30Hz
 "digital":0,        # 1: Use digital files to align LFP and videos; 0: manually alignment
 "offset":7.4        # offset between LFP and video (always start the recording first then save the video), only works when digital == 0
}


Please check XYF03.json file.  
```

##### Find best channels  
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
##### Running LFP extract  
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
or
```diff
- Use method above using spikesorter  
- import Sleep_Wake_Scoring as sw 
- sw.extract_lfp('XYF03.json')
``` 

#### Get movement traces from DLC analysis

```
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

##### Running Sleep Wake Scoring Module  
*ipython*
```
import Sleep_Wake_Scoring as sw
sw.load_data_for_sw('XYF03.json')


this code is supressing warnings
What hour are you working on? (starts at 1): 500
loading delta and theta...
loading motion...
Were1 these videos seperated for DLC? (y/n)y
initializing motion...
loading video...
Were1 these videos seperated for DLC? (y/n)y
Generating EEG vectors...
Extracting delta bandpower...
Extracting theta bandpower...
Extracting alpha bandpower...
Extracting beta bandpower...
Extracting gamma bandpower...
Extracting narrow-band theta bandpower...
Boom. Boom. FIYA POWER...

     
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
        2. The random forest model's predicted states
        3. The binned motion for the hour
        
        TO CORRECT BINS:
        - click once on the middle figure to select the start of the bin you want to change
        - then click the last spot of the bin you want to change   
        - switch to terminal and type the state you want that bin to become
        
        VIDEO / RAW DATA:
        - if you hover over the motion figure you enter ~~ movie mode ~~  
        - click on that figure where you want to pull up movie and the raw trace for
            the 4 seconds before, during, and after the point that you clicked
        
        CURSOR:
        - because you have to click in certain figures it can be annoying to line up your mouse
            with where you want to inspect 
        - clicking figure (spectrogram) will created magenta dashed line across all 3 plots to check alignment.
        - clicking again removes old line and shows new line.
        
        EXITING SCORING:     
        - are you done correcting bins?
        - are you sure?
        - are you going to come to me/clayton/lizzie and ask how you 'go back' and 'change a few more bins'?
        - think for a second and then, when you're sure, press 'd'
        - it will then ask you if you want to save your states and/or update the random forest model
            - choose wisely 
        
        NOTES:
        - all keys pressed should be lowercase. don't 'shift + d'. just 'd'.
        - the video window along with the raw trace figure will remain up and update when you click a new bin
            don't worry about closing them or quitting them, it will probably error if you do.
        - slack me any errors if you get them or you have ideas for new functionality/GUI design
            - always looking to stay ~fresh~ with those ~graphics~
        - if something isn't working, make sure you're on Figure 2 and not the raw trace/terminal/video
        - plz don't toggle line while in motion axes, it messes up the axes limits, not sure why, working on it
        
        coming soon to sleep-wake code near you:
        - coding the state while you're slected in the figure, so you don't have to switch to terminal 
        - automatically highlighting problem areas where the model isn't sure or a red flag is raised (going wake/rem/wake/rem)
        - letting you choose the best fitting model before you fix the states to limit the amont of corrections
        
        
        ANOUNCEMENTS:
        - if you're trying to code each bin individually (a.k.a. when it asks you if you want to correct the model you say 'no')
            it doesn't save afterward yet. you will have to manually save it after you're done for the time being 
                                              
                                               

```
#### Frequently asked question

- ##### Running Sleep Wake Scoring Module too slow
Fix  
Please copy  
Model directory "/media/HlabShare/Sleep_Model/" to your local machine and  
change *.json file  
model_dir" : "model_directory_local_location",  

