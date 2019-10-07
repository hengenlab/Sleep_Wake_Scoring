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
{"rawdat_dir" : "/media/bs004r/KNR00004/KNR00004_2019-08-01_16-43-45_p1_c3/",
 "motion_dir" : "/media/bs004r/KNR00004/KNR00004_2019-08-01_16-43-45_p1_c3_labeled_video/",
 "model_dir" : "/media/HlabShare/Sleep_Model/",
 "digi_dir" : "/media/bs004r/D1/2019-08-01_16-43-11_d2_c2/",
 "animal": "KNR00004",
 "mod_name" : "rat_mouse",
 "epochlen" : 4,
 "fs" : 200,
 "emg" : 0 ,
 "pos": 1,
 "vid": 1,
 "move_flag": 0,
 "num" : 200,
 "num_labels": 5,
 "cort": 0,
 "EMGinput": 0,
 "numchan": 64,
 "HS": "eibless64",
 "LFP_check" : 1
}

Please check KNR00004.json file.  
```

##### Find best channels  
*ipython*
```
 import Sleep_Wake_Scoring as sw 
 rawdat_dir = '/media/bs004r/EAB00040/EAB00040_2019-03-29_10-28-27_p9_c5/'                                                                                                                                                                  
 hstype = 'silicon_probe2'                            
 hour = 5                                             
 num_chans = 256                                      
 sw.checkLFPchan(rawdat_dir, hstype, hour, num_chans= num_chans, start_chan=0)
``` 



##### Running LFP extract  
*ipython*
```
import Sleep_Wake_Scoring as sw 
sw.extract_lfp('KNR00004.json')
``` 

##### Running Sleep Wake Scoring Module  
*ipython*
```
import Sleep_Wake_Scoring as sw
sw.load_data_for_sw('/home/kbn/KNR00004.json')


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
Use a random forest? y/n: y
Satisfied?: y/n n
Do you want to fix the models states?: y/ny

     
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
        - while selected in the scoring figure (called Figure 2) press 'l' (as in Lizzie) to toggle a black line across each plot
        - this line will stay there until you press 'l' again, then it will erase and move
        - adjust until you like your location, then click to select a bin or watch a movie
        
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
