import os
from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import math
import glob

def split_clips(file):
	os.chdir(file)
	vids = glob.glob('*.mp4')
	os.mkdir("separated_videos")
	os.chdir(file+'/separated_videos')

	#os stats, find size, divide by 500e^6, divide the length of the video by 
	for vid in vids:
		size = os.stat(file+'/'+vid).st_size
		chunks = math.ceil(size/500e6)
		print("loading clip...")
		clip = VideoFileClip(file+'/'+vid)
		length = clip.duration
		num_secs = math.ceil(length/chunks)
		print("splitting clip...")
		for i in range(chunks):
			if i == (chunks-1):
				ffmpeg_extract_subclip(file+'/'+vid, i*num_secs, length, targetname='{}_separated_{}.mp4'.format(vid,i))
			else:
				ffmpeg_extract_subclip(file+'/'+vid, i*num_secs,(i+1)*num_secs, targetname='{}_separated_{}.mp4'.format(vid,i))
	
