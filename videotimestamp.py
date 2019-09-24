#Requires string of digital input file name
#Returns first time stamp of video starting/stopping

import numpy as np
import neuraltoolkit as ntk
from matplotlib import pyplot as plt
import os

# #TEST DIGITAL OUTPUT 1
# os.chdir('/Volumes/rawdata/video_timestamp/2019-01-08_12-55-25')
# rawfile = 'Digital_64_Channels_int64_2019-01-08_12-55-25.bin'
# f = open(rawfile, 'rb')
# t = np.fromfile(f, dtype = np.uint64, count =  1)
# d = np.fromfile(f, dtype = np.int64,  count = -1)
# # Look at beginning of pulse and 1 min into pulse
# plt.plot(d[0:25000])
# plt.plot(d[(25000*60):(25000*61)])
# plt.show()

#TEST DIGITAL OUTPUT 2
# os.chdir('/Volumes/rawdata/video_timestamp/2019-01-08_15-08-55')
# rawfile = 'Digital_1_Channels_int64_2019-01-08_15-08-55.bin'
# t, d = ntk.load_digital_binary(rawfile)


def vidtimestamp(digitalfile):

	t,d = ntk.load_digital_binary(digitalfile)

	#Count contiguous consecutive elements of d
	count = 0
	counts = []
	for i in range(np.shape(d)[0]):
		if (d[i] == d[i-1]):
			count = count + 1
		else:
			counts.append(count)
			count = 0
	#convert counts list to array
	counts = np.array(counts)

	#get recording states
	recstates = np.zeros(counts.size)
	recind = np.where((counts==832)|(counts==833))[0]
	stopind = np.where((counts!=832)&(counts!=833))[0]
	recstates[recind] = 1
	recstates[stopind] = 0

	#get times that recording was started or stopped
	epoch = []
	rectimes = []
	for i in range(recstates.size+1):
		try:
			if (recstates[i] == recstates[i+1]):
				epoch.append(counts[i])
			else:
				epoch.append(counts[i])
				rectimes.append(np.sum(np.array(epoch)))
		except:
			pass

	# #verify that timestamp is correct
	# plt.ion()
	# plt.plot(d[(rectimes[0]-12500):(rectimes[0]+12500)])
	# plt.show()

	# print('The recording started at ' + str(rectimes[0]))

	timestamp = t[0] + (rectimes[0]/25000.0 * (1000*1000*1000))
	return(timestamp)
