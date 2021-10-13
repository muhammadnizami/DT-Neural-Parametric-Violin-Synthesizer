import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tools/sms-tools/software/models/'))
import utilFunctions as UF
import guidedTwmHpsModel as HPS
from misc import *

def force_hfreq_nonzero(hfreq, hmag):
	hfreq = np.copy(hfreq)
	hmag = np.copy(hmag)
	for i in range(len(hfreq)-1):
		for j in range(len(hfreq[i])):
			if hfreq[i+1][j]==0 and hfreq[i][j] > 0:
				hfreq[i+1][j] = hfreq[i][j]
				hmag[i+1][j]=-120

	for i in range(1,len(hfreq)):
		for j in range(len(hfreq[-i])):
			if hfreq[-i-1][j] ==0 and hfreq[-i][j] >0:
				hfreq[-i-1][j] = hfreq[-i][j]
				hmag[-i-1][j]=-120

	return hfreq, hmag

def main(inputFile, midiFile, timingFile, outputFile, window='blackman', M=601, N=1024, t=-120,
	minSineDur=0.1, nH=20, minf0=170, maxf0=2000, f0et=5000, harmDevSlope=0.05, stocf=0.04):

	# size of fft used in synthesis
	Ns = 512
	# hop size (has to be 1/4 of Ns)
	H = 128

	# read input sound
	fs, x = UF.wavread(inputFile)
	print('samplerate: {}'.format(fs))

	# compute analysis window
	w = get_window(window, M)

	if midiFile[-4:]=='.mid':
		notes = read_midi(midiFile).instruments[0].notes
	else:
		notes = musicXML_to_notes_and_rests(midiFile)
	notechangetimes = read_timings(timingFile)

	# analyze the sound with the sinusoidal model
	hfreq, hmag, hphase, stocEnv = HPS.hpsModelAnal(x, fs, w, N, H, t, nH, notes, notechangetimes, minf0, maxf0, f0et, harmDevSlope, minSineDur, Ns, stocf)
	hfreq, hmag = force_hfreq_nonzero(hfreq,hmag)
	
	processed_data = list(zip(*(hfreq, hmag, stocEnv)))

	print('saving...')
	np.save(outputFile,processed_data)
	print('done')


	# create figure to plot
	plt.figure(figsize=(12, 9))

	# frequency range to plot
	maxplotfreq = 15000.0

	# plot the input sound
	plt.subplot(3,1,1)
	plt.plot(np.arange(x.size)/float(fs), x)
	plt.axis([0, x.size/float(fs), min(x), max(x)])
	plt.ylabel('amplitude')
	plt.xlabel('time (sec)')
	plt.title('input sound: x')

	# plot spectrogram stochastic component
	plt.subplot(3,1,2)
	numFrames = int(stocEnv[:,0].size)
	sizeEnv = int(stocEnv[0,:].size)
	frmTime = H*np.arange(numFrames)/float(fs)
	binFreq = (.5*fs)*np.arange(sizeEnv*maxplotfreq/(.5*fs))/sizeEnv
	plt.pcolormesh(frmTime, binFreq, np.transpose(stocEnv[:,:int(sizeEnv*maxplotfreq/(.5*fs)+1)]))
	plt.autoscale(tight=True)

	# plot harmonic on top of stochastic spectrogram
	if (hfreq.shape[1] > 0):
		harms = hfreq*np.less(hfreq,maxplotfreq)
		harms[harms==0] = np.nan
		numFrames = harms.shape[0]
		frmTime = H*np.arange(numFrames)/float(fs)
		plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
		plt.xlabel('time (sec)')
		plt.ylabel('frequency (Hz)')
		plt.autoscale(tight=True)
		plt.title('harmonics + stochastic spectrogram')

	plt.tight_layout()
	plt.ion()
	plt.show()