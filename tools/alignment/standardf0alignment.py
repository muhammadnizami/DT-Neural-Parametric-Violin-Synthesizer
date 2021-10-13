import numpy as np
import pretty_midi
import music21
from common import *
import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../sms-tools/software/models/'))
import harmonicModel as HM
import utilFunctions as UF
from scipy.signal import get_window
import progressbar

#THIS ALIGNMENT DOESN'T  SUPPORT RESTS YET FOR MIDI FILES
#only supports rests for musicxml file
def align(x, fs, notes, w, N, H, t, minf0, maxf0, f0et):
	print('f0 detection...')
	audio_f0s = HM.f0Detection(x, fs, w, N, H, t, minf0, maxf0, f0et)
	#audio_f0s = force_nonzero(audio_f0s)
	audio_log_f0s = np.log(audio_f0s)

	note_f0s = np.array([pretty_midi.note_number_to_hz(note.pitch) for note in notes])
	note_log_f0s = np.log(note_f0s)

	print('calculating cost matrix...')
	cost_matrix = np.clip((np.expand_dims(audio_log_f0s,1)-np.expand_dims(note_log_f0s,0))**2,0,0.015)

	partial_cost = np.copy(cost_matrix)
	prevs = np.zeros((len(audio_log_f0s),len(note_log_f0s)), dtype=int)-1

	print('partial cost sums...')
	partial_cost[0,0]=cost_matrix[0,0]
	for i in progressbar.progressbar(range(1,len(audio_log_f0s))):
		minj = max(0,len(note_log_f0s)-len(audio_log_f0s)+i)
		maxj = min(len(note_log_f0s),i+1)
		for j in range(minj,maxj-1):
			if j>0 and partial_cost[i-1,j-1]<partial_cost[i-1,j]:
				prevs[i,j]=j-1
			else:
				prevs[i,j]=j
			partial_cost[i,j] += partial_cost[i-1,prevs[i,j]]
		prevs[i,maxj-1] = maxj-2
		partial_cost[i,maxj-1] += partial_cost[i-1,prevs[i,maxj-1]]

	j = len(note_log_f0s)-1
	corresponding_note_idxs = np.zeros(len(audio_log_f0s))
	print('backtracking...')
	for i in progressbar.progressbar(range(len(audio_log_f0s)-1,-1,-1)):
		corresponding_note_idxs[i] = j
		j = prevs[i,j]

	prev = -1
	note_change_times = []
	print('converting...')
	for i in progressbar.progressbar(range(len(corresponding_note_idxs))):
		if prev != corresponding_note_idxs[i]:
			note_change_times.append(float(i)*len(x)/(fs*len(corresponding_note_idxs)))
		prev = corresponding_note_idxs[i]
	note_change_times = np.array(note_change_times)
	print(len(note_change_times))
	print('done!')

	return corresponding_note_idxs, note_change_times

#THIS ALIGNMENT DOESN'T  SUPPORT RESTS YET FOR MIDI FILES
#only supports rests for musicxml file
def align_memoptimized(x, fs, notes, w, N, H, t, minf0, maxf0, f0et):
	print('f0 detection...')
	audio_f0s = HM.f0Detection(x, fs, w, N, H, t, minf0, maxf0, f0et)
	#audio_f0s = force_nonzero(audio_f0s)
	audio_log_f0s = np.log(audio_f0s)

	note_f0s = np.array([pretty_midi.note_number_to_hz(note.pitch) for note in notes])
	note_log_f0s = np.log(note_f0s)

	print('feedforward costs...')
	prev_costs = np.clip((audio_log_f0s[0]-note_log_f0s)**2,0,0.015)
	prevs = np.zeros((len(audio_log_f0s),len(note_log_f0s)), dtype=int)-1
	for i in progressbar.progressbar(range(1,len(audio_log_f0s))):
		minj = max(0,len(note_log_f0s)-len(audio_log_f0s)+i)
		maxj = min(len(note_log_f0s),i+1)
		new_costs = np.clip((audio_log_f0s[i]-note_log_f0s)**2,0,0.015)
		for j in range(minj,maxj-1):
			if j>0 and prev_costs[j-1]<prev_costs[j]:
				prevs[i,j]=j-1
			else:
				prevs[i,j]=j
			new_costs[j] += prev_costs[prevs[i,j]]
		prevs[i,maxj-1] = maxj-2
		new_costs[maxj-1] += prev_costs[prevs[i,maxj-1]]
		prev_costs = new_costs

	j = len(note_log_f0s)-1
	corresponding_note_idxs = np.zeros(len(audio_log_f0s))
	print('backtracking...')
	for i in progressbar.progressbar(range(len(audio_log_f0s)-1,-1,-1)):
		corresponding_note_idxs[i] = j
		j = prevs[i,j]

	prev = -1
	note_change_times = []
	print('converting...')
	for i in progressbar.progressbar(range(len(corresponding_note_idxs))):
		if prev != corresponding_note_idxs[i]:
			note_change_times.append(float(i)*len(x)/(fs*len(corresponding_note_idxs)))
		prev = corresponding_note_idxs[i]
	note_change_times = np.array(note_change_times)
	print(len(note_change_times))
	print('done!')

	return corresponding_note_idxs, note_change_times


def force_nonzero(f0):
	f0 = np.copy(f0)
	for i in range(len(f0)-1):
		if f0[i+1]==0 and f0[i] > 0:
			f0[i+1] = f0[i]

	for i in range(1,len(f0)):
		if f0[-i-1] ==0 and f0[-i] >0:
			f0[-i-1] = f0[-i]

	return f0

def main(audioInFile, midiInFile, midiOutFile=None, txtOutFile=None, window='blackman', M=601, N=1024, t=-1000,
	nH=20, minf0=50, maxf0=4000, f0et=5):
	# size of fft used in synthesis
	Ns = 512
	# hop size (has to be 1/4 of Ns)
	H = 128

	fs, x = UF.wavread(audioInFile)

	w = get_window(window, M)

	if midiInFile[-4:]=='.mid':
		midiObj = pretty_midi.PrettyMIDI(midiInFile)
		print(len(midiObj.instruments[0].notes))
		notes = midiObj.instruments[0].notes
	else: #xml file
		notes = readMusicXMLNotesAndRests(midiInFile)

	corresponding_note_idxs, note_change_times = align_memoptimized(x, fs, notes, w, N, H, t, minf0, maxf0, f0et)

	if midiOutFile and midiInFile[-4:]=='.mid':
		midiObj.adjust_times(np.array([note.start for note in notes] + [notes[-1].end]), np.concatenate([note_change_times,[(len(x)-1)/float(fs)]]))
		print(len(midiObj.instruments[0].notes))
		midiObj.write(midiOutFile)
	if txtOutFile:
		np.savetxt(txtOutFile,np.expand_dims(note_change_times,axis=1),fmt='%.10f')