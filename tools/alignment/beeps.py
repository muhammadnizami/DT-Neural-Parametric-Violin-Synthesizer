import numpy as np
from common import *
import pretty_midi
import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../sms-tools/software/models/'))
import utilFunctions as UF

def generatef0beeps(midiFile, timingFile, outFile, fs):
	notes = readNotes(midiFile)
	freqs = np.array([pretty_midi.note_number_to_hz(note.pitch) for note in notes])
	timings = read_timings(timingFile)
	num_samples = int(timings[-1]*fs)
	sample_indices = np.arange(num_samples)
	sample_times = sample_indices/float(fs)
	note_indices = np.digitize(sample_times,timings[:-1])-1

	sample_freqs=freqs[note_indices]
	sample = np.sin(2*np.pi*sample_freqs*sample_times)
	UF.wavwrite(sample, fs, outFile)