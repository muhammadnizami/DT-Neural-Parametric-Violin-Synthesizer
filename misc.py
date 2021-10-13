import numpy as np
import configparser
import music21

config = configparser.ConfigParser()
config.read('config.ini')

Ns = int(config['Audio']['Ns'])
H = int(config['Audio']['H'])
fs = int(config['Audio']['fs'])
spikethreshold = float(config['Audio']['spikeThreshold'])

class NoteOrRest:
	def __init__(self, pitch, keytonic, start, end, bar_position, beat_strength, quarter_length, is_grace=False, has_trill=False):
		self.pitch=pitch
		self.keytonic = keytonic
		self.start = start
		self.end = end
		self.bar_position = bar_position
		self.beat_strength = beat_strength
		self.quarter_length = quarter_length
		self.is_grace = is_grace
		self.has_trill = has_trill

	def isRest(self):
		return self.pitch==0

def freq_to_m(fm):
	return np.where(fm>0,12*np.log2(fm/440) + 69,0)

def m_to_freq(m):
	return np.power(2,(m-69)/12)*(440)

def m_to_deviation(m, m_written):
	#return m
	return m-12*np.log2(np.arange(0,m.shape[1])+1)-np.expand_dims(m_written,axis=1)

def deviation_to_m(dev, m_written):
	#return dev
	return (dev+np.expand_dims(m_written,axis=1))+12*np.log2(np.arange(0,dev.shape[1])+1)

def musicXML_to_notes_and_rests(filepath):
	score = music21.converter.parse(filepath)
	part = None
	notes = []

	for p in score:
		if isinstance(p,music21.stream.Part):
			assert part is None, "must be monophonic. found another part in file {}: {}".format(filepath,p)
			part = p

	measure_time = 0
	key_signature = 0
	time_signature = None
	tempo_marking = None
	for m in part:
		if isinstance(m,music21.stream.Measure):
			for n in m:
				if isinstance(n, music21.note.Note):
					start = measure_time + n.offset*tempo_marking.secondsPerQuarter()
					end = measure_time + n.offset*tempo_marking.secondsPerQuarter() + n.seconds
					keytonic = key_signature.asKey().tonic.midi
					has_trill = any([isinstance(x,music21.expressions.Trill) for x  in n.expressions])
					bar_position = n.offset/m.quarterLength
					beat_strength = n.beatStrength
					quarter_length = n.quarterLength
					is_grace = n.duration.isGrace
					notes.append(NoteOrRest(n.pitch.midi, keytonic, start, end, bar_position, beat_strength, quarter_length, is_grace, has_trill))
				elif isinstance(n, music21.note.Rest):
					start = measure_time + n.offset*tempo_marking.secondsPerQuarter()
					end = measure_time + n.offset*tempo_marking.secondsPerQuarter() + n.seconds
					keytonic = key_signature.asKey().tonic.midi
					bar_position = n.offset/m.quarterLength
					beat_strength = n.beatStrength
					quarter_length = n.quarterLength
					notes.append(NoteOrRest(0,keytonic,start,end,bar_position, beat_strength, quarter_length))
				elif isinstance(n, music21.chord.Chord):
					print("WARNING: is not monophonic. Ignoring chord {} and replacing it with rests...".format(n))
					notes.append(NoteOrRest(0))
				elif isinstance(n, music21.key.KeySignature):
					key_signature=n
				elif isinstance(n, music21.tempo.MetronomeMark):
					tempo_marking=n
			measure_time = measure_time + m.seconds

	return notes

def generate_written_timings(sheet_path, outputFile):
	notes_and_rests = musicXML_to_notes_and_rests(sheet_path)
	actdurs = [n.end-n.start for n in notes_and_rests]

	timings = [0.]
	for i in range(len(actdurs)):
		timings.append(timings[-1]+actdurs[i])

	np.savetxt(outputFile,np.expand_dims(timings,axis=1),fmt='%.10f')

def read_timings(timing_file_path):
	with open(timing_file_path) as f:
		lines = f.read().splitlines()
	return np.array([float(x) for x in lines])

def written_note_pitches(sheet_file_path,timing_file_path,num_samples=None):
	notes = musicXML_to_notes_and_rests(sheet_file_path)
	timings = read_timings(timing_file_path)
	for i in range(1,len(timings)):
		if timings[i]<timings[i-1]:
			timings[i]=timings[i-1]
	if len(timings)==len(notes):
		if num_samples is not None:
			timings = np.append(timings,num_samples*H/float(fs))
		else:
			raise 'num samples must not be None or length of timing must be length of notes + 1'
	if num_samples is None:
		num_samples = int(timings[-1]*fs/H)

	#note level features
	notes_pitch = np.array([note.pitch for note in notes])

	#sample level features
	#some are translated from note level features
	sample_indices = np.arange(num_samples)
	sample_times = sample_indices*H/float(fs)
	note_indices = np.digitize(sample_times,timings[:-1])-1

	cur_note_pitch = notes_pitch[note_indices]
	return cur_note_pitch

def preprocess_hps_data(hps, sheet_file_path, timing_file_path):
	hfreq, hmag, stocEnv = [np.array(a) for a in zip(*hps)]
	hfreqp = []
	guidepitch = written_note_pitches(sheet_file_path,timing_file_path,len(hps))
	for i in range(len(hps)):
		if i%100==0:
			print(i,len(hps))
		if np.any(hfreq != 0):
			hfreq[i], hmag[i] = harmonicCorrection(hfreq[i], hmag[i], hfreqp, guidepitch[i])
			hfreqp = hfreq[i]
	hfreq, hmag = spikeRemoval(hfreq,hmag,None,spikethreshold,np.expand_dims(guidepitch,axis=1))
	hmag = interpolate_nans(hmag)
	return hfreq, hmag, stocEnv