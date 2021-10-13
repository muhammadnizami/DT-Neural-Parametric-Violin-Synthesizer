import numpy as np
from midiutil import MIDIFile
import music21
from misc import read_timings, freq_to_m
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

Ns = int(config['Audio']['Ns'])
H = int(config['Audio']['H'])
fs = int(config['Audio']['fs'])


int_volume_max=127 #for converting volume from 0.-1. to 0-127 ints
int_volume_min=0
def volume_to_int(volume_float):
	return np.clip(np.round(volume_float*(int_volume_max-int_volume_min)+int_volume_min),int_volume_min,int_volume_max).astype(np.int8)

mag_min=-120.
mag_max=0.

grace_placeholder_quarter_length=0.0625

channel=0
EXPRESSION_CC=11
VIOLIN_PROG_NUM=41

def calculate_bpm(note_quarters, dur_seconds):
	return np.clip((note_quarters/dur_seconds)*60,0,6e7)

def sample_index(time):
	return int(time*float(fs)/H)

def convert(hpsFile, sheetFile, timingFile, outputFile,use_pitch_bend=True,dynamics='Full'):
	#dynamics: None/'None', 'NoteWise', 'Full'
	score = music21.converter.parse(sheetFile)
	part = None
	notes = []

	for p in score:
		if isinstance(p,music21.stream.Part):
			assert part is None, "must be monophonic. found another part in file {}: {}".format(filepath,p)
			part = p

	if use_pitch_bend or (dynamics and dynamics.lower()!='none'):
		hfreq, hmag, stocEnv = [np.array(a) for a in zip(*np.load(hpsFile))]
	if dynamics and dynamics.lower()!='none':
		volumes = (10*np.log(np.sum(10**(hmag/10),axis=1))-mag_min)/(mag_max-mag_min)
	if use_pitch_bend:
		f0s = hfreq[:,0]
		f0s_m = freq_to_m(f0s)
	timings = read_timings(timingFile)

	MyMIDI = MIDIFile(1,file_format=0)
	MyMIDI.addProgramChange(0, channel, 0, 41)
	current_tempo_bpm = None
	note_i=0
	additional_offset=0
	for m in part:
		if isinstance(m,music21.stream.Measure):
			for n in m:
				time = n.offset+m.offset+additional_offset
				if isinstance(n, music21.note.Note):
					actual_dur_seconds = timings[note_i+1]-timings[note_i]
					quarterLength = n.quarterLength
					if n.duration.isGrace:
						quarterLength = grace_placeholder_quarter_length
						additional_offset += quarterLength
					actual_tempo = calculate_bpm(quarterLength,actual_dur_seconds)
					MyMIDI.addTempo(0,time,actual_tempo)

					start_sample_index = max(0,sample_index(timings[note_i]))
					end_sample_index = min(len(hfreq),sample_index(timings[note_i+1]))

					volume = 1.
					if dynamics and end_sample_index-start_sample_index > 0:
						if dynamics.lower() == 'notewise':
							volume = np.mean(volumes[start_sample_index:end_sample_index])
						elif dynamics.lower() == 'full':
							volume = np.max(volumes[start_sample_index:end_sample_index])
					volume_int = volume_to_int(volume)
					MyMIDI.addNote(0, channel, n.pitch.midi, time, quarterLength, volume_int)

					if (dynamics and dynamics.lower() == 'full') or use_pitch_bend:
						num_events = end_sample_index-start_sample_index
						event_times_in_note = time+np.arange(num_events)*quarterLength/num_events
					if (dynamics and dynamics.lower() == 'full'):
						in_note_volumes = volume_to_int(volumes[start_sample_index:end_sample_index]/volume)
						for i in range(num_events):
							MyMIDI.addControllerEvent(0,channel,event_times_in_note[i],EXPRESSION_CC,in_note_volumes[i])
					if use_pitch_bend:
						m_difs = f0s_m[start_sample_index:end_sample_index] - n.pitch.midi
						pitchWheel = (m_difs/2*8192).astype(np.int32)
						if np.any(np.logical_or(pitchWheel>8192,pitchWheel<-8192)):
							print('WARNING: excessive pitch bend', np.where(np.logical_or(pitchWheel>8192,pitchWheel<-8192)))
						pitchWheel = np.clip(pitchWheel,-8192,8192)
						for i in range(num_events):
							MyMIDI.addPitchWheelEvent(0, channel, event_times_in_note[i], pitchWheel[i])

					note_i += 1
				elif isinstance(n, music21.note.Rest):
					actual_dur_seconds = timings[note_i+1]-timings[note_i]
					quarterLength = n.quarterLength
					actual_tempo = calculate_bpm(n.quarterLength,actual_dur_seconds)
					MyMIDI.addTempo(0,time,actual_tempo)
					note_i += 1
				elif isinstance(n, music21.chord.Chord):
					print("WARNING: is not monophonic. Ignoring chord {} and replacing it with rests...".format(n))
					notes.append(NoteOrRest(0))
				elif isinstance(n, music21.key.KeySignature):
					pass
				elif isinstance(n, music21.tempo.MetronomeMark):
					# tempo_marking=n
					current_tempo_bpm=n.getQuarterBPM()
					MyMIDI.addTempo(0,time,current_tempo_bpm)
	with open(outputFile,'wb') as f:
		MyMIDI.writeFile(f)

import ntpath
import csv, os

def main(input_dir, output_dir, data_paths_file, use_pitch_bend=True,dynamics='Full'):
	dynamics = dynamics.lower()
	if os.path.isdir(input_dir) and os.path.isdir(output_dir):
		with open(data_paths_file) as f:
			reader = csv.reader(f, delimiter="\t")
			data_paths = list(reader)
		for sheet_path, timing_path, hps_path in data_paths:
			outfile_hps=ntpath.join(output_dir,ntpath.basename(hps_path))
			infile_hps=ntpath.join(input_dir,ntpath.basename(hps_path))
			timing_path=ntpath.join(input_dir,ntpath.basename(timing_path))

			outfile_midi = outfile_hps.replace('.guidedtwmhps','').replace('.npy','') + \
				('-pitchbend' if use_pitch_bend else '') + \
				'-'+dynamics+'dynamics'+'.mid'

			print('converting',infile_hps,'to',outfile_midi)
			convert(infile_hps,sheet_path,timing_path,outfile_midi, use_pitch_bend, dynamics)
			
	else:
		print('paths must be both existing directory or input path is \'.npy\' file')

import argparse
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input-dir',required=True)
	parser.add_argument('--output-dir',required=True)
	parser.add_argument('--data-paths-file',required=True)
	parser.add_argument('--use-pitch-bend',action='store_true')
	parser.add_argument('--dynamics',choices=['full','notewise','no'],default='no')
	args = parser.parse_args()
	input_dir = args.input_dir
	output_dir = args.output_dir
	data_paths_file = args.data_paths_file
	use_pitch_bend = args.use_pitch_bend
	dynamics = args.dynamics
	main(input_dir,output_dir,data_paths_file, use_pitch_bend, dynamics)