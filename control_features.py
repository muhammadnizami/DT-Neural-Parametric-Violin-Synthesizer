import configparser
import numpy as np
from misc import *

config = configparser.ConfigParser()
config.read('config.ini')

Ns = int(config['Audio']['Ns'])
H = int(config['Audio']['H'])
fs = int(config['Audio']['fs'])

def control_features(sheet_file_path,timing_file_path,num_samples=None,pitch_scaler=None):
	notes = musicXML_to_notes_and_rests(sheet_file_path)

	timings = read_timings(timing_file_path)
	if len(timings)==len(notes):
		if num_samples is not None:
			timings = np.append(timings,num_samples*H/float(fs))
		else:
			raise 'num samples must not be None or length of timing must be length of notes + 1'
	if num_samples is None:
		num_samples = int(timings[-1]*fs/H)

	#note level features
	notes_pitch = np.array([note.pitch for note in notes])
	notes_written_length = np.array([note.end - note.start for note in notes])
	notes_written_start = np.array([note.start for note in notes])
	notes_actual_length = timings[1:]-timings[:-1]

	notes_key = np.array([note.keytonic for note in notes])

	notes_is_rest = notes_pitch==0
	notes_is_note = notes_pitch>0
	
	notes_pitch_relative = (notes_pitch - 24 - notes_key) % 12
	notes_pitch_relative_onehot = np.zeros((len(notes_pitch_relative),12))
	notes_pitch_relative_onehot[np.arange(len(notes_pitch_relative)),notes_pitch_relative]=np.where(
		notes_is_note,1,0)

	notes_bar_position = np.array([note.bar_position for note in notes])
	notes_beat_strength = np.array([note.beat_strength for note in notes])
	notes_quarter_length = np.array([note.quarter_length for note in notes])
	notes_is_grace = np.array([note.is_grace for note in notes])
	notes_has_trill = np.array([note.has_trill for note in notes])

	if pitch_scaler != None:
		notes_pitch = pitch_scaler.transform(notes_pitch.reshape(-1, 1))
	
	notes_control_features = np.column_stack((notes_pitch, notes_written_length, notes_written_start,
		notes_actual_length, notes_key, notes_pitch_relative_onehot, notes_bar_position, notes_beat_strength,
		notes_quarter_length, notes_is_grace, notes_has_trill, notes_is_rest, notes_is_note))
	notes_prev_control_features = np.pad(notes_control_features[:-1],((1,0),(0,0)),'constant')
	notes_next_control_features = np.pad(notes_control_features[1:],((0,1),(0,0)),'constant')

	notes_cur_and_context_control_features = np.column_stack((notes_control_features,
		notes_prev_control_features, notes_next_control_features))

	#sample level features
	#some are translated from note level features
	sample_indices = np.arange(num_samples)
	sample_times = sample_indices*H/float(fs)
	note_indices = np.digitize(sample_times,timings[:-1])-1

	current_note_control_features = notes_cur_and_context_control_features[note_indices]

	cur_note_actual_length = notes_actual_length[note_indices]

	in_note_pos = sample_times-timings[note_indices]
	in_note_pos_relative = in_note_pos / cur_note_actual_length

	return np.column_stack((current_note_control_features,in_note_pos,in_note_pos_relative))

control_feature_dim = 74 #THIS MUST BE CHANGED EVERYTIME THE CONTROL FEATURE FUNCTION IS CHANGED