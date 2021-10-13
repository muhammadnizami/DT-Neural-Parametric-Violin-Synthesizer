import numpy as np
from misc import *

def sheet_features(sheet_file_path,num_prev_context,num_next_context):
	notes = musicXML_to_notes_and_rests(sheet_file_path)

	#note level features
	notes_pitch = np.array([note.pitch for note in notes])
	notes_written_length = np.array([note.end - note.start for note in notes])
	notes_written_start = np.array([note.start for note in notes])

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

	notes_control_features = np.column_stack((notes_pitch, notes_written_length, notes_written_start,
		notes_key, notes_pitch_relative_onehot, notes_bar_position, notes_beat_strength, notes_quarter_length,
		notes_is_grace, notes_has_trill,notes_is_rest, notes_is_note))
	notes_prev_control_features = [np.pad(notes_control_features[:-i],((np.min((i,len(notes))),0),(0,0)),'constant') for i in range(1,num_prev_context+1)]
	notes_next_control_features = [np.pad(notes_control_features[i:],((0,np.min((i,len(notes)))),(0,0)),'constant') for i in range(1,num_next_context+1)]

	notes_cur_and_context_control_features = np.column_stack([notes_control_features]
		+ notes_prev_control_features + notes_next_control_features)

	return notes_cur_and_context_control_features

note_feature_dim = 23 #THIS MUST BE CHANGED EVERYTIME THE CONTROL FEATURE FUNCTION IS CHANGED

def sheet_feature_dim(num_prev_context, num_next_context):
	return note_feature_dim * (1 + num_prev_context + num_next_context)

def prev_timingdev_feature_dim(num_prev_context):
	return num_prev_context

def all_feature_dim(num_prev_context, num_next_context):
	return sheet_feature_dim(num_prev_context,num_next_context) + prev_timingdev_feature_dim(num_prev_context)

def dataset_features(sheet_file_path, binned_timingdevs, num_prev_context, num_next_context):
	sheet_f = sheet_features(sheet_file_path, num_prev_context, num_next_context)
	prev_timingdevs = [np.pad(binned_timingdevs[:-i],((np.min((i,len(sheet_f))),0),(0,0)),'constant') for i in range(1,num_prev_context+1)]
	return np.column_stack([sheet_f] + prev_timingdevs)