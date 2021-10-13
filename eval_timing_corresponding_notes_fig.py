import numpy as np
import misc
import seaborn as sns; sns.set(style="white", color_codes=True)
import matplotlib.pyplot as plt

def eval_correspondence(musicxml_and_ref_and_output_paths, ylabel=None):
	print('NOTE CORRESPONDENCE')
	timing_ref_list = []
	timing_output_list = []
	notes_and_rests_list = []
	written_note_pitches_ref_list = []
	written_note_pitches_output_list = []
	for musicxml_path, ref_path, output_path in musicxml_and_ref_and_output_paths:
		notes_and_rests = misc.musicXML_to_notes_and_rests(musicxml_path)
		timing_ref = misc.read_timings(ref_path)
		timing_output = misc.read_timings(output_path)
		notes_and_rests_list.append(notes_and_rests)
		timing_ref_list.append(timing_ref)
		timing_output_list.append(timing_output)
		written_note_pitches_ref = misc.written_note_pitches(musicxml_path, ref_path)
		written_note_pitches_output = misc.written_note_pitches(musicxml_path, output_path)
		if len(written_note_pitches_ref) > len(written_note_pitches_output):
			pad_size = len(written_note_pitches_ref)-len(written_note_pitches_output)
			written_note_pitches_output = np.pad(written_note_pitches_output,((0,pad_size)),'constant')
		elif len(written_note_pitches_output) > len(written_note_pitches_ref):
			pad_size = len(written_note_pitches_output)-len(written_note_pitches_ref)
			written_note_pitches_ref = np.pad(written_note_pitches_ref,((0,pad_size)),'constant')
		written_note_pitches_ref_list.append(written_note_pitches_ref)
		written_note_pitches_output_list.append(written_note_pitches_output)

	total_dur=0
	corr_dur=0
	noncorr_dur=0
	for i in range(len(timing_ref_list)):
		ref_j = 0
		output_j = 0
		timing_ref = timing_ref_list[i][1:]
		timing_output = timing_output_list[i][1:]
		notes_and_rests = notes_and_rests_list[i]
		seg_total_dur = max(timing_ref[-1],timing_output[-1])

		current_timing=0
		while ref_j < len(timing_ref) and output_j < len(timing_output):
			next_timing = min(timing_ref[ref_j],timing_output[output_j])
			if notes_and_rests[ref_j].pitch == notes_and_rests[output_j].pitch:
				corr_dur += next_timing-current_timing
			else:
				noncorr_dur += next_timing-current_timing

			if next_timing==timing_ref[ref_j]:
				ref_j+=1
			if next_timing==timing_output[output_j]:
				output_j+=1
			current_timing=next_timing

		noncorr_dur += seg_total_dur-current_timing
		total_dur += seg_total_dur
	print('total dur:',total_dur)
	print('corresponding dur:',corr_dur)
	print('non corresponding dur:',noncorr_dur)
	print('correspondence ratio:',corr_dur/total_dur)

	written_note_pitches_output = np.concatenate(written_note_pitches_output_list)
	written_note_pitches_ref = np.concatenate(written_note_pitches_ref_list)
	print('corresponding notes corr:')

	nonzero = np.logical_and(written_note_pitches_output != 0,written_note_pitches_ref != 0)
	written_note_pitches_output = written_note_pitches_output[nonzero]
	written_note_pitches_ref = written_note_pitches_ref[nonzero]
	print(np.corrcoef(written_note_pitches_ref,written_note_pitches_output,rowvar=False))
	print('corresponding notes corr:')
	print(np.corrcoef(written_note_pitches_ref,written_note_pitches_output,rowvar=False))
	h = sns.jointplot(x=written_note_pitches_ref, y=written_note_pitches_output, kind="kde")
	if ylabel:
		ylabel = 'pitch with {} output timing'.format(ylabel)
	else:
		ylabel = 'pitch with output timing'
	h.set_axis_labels('pitch with ref timing', ylabel)
	plt.show()


import argparse
import ntpath
import csv

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--data-paths-file',required=True)
	parser.add_argument(
		'--output-dir-timing')
	parser.add_argument(
		'--ylabel')
	
	args = parser.parse_args()

	with open(args.data_paths_file) as f:
		reader = csv.reader(f, delimiter="\t")
		data_paths = list(reader)

	if args.output_dir_timing:
		ref_and_output_paths_timing = [(timing_path, ntpath.join(args.output_dir_timing,ntpath.basename(timing_path)))
			for midi_path, timing_path, hps_path in data_paths]

		musicxml_and_ref_and_output_paths = [(musicxml_path, timing_path, ntpath.join(args.output_dir_timing,ntpath.basename(timing_path)))
			for musicxml_path, timing_path, hps_path in data_paths]
		eval_correspondence(musicxml_and_ref_and_output_paths, args.ylabel)
