import numpy as np
import misc

def eval_times(ref_and_output_paths):
	#ref_and_output_paths: array of (ref_path, output_path) pairs
	timing_ref_list = []
	timing_output_list = []
	for ref_path, output_path in ref_and_output_paths:
		timing_ref = misc.read_timings(ref_path)
		timing_output = misc.read_timings(output_path)
		timing_ref_list.append(timing_ref)
		timing_output_list.append(timing_output)
	timing_ref = np.concatenate(timing_ref_list)
	timing_output = np.concatenate(timing_output_list)
	print('evaluating...')
	print('times correlation:')
	print(np.corrcoef(timing_ref,timing_output))
	print('times rmse: ', np.sqrt(np.mean((timing_ref-timing_output)**2)))
	print('times mae: ', np.mean(np.abs(timing_ref-timing_output)))

def eval_correspondence(musicxml_and_ref_and_output_paths):
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
	print(np.corrcoef(written_note_pitches_ref,written_note_pitches_output,rowvar=False))

def eval_durs(ref_and_output_paths):
	#ref_and_output_paths: array of (ref_path, output_path) pairs
	durs_ref_list = []
	durs_output_list = []
	for ref_path, output_path in ref_and_output_paths:
		timing_ref = misc.read_timings(ref_path)
		timing_output = misc.read_timings(output_path)
		durs_ref = timing_ref[1:]-timing_ref[:-1]
		durs_output = timing_output[1:]-timing_output[:-1]
		durs_ref_list.append(durs_ref)
		durs_output_list.append(durs_output)
	durs_ref = np.concatenate(durs_ref_list)
	durs_output = np.concatenate(durs_output_list)
	print('evaluating...')
	print('durs correlation:')
	print(np.corrcoef(durs_ref,durs_output))
	print('durs rmse: ', np.sqrt(np.mean((durs_ref-durs_output)**2)))

def eval_dur_devs(musicxml_and_ref_and_output_paths):
	dur_devs_ref_list = []
	dur_devs_output_list = []
	for musicxml_path, ref_path, output_path in musicxml_and_ref_and_output_paths:
		notes_and_rests = misc.musicXML_to_notes_and_rests(musicxml_path)
		writdurs = [n.end-n.start for n in notes_and_rests]
		timing_ref = misc.read_timings(ref_path)
		timing_output = misc.read_timings(output_path)
		durs_ref = timing_ref[1:]-timing_ref[:-1]
		durs_output = timing_output[1:]-timing_output[:-1]
		dur_devs_ref = durs_ref-writdurs
		dur_devs_output = durs_output-writdurs
		dur_devs_ref_list.append(dur_devs_ref)
		dur_devs_output_list.append(dur_devs_output)
	dur_devs_ref = np.concatenate(dur_devs_ref_list)
	dur_devs_output = np.concatenate(dur_devs_output_list)
	print('evaluating...')
	print('dur devs correlation:')
	print(np.corrcoef(durs_ref,durs_output))
	print('dur devs rmse: ', np.sqrt(np.mean((durs_ref-durs_output)**2)))

def eval_harmonics(ref_and_output_paths,ignore_stoc=False,replace_nans_with_zeros=False):
	#ref_and_output_paths: array of (ref_path, output_path) pairs

	print('loading and preparing data...')
	hfreq_ref_list = []
	hmag_ref_list = []
	stocEnv_ref_list = []
	hfreq_output_list = []
	hmag_output_list = []
	stocEnv_output_list = []
	for ref_path, output_path in ref_and_output_paths:
		hfreq_ref, hmag_ref, stocEnv_ref = [np.array(a) for a in zip(*np.load(ref_path))]
		hfreq_output, hmag_output, stocEnv_output = [np.array(a) for a in zip(*np.load(output_path))]
		if len(hfreq_ref) > len(hfreq_output):
			pad_size = len(hfreq_ref)-len(hfreq_output)
			hfreq_output = np.pad(hfreq_output,((0,pad_size),(0,0)),'constant')
			hmag_output = np.pad(hmag_output,((0,pad_size),(0,0)),'constant')
			stocEnv_output = np.pad(stocEnv_output,((0,pad_size),(0,0)),'constant')
		elif len(hfreq_output) > len(hfreq_ref):
			pad_size = len(hfreq_output)-len(hfreq_ref)
			hfreq_ref = np.pad(hfreq_ref,((0,pad_size),(0,0)),'constant')
			hmag_ref = np.pad(hmag_ref,((0,pad_size),(0,0)),'constant')
			stocEnv_ref = np.pad(stocEnv_ref,((0,pad_size),(0,0)),'constant')
		hfreq_ref_list.append(hfreq_ref)
		hmag_ref_list.append(hmag_ref)
		stocEnv_ref_list.append(stocEnv_ref)
		hfreq_output_list.append(hfreq_output)
		hmag_output_list.append(hmag_output)
		stocEnv_output_list.append(stocEnv_output)
	hfreq_ref = np.concatenate(hfreq_ref_list)
	hmag_ref = np.concatenate(hmag_ref_list)
	stocEnv_ref = np.concatenate(stocEnv_ref_list)
	hfreq_output = np.concatenate(hfreq_output_list)
	hmag_output = np.concatenate(hmag_output_list)
	stocEnv_output = np.concatenate(stocEnv_output_list)
	if ignore_stoc:
		stocEnv_output=np.zeros(stocEnv_output.shape)

	mean_func = np.nanmean if not replace_nans_with_zeros else (lambda x: np.mean(np.nan_to_num(x.copy(),0)))
	corrcoef_mean_func = lambda x: np.tanh(mean_func(np.arctanh(x)))
	print('evaluating...')
	hfreq_corrcoefs = np.diagonal(np.corrcoef(hfreq_ref,hfreq_output,rowvar=False)[hfreq_ref.shape[1]:,:hfreq_ref.shape[1]])
	print('hfreq correlation coefficients: ', hfreq_corrcoefs)
	print('hfreq mean correlation coefficient: ', corrcoef_mean_func(hfreq_corrcoefs))
	hmag_corrcoefs = np.diagonal(np.corrcoef(hmag_ref,hmag_output,rowvar=False)[hmag_ref.shape[1]:,:hmag_ref.shape[1]])
	print('hmag correlation coefficients: ', hmag_corrcoefs)
	print('hmag mean correlation coefficient: ', corrcoef_mean_func(hmag_corrcoefs))
	stocEnv_corrcoefs = np.diagonal(np.corrcoef(stocEnv_ref,stocEnv_output,rowvar=False)[stocEnv_ref.shape[1]:,:stocEnv_ref.shape[1]])
	print('stocEnv correlation coefficients: ', stocEnv_corrcoefs)
	print('stocEnv mean correlation coefficient: ', corrcoef_mean_func(stocEnv_corrcoefs))
	print('mean all correlation coefficients: ', corrcoef_mean_func(np.concatenate([hfreq_corrcoefs, hmag_corrcoefs, stocEnv_corrcoefs])))

	print('')
	hfreq_rmse = np.sqrt(mean_func((hfreq_output-hfreq_ref)**2))
	hmag_rmse = np.sqrt(mean_func((hmag_output-hmag_ref)**2))
	stocEnv_rmse = np.sqrt(mean_func((stocEnv_output-stocEnv_ref)**2))
	all_rmse = np.sqrt(mean_func((np.concatenate([hfreq_output,hmag_output,stocEnv_output],axis=-1)
		-np.concatenate([hfreq_ref,hmag_ref,stocEnv_ref],axis=-1))**2))
	print('hfreq RMSE:',hfreq_rmse)
	print('hmag RMSE:',hmag_rmse)
	print('stocEnv RMSE:',stocEnv_rmse)
	print('all RMSE:',all_rmse)

from scipy.signal import get_window
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tools/sms-tools/software/models/'))
import utilFunctions as UF
import stft as STFT

def eval_spectral(ref_and_output_paths, window = 'hamming', M = 1024, N = 1024, H = 512):

	print('loading and preparing data...')
	mX_ref_list = []
	mX_output_list = []

	for ref_path, output_path in ref_and_output_paths:	# read input sound (monophonic with sampling rate of 44100)
		w = get_window(window, M)
		_, x = UF.wavread(ref_path)
		mX_ref, _ = STFT.stftAnal(x, w, N, H)
		_, x = UF.wavread(output_path)
		mX_output, _ = STFT.stftAnal(x, w, N, H)

		lref = len(mX_ref)
		loutput = len(mX_output)
		if lref > loutput:
			mX_output = np.pad(mX_output,((0,lref-loutput),(0,0)),'constant')
		elif loutput > lref:
			mX_ref = np.pad(mX_ref,((0,loutput-lref),(0,0)),'constant')

		mX_ref_list.append(mX_ref)
		mX_output_list.append(mX_output)

	mX_ref = np.concatenate(mX_ref_list)
	mX_output = np.concatenate(mX_output_list)

	print('evaluating...')
	print('mX rmse: ', np.sqrt(np.nanmean((mX_output-mX_ref)**2)))
	corrcoef = np.diagonal(np.corrcoef(mX_ref,mX_output,rowvar=False)[mX_ref.shape[1]:,:mX_ref.shape[1]])
	print('mX corrcoef: ', corrcoef)
	print('mX mean corrcoef:', np.nanmean(corrcoef))

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
		'--output-dir-hps')
	parser.add_argument(
		'--output-dir-wav')
	parser.add_argument(
		'--replace', required=False, nargs=2)
	parser.add_argument(
		'--ignore-stoc',action='store_true'
	)
	parser.add_argument(
		'--replace-nans-with-zeros',action='store_true'
	)
	
	args = parser.parse_args()

	with open(args.data_paths_file) as f:
		reader = csv.reader(f, delimiter="\t")
		data_paths = list(reader)

	if args.output_dir_timing:
		ref_and_output_paths_timing = [(timing_path, ntpath.join(args.output_dir_timing,ntpath.basename(timing_path)))
			for midi_path, timing_path, hps_path in data_paths]

		eval_times(ref_and_output_paths_timing)
		eval_durs(ref_and_output_paths_timing)

		musicxml_and_ref_and_output_paths = [(musicxml_path, timing_path, ntpath.join(args.output_dir_timing,ntpath.basename(timing_path)))
			for musicxml_path, timing_path, hps_path in data_paths]
		eval_correspondence(musicxml_and_ref_and_output_paths)
		eval_dur_devs(musicxml_and_ref_and_output_paths)

	if args.output_dir_hps:
		ref_and_output_paths_hps = [(hps_path, 
			ntpath.join(args.output_dir_hps,ntpath.basename(hps_path) if not args.replace else 
				ntpath.basename(hps_path).replace(args.replace[0],args.replace[1])))
			for midi_path, timing_path, hps_path in data_paths]

		eval_harmonics(ref_and_output_paths_hps,args.ignore_stoc, args.replace_nans_with_zeros)

	if args.output_dir_wav:
		ref_and_output_paths_wav = []
		for refhps, outhps in ref_and_output_paths_hps:
			refwav = refhps.replace('.guidedtwmhps.npy','.wav').replace('/hps/','/wavs/')
			outwav = ntpath.join(args.output_dir_wav,ntpath.basename(outhps))[:-4]+'.wav'
			ref_and_output_paths_wav.append((refwav,outwav))
		print(ref_and_output_paths_wav)
		eval_spectral(ref_and_output_paths_wav)
