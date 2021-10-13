import configparser
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np
import progressbar
from random import sample
import misc

config = configparser.ConfigParser()
config.read('config.ini')
audio_config = config['Audio']
audio_mag_t = float(config['Audio']['t'])

def parsemodelconf(confdict):
	return {key: (float(confdict[key]) if key in {'corruption', 'spike_prob', 'spike_width',
		'gamma_u', 'gamma_s', 'gamma_w', 'epsilon', 'lr', 'decay', 'min_delta'} else 
		[int(factor) for factor in confdict[key].split(',')] if key=='dilation_factors' else
		[float(factor) for factor in confdict[key].split(',')] if key=='temperature' else
		confdict[key] if key in {'output_dist','monitor','generation_algorithm'} else
		int(confdict[key])
		) for key in confdict }

f0_config=parsemodelconf(config['F0Model'])
freq_config=parsemodelconf(config['FreqModel'])
mag_config=parsemodelconf(config['MagModel'])
stoc_config=parsemodelconf(config['StocModel'])

def parsemodelconf(confdict):
	return {key: (float(confdict[key]) if key in {'corruption', 'spike_prob', 'spike_width',
		'gamma_u', 'gamma_s', 'gamma_w', 'epsilon', 'lr', 'decay', 'min_delta'} else 
		[int(factor) for factor in confdict[key].split(',')] if key=='dilation_factors' else
		[float(factor) for factor in confdict[key].split(',')] if key=='temperature' else
		confdict[key] if key in {'output_dist','monitor','generation_algorithm'} else
		int(confdict[key])
		) for key in confdict }

def train(data_paths, sample_ratio=1.0, prev_weight=0.33):
	start = time.time()
	all_hps = []
	all_features = []
	for sheet_path, timing_path, hps_path in data_paths:
		original_hps = np.array([np.concatenate(x) for x in np.load(hps_path)])
		written_note_pitches = misc.written_note_pitches(sheet_path,timing_path,len(original_hps));
		hps = np.copy(original_hps)
		hps[:,:freq_config['feature_dim']]=misc.m_to_deviation(misc.freq_to_m(hps[:,:freq_config['feature_dim']]),written_note_pitches)
		features = np.column_stack([hps,np.pad(hps[:-1],((1,0),(0,0)),'constant')])

		all_hps.append(hps)
		all_features.append(features)
	all_hps=np.concatenate(all_hps)
	all_features=np.concatenate(all_features)
	if sample_ratio<1.0:
		sample_idx = np.random.randint(len(all_hps),size=int(sample_ratio*len(all_hps)))
		all_hps=all_hps[sample_idx]
		all_features=all_features[sample_idx]
	end = time.time()
	print('preprocess',end-start)

	start = time.time()
	scaler = MinMaxScaler((0,1))
	scaler.fit(all_features)
	scaler.min_*=prev_weight
	scaler.scale_*=prev_weight
	all_features_scaled = scaler.transform(all_features)
	model = NearestNeighbors(1,leaf_size=15)
	model.fit(all_features_scaled)
	end = time.time()
	print('training',end-start)
	neigh_arr = all_hps

	return model, neigh_arr, scaler

def convert(model,neigh_arr,scaler,input_sheet_file,input_timing_file,input_hps_file,output_file):
	input_hps = np.array([np.concatenate(x) for x in np.load(input_hps_file)])
	written_note_pitches = misc.written_note_pitches(input_sheet_file,input_timing_file,len(input_hps));
	input_hps[:,:freq_config['feature_dim']]=misc.m_to_deviation(misc.freq_to_m(input_hps[:,:freq_config['feature_dim']]),written_note_pitches)
	input_features=np.column_stack([input_hps,np.pad(input_hps[:-1],((1,0),(0,0)),'constant')])
	input_features_scaled = scaler.transform(input_features)
	output_dists, output_indices = model.kneighbors(input_features_scaled)
	num = len(input_hps)
	output_hps = [None]*num
	for i in range(num):
		idx = output_indices[i][0]
		hfreq=misc.m_to_freq(misc.deviation_to_m(neigh_arr[idx:idx+1,:freq_config['feature_dim']],written_note_pitches[i:i+1]))[0]
		output_hps[i] = [hfreq,#input_hps[i],
			neigh_arr[idx,freq_config['feature_dim']:freq_config['feature_dim']+mag_config['feature_dim']],
			neigh_arr[idx,freq_config['feature_dim']+mag_config['feature_dim']:freq_config['feature_dim']+mag_config['feature_dim']+stoc_config['feature_dim']]]
	output_hps = np.array(output_hps)
	print('shape',output_hps.shape)
	print('saving...')
	np.save(output_file,output_hps)



import ntpath
import csv, os

def main(model, neigh_arr, scaler, input_dir, output_dir, data_paths_file, use_original_timing):
	if os.path.isdir(input_dir) and os.path.isdir(output_dir):
		with open(data_paths_file) as f:
			reader = csv.reader(f, delimiter="\t")
			data_paths = list(reader)
		for sheet_path, timing_path, hps_path in data_paths:
			outfile_hps=ntpath.join(output_dir,ntpath.basename(hps_path))
			infile_hps=ntpath.join(input_dir,ntpath.basename(hps_path))
			if not use_original_timing:
				timing_path=ntpath.join(input_dir,ntpath.basename(timing_path))

			print('converting',infile_hps,'to',outfile_hps)
			convert(model, neigh_arr, scaler, sheet_path,timing_path,infile_hps,outfile_hps)
			
	else:
		print('paths must be both existing directory or input path is \'.npy\' file')

import argparse
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input-dir',required=True)
	parser.add_argument('--output-dir',required=True)
	parser.add_argument('--data-paths-file',required=True)
	parser.add_argument('--use-original-timing',action='store-true')
	args = parser.parse_args()
	input_dir = args.input_dir
	output_dir = args.output_dir
	main(input_dir,output_dir)