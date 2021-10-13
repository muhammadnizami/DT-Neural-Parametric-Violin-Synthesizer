import csv
import shutil, os
import argparse
import timing_dt
import configparser
import json
from sklearn.externals import joblib
config = configparser.ConfigParser()
config.read('config.ini')

if __name__ == '__main__':
	train_data_paths_file = config['Experiment']['train_data_paths_file']
	model_dir = config['Experiment']['model_dir']

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--timing',action='store_true')
	parser.add_argument(
		'--f0',action='store_true')
	parser.add_argument(
		'--freq',action='store_true')
	parser.add_argument(
		'--mag',action='store_true')
	parser.add_argument(
		'--stoc',action='store_true')

	#arguments for hyperparameter tuning experiment
	parser.add_argument('--train-data-paths-file')
	parser.add_argument('--model-dir')
	parser.add_argument('--timing-model-dir')
	parser.add_argument('--f0-model-dir')
	parser.add_argument('--freq-model-dir')
	parser.add_argument('--mag-model-dir')
	parser.add_argument('--stoc-model-dir')
	parser.add_argument('--change-timing-config')
	parser.add_argument('--change-f0-config')
	parser.add_argument('--change-freq-config')
	parser.add_argument('--change-mag-config')
	parser.add_argument('--change-stoc-config')
	args = parser.parse_args()

	if args.train_data_paths_file:
		train_data_paths_file = args.train_data_paths_file
	if args.model_dir:
		model_dir = args.model_dir
	timing_model_dir = model_dir
	f0_model_dir = model_dir
	freq_model_dir = model_dir
	mag_model_dir = model_dir
	stoc_model_dir = model_dir
	if args.timing_model_dir:
		timing_model_dir = args.timing_model_dir
	if args.f0_model_dir:
		f0_model_dir = args.f0_model_dir
	if args.freq_model_dir:
		freq_model_dir = args.freq_model_dir
	if args.mag_model_dir:
		mag_model_dir = args.mag_model_dir
	if args.stoc_model_dir:
		stoc_model_dir = args.stoc_model_dir

	timing_train = args.timing
	f0_train = args.f0
	freq_train = args.freq
	mag_train = args.mag
	stoc_train = args.stoc
	
	if not any([timing_train, f0_train, freq_train, mag_train, stoc_train]):
		timing_train = True
		f0_train = True
		freq_train = True
		mag_train = True
		stoc_train = True
	if any([f0_train, freq_train, mag_train, stoc_train]):
		import modified_wavenet_hps

	if args.change_timing_config:
		timing_dt.change_timing_config(json.loads(args.change_timing_config))
	if args.change_f0_config:
		modified_wavenet_hps.change_f0_config(json.loads(args.change_f0_config))
	if args.change_freq_config:
		modified_wavenet_hps.change_freq_config(json.loads(args.change_freq_config))
	if args.change_mag_config:
		modified_wavenet_hps.change_mag_config(json.loads(args.change_mag_config))
	if args.change_stoc_config:
		modified_wavenet_hps.change_stoc_config(json.loads(args.change_stoc_config))

	with open(train_data_paths_file) as f:
		reader = csv.reader(f, delimiter="\t")
		dataset_paths = list(reader)

		if timing_train:
			timing_model = timing_dt.train_timing_model(dataset_paths)
			joblib.dump(timing_model, os.path.join(timing_model_dir,'timing_model'))

		if f0_train:
			f0_model = modified_wavenet_hps.train_f0_model(dataset_paths)
			f0_model.save_weights(os.path.join(f0_model_dir,'f0_model.h5'))

		if freq_train:
			freq_model = modified_wavenet_hps.train_freq_model(dataset_paths)
			freq_model.save_weights(os.path.join(freq_model_dir,'freq_model.h5'))

		if mag_train:
			mag_model, mag_scaler = modified_wavenet_hps.train_mag_model(dataset_paths)
			mag_model.save_weights(os.path.join(mag_model_dir,'mag_model.h5'))
			joblib.dump(mag_scaler,os.path.join(mag_model_dir,'mag_scaler'))
		elif stoc_train:
			mag_scaler = joblib.load(os.path.join(mag_model_dir,'mag_scaler'))

		if stoc_train:
			stoc_model, stoc_scaler = modified_wavenet_hps.train_stoc_model(dataset_paths,mag_scaler)
			stoc_model.save_weights(os.path.join(stoc_model_dir,'stoc_model.h5'))
			joblib.dump(stoc_scaler,os.path.join(stoc_model_dir,'stoc_scaler'))
