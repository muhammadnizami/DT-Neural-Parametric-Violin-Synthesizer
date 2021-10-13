import csv
import shutil, os
import timing_dt
import configparser
import argparse
import ntpath
import json
from sklearn.externals import joblib

config = configparser.ConfigParser()
config.read('config.ini')
if config['Experiment']['generate_use_cpu']=='True':
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
	model_dir = config['Experiment']['model_dir']
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--data-paths-file',required=True)
	parser.add_argument(
		'--output-dir',required=True)
	
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
	parser.add_argument('--only',choices=['timing','f0','freq','mag','stoc'])
	parser.add_argument('--no-timing-model', action='store_true')
	parser.add_argument('--existing-data-dir',help='used if --only')
	parser.add_argument('--existing-timing-dir',help='used if --only')
	args = parser.parse_args()

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
	if args.change_timing_config:
		timing_dt.change_timing_config(json.loads(args.change_timing_config))
	if args.only != 'timing':
		import modified_wavenet_hps
	if args.change_f0_config:
		modified_wavenet_hps.change_f0_config(json.loads(args.change_f0_config))
	if args.change_freq_config:
		modified_wavenet_hps.change_freq_config(json.loads(args.change_freq_config))
	if args.change_mag_config:
		modified_wavenet_hps.change_mag_config(json.loads(args.change_mag_config))
	if args.change_stoc_config:
		modified_wavenet_hps.change_stoc_config(json.loads(args.change_stoc_config))

	with open(args.data_paths_file) as f:
		reader = csv.reader(f, delimiter="\t")
		data_paths = list(reader)

		if args.only in {'timing', None} and not args.no_timing_model:
			timing_model = joblib.load(os.path.join(timing_model_dir, 'timing_model'))

		f0_model, freq_model, mag_model, stoc_model = None, None, None, None
		if args.only in {'f0',None}:
			f0_model = modified_wavenet_hps.create_f0_model()
			f0_model.load_weights(os.path.join(f0_model_dir,'f0_model.h5'))
		if args.only in {'freq',None}:
			freq_model = modified_wavenet_hps.create_freq_model()
			freq_model.load_weights(os.path.join(freq_model_dir,'freq_model.h5'))
		if args.only in {'mag',None}:
			mag_model = modified_wavenet_hps.create_mag_model()
			mag_model.load_weights(os.path.join(mag_model_dir,'mag_model.h5'))
		mag_scaler = joblib.load(os.path.join(mag_model_dir,'mag_scaler'))
		if args.only in {'stoc',None}:
			stoc_model = modified_wavenet_hps.create_stoc_model()
			stoc_model.load_weights(os.path.join(stoc_model_dir,'stoc_model.h5'))
		stoc_scaler = joblib.load(os.path.join(stoc_model_dir,'stoc_scaler'))
		
		for midi_path, timing_path, hps_path in data_paths:
			timing_output = ntpath.join(args.output_dir,ntpath.basename(timing_path))
			if args.only in {'timing', None}:
				if not args.no_timing_model:
					print('generating',timing_output)
					timing_dt.generate(timing_model, midi_path, timing_output)
				else:
					from misc import generate_written_timings
					print('using written timings to',timing_output)
					generate_written_timings(midi_path, timing_output)

			if args.only in {'f0','freq','mag','stoc',None}:
				existing_data_path=None
				if args.only=='f0':
					if args.existing_data_dir:
						timing_output=ntpath.join(args.existing_data_dir,ntpath.basename(timing_path))
					elif args.existing_timing_dir:
						timing_output=ntpath.join(args.existing_timing_dir,ntpath.basename(timing_path))
				elif args.only != None:
					existing_data_path=ntpath.join(args.existing_data_dir,ntpath.basename(hps_path))
					timing_output=ntpath.join(args.existing_timing_dir,ntpath.basename(timing_path))
				hps_output = ntpath.join(args.output_dir,ntpath.basename(hps_path))
				print('generating',hps_output)
				modified_wavenet_hps.generate(f0_model,freq_model,mag_model,stoc_model,
					mag_scaler, stoc_scaler,
					midi_path,timing_output,None,hps_output,args.only,existing_data_path)