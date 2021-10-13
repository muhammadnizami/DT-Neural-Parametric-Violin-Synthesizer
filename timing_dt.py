import timing_features
import time
from sklearn.tree import DecisionTreeRegressor
from misc import *
import progressbar

import configparser

config = configparser.ConfigParser()
config.read('config.ini')

def parsemodelconf(confdict):
	return {key: (float(confdict[key]) if key in {'corruption', 'spike_prob', 'spike_width',
		'gamma_u', 'gamma_s', 'gamma_w', 'epsilon', 'lr', 'decay'} else 
		[int(factor) for factor in confdict[key].split(',') if factor != ''] if key in {'dilation_factors','hidden_neurons'} else
		[float(factor) for factor in confdict[key].split(',')] if key in {'temperature','dropout_rates'} else
		confdict[key] if key in {'output_dist'} else
		int(confdict[key])
		) for key in confdict }
#for experiment
def change_timing_config(confdict):
	global timing_config
	newconf = parsemodelconf(confdict)
	for key in newconf:
		timing_config[key]=newconf[key]
	parse_timing_config()

def parse_timing_config():
	global num_prev_context, num_next_context, max_depth, input_dim
	num_prev_context=timing_config['num_prev_context']
	num_next_context=timing_config['num_next_context']
	max_depth=timing_config['max_depth']
	input_dim=timing_features.all_feature_dim(num_prev_context,num_next_context)

timing_config=parsemodelconf(config['TimingModel'])
parse_timing_config()

def create_timing_model():
	return DecisionTreeRegressor(max_depth=max_depth)

def preprocess_timing_data(data_paths):
	timingdevslist = []
	featureslist = []
	for sheet_path, timing_path, _ in data_paths:
		timings = read_timings(timing_path)
		actdurs = timings[1:]-timings[:-1]
		notes_and_rests = musicXML_to_notes_and_rests(sheet_path)
		writdur = np.array([n.end-n.start for n in notes_and_rests])
		timingdevs = np.expand_dims(actdurs - writdur,axis=1)
		timingdevslist.append(timingdevs)
		features = timing_features.dataset_features(sheet_path, timingdevs,
			num_prev_context, num_next_context)
		featureslist.append(features)

	return np.concatenate(featureslist), np.concatenate(timingdevslist)

def train_timing_model(data_paths):
	model = create_timing_model()
	print('training timing model')
	start = time.time()
	X, Y = preprocess_timing_data(data_paths)
	end = time.time()
	print('preprocess',end-start)
	start = time.time()
	model.fit(np.array(X),Y)
	end = time.time()
	print('training',end-start)
	return model

import matplotlib.pyplot as plt
def generate(timing_model, sheet_path, outputFile):
	sheet_features = timing_features.sheet_features(sheet_path,num_prev_context,num_next_context)
	model_outputs = np.zeros((num_prev_context+len(sheet_features),1))
	for i in progressbar.progressbar(range(len(sheet_features))):
		prev_outputs = np.concatenate(model_outputs[i:i+num_prev_context])
		features = np.concatenate([sheet_features[i],prev_outputs])
		current_output = timing_model.predict(np.expand_dims(features,axis=0))[0]
		model_outputs[i+num_prev_context]=current_output
	generated_timing_devs = model_outputs[num_prev_context:]
	
	notes_and_rests = musicXML_to_notes_and_rests(sheet_path)
	writdurs = [n.end-n.start for n in notes_and_rests]
	actdurs = writdurs + generated_timing_devs[:,0]
	actdurs[actdurs<0]=0

	timings = [0.]
	for i in range(len(actdurs)):
		timings.append(timings[-1]+actdurs[i])

	np.savetxt(outputFile,np.expand_dims(timings,axis=1),fmt='%.10f')