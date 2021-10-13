import configparser
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import modified_wavenet
import control_features
import time
import os
from misc import *

config = configparser.ConfigParser()
config.read('config.ini')

def parsemodelconf(confdict):
	return {key: (float(confdict[key]) if key in {'corruption', 'spike_prob', 'spike_width',
		'gamma_u', 'gamma_s', 'gamma_w', 'epsilon', 'lr', 'decay', 'min_delta'} else 
		[int(factor) for factor in confdict[key].split(',')] if key=='dilation_factors' else
		[float(factor) for factor in confdict[key].split(',')] if key=='temperature' else
		confdict[key] if key in {'output_dist','monitor','generation_algorithm'} else
		int(confdict[key])
		) for key in confdict }

#for experiment
def change_f0_config(confdict):
	global f0_config
	newconf = parsemodelconf(confdict)
	for key in newconf:
		f0_config[key]=newconf[key]
def change_freq_config(confdict):
	global freq_config
	newconf = parsemodelconf(confdict)
	for key in newconf:
		freq_config[key]=newconf[key]
def change_mag_config(confdict):
	global mag_config
	newconf = parsemodelconf(confdict)
	for key in newconf:
		mag_config[key]=newconf[key]
def change_stoc_config(confdict):
	global stoc_config
	newconf = parsemodelconf(confdict)
	for key in newconf:
		stoc_config[key]=newconf[key]

earlystopping_config_keys = ['patience','min_delta','monitor','restore_best_weights']
fit_config_keys = ['epochs','batch_size','validation_split']
compile_config_keys = ['lr','delay','optimizer','loss','loss_weights','sample_weight_mode','weighted_metrics','target_tensors']
def get_config_for(bigdict,keys):
	return dict([(k, bigdict[k]) for k in keys if k in bigdict])


audio_config = config['Audio']
audio_mag_t = float(config['Audio']['t'])

f0_config=parsemodelconf(config['F0Model'])
f0_config['additional_input_dim']=0

def preprocess_f0_data(model, hps, sheet_path, timing_path):
	f0s = np.array([freq_to_m(x[0][:1]) for x in hps])
	written_note_pitches = control_features.written_note_pitches(sheet_path,timing_path,len(f0s))
	f0s =  m_to_deviation(f0s,written_note_pitches)
	controlfeatures = control_features.control_features(sheet_path,timing_path,f0s.shape[0])
	
	r, c, o = model.window_data(f0s, None, controlfeatures)
	return r,c,o

def create_f0_model():
	return modified_wavenet.ModifiedWavenet(**f0_config,control_input_dim=control_features.control_feature_dim)

def train_f0_model(data_paths):
	epochs=f0_config['epochs']
	model = create_f0_model()
	print('training f0 model')
	X, Xcontrols, Y = [], [], []
	start = time.time()
	for sheet_path, timing_path, hps_path in data_paths:
		hps = np.load(hps_path)
		X_, Xcontrols_, Y_ = preprocess_f0_data(model, hps, sheet_path, timing_path)
		X.extend(X_)
		Xcontrols.extend(Xcontrols_)
		Y.extend(Y_)
	X = np.array(X)
	Xcontrols = np.array(Xcontrols)
	Y = np.array(Y)
	end = time.time()
	print('preprocess',end-start)
	start = time.time()
	model.compile(**get_config_for(f0_config,compile_config_keys))
	if epochs>0:
		model.fit([np.array(X),np.array(Xcontrols)],Y,
			callbacks=[EarlyStopping(**get_config_for(f0_config,earlystopping_config_keys))],
			**get_config_for(f0_config,fit_config_keys))
	end = time.time()
	print('training',end-start)
	return model

freq_config=parsemodelconf(config['FreqModel'])
freq_config['additional_input_dim']=f0_config['feature_dim']

def preprocess_freq_data(model, hps, sheet_path, timing_path):
	f0s = np.array([freq_to_m(x[0][:1]) for x in hps])
	written_note_pitches = control_features.written_note_pitches(sheet_path,timing_path,len(f0s))
	f0s_dev = m_to_deviation(f0s,written_note_pitches)
	freqs = np.array([freq_to_m(x[0]) for x in hps])
	freqs = m_to_deviation(freqs,f0s[:,0])
	controlfeatures = control_features.control_features(sheet_path,timing_path,freqs.shape[0])
	
	r, c, o = model.window_data(freqs, f0s_dev, controlfeatures)
	return r,c,o

def create_freq_model():
	return modified_wavenet.ModifiedWavenet(**freq_config,control_input_dim=control_features.control_feature_dim)

def train_freq_model(data_paths):
	epochs=freq_config['epochs']
	model = create_freq_model()
	print('training freq model')
	X, Xcontrols, Y = [], [], []
	start = time.time()
	for sheet_path, timing_path, hps_path in data_paths:
		hps = np.load(hps_path)
		X_, Xcontrols_, Y_ = preprocess_freq_data(model, hps, sheet_path, timing_path)
		X.extend(X_)
		Xcontrols.extend(Xcontrols_)
		Y.extend(Y_)
	X = np.array(X)
	Xcontrols = np.array(Xcontrols)
	Y = np.array(Y)
	end = time.time()
	print('preprocess',end-start)
	start = time.time()
	model.compile(**get_config_for(freq_config,compile_config_keys))
	if epochs>0:
		model.fit([np.array(X),np.array(Xcontrols)],Y,
			callbacks=[EarlyStopping(**get_config_for(freq_config,earlystopping_config_keys))],
			**get_config_for(freq_config,fit_config_keys))
	end = time.time()
	print('training',end-start)
	return model

mag_config=parsemodelconf(config['MagModel'])
mag_config['additional_input_dim']=f0_config['feature_dim']+freq_config['feature_dim']

def preprocess_mag_data(model, data_paths):
	magslist = []
	additionalinputslist = []
	controlfeatureslist = []
	for sheet_path, timing_path, hps_path in data_paths:
		hps = np.load(hps_path)
		mags = np.array([x[1] for x in hps])

		f0s = np.array([freq_to_m(x[0][:1]) for x in hps])
		written_note_pitches = control_features.written_note_pitches(sheet_path,timing_path,len(f0s))
		f0s_dev = m_to_deviation(f0s,written_note_pitches)
		freqs = np.array([freq_to_m(x[0]) for x in hps])
		freqs = m_to_deviation(freqs,f0s[:,0])
		controlfeatures = control_features.control_features(sheet_path,timing_path,mags.shape[0])

		magslist.append(mags)
		additionalinputslist.append(np.concatenate([f0s_dev,freqs],axis=-1))
		controlfeatureslist.append(controlfeatures)

	scaler = MinMaxScaler((0,1))
	scaler.fit(np.concatenate(magslist))

	allrs, allcs, allos = [], [], []
	for i in range(len(magslist)):
		mags = scaler.transform(magslist[i])
		additionalinputs = additionalinputslist[i]
		controlfeatures = controlfeatureslist[i]		
		r, c, o = model.window_data(mags, additionalinputs, controlfeatures)
		allrs.append(r)
		allcs.append(c)
		allos.append(o)
	return np.concatenate(allrs),np.concatenate(allcs),np.concatenate(allos),scaler

def create_mag_model():
	return modified_wavenet.ModifiedWavenet(control_input_dim=control_features.control_feature_dim, **mag_config)

def train_mag_model(data_paths):
	epochs=mag_config['epochs']
	print('training mag model')
	model = create_mag_model()
	start = time.time()
	X, Xcontrols, Y, scaler = preprocess_mag_data(model, data_paths)
	end = time.time()
	print('preprocess',end-start)
	start = time.time()
	model.compile(**get_config_for(mag_config,compile_config_keys))
	if epochs>0:
		model.fit([np.array(X),np.array(Xcontrols)],Y,
			callbacks=[EarlyStopping(**get_config_for(mag_config,earlystopping_config_keys))],
			**get_config_for(mag_config,fit_config_keys))
	end = time.time()
	print('training',end-start)
	return model, scaler

stoc_config=parsemodelconf(config['StocModel'])
stoc_config['additional_input_dim']=f0_config['feature_dim']+freq_config['feature_dim']+mag_config['feature_dim']

def preprocess_stoc_data(model, data_paths, mag_scaler):
	stocslist = []
	additionalinputslist = []
	controlfeatureslist = []
	for sheet_path, timing_path, hps_path in data_paths:
		hps = np.load(hps_path)
		stocs = np.array([x[2] for x in hps])

		f0s = np.array([freq_to_m(x[0][:1]) for x in hps])
		written_note_pitches = control_features.written_note_pitches(sheet_path,timing_path,len(f0s))
		f0s_dev = m_to_deviation(f0s,written_note_pitches)
		freqs = np.array([freq_to_m(x[0]) for x in hps])
		mags = np.array([(x[1]-audio_mag_t)/audio_mag_t for x in hps])
		freqs = m_to_deviation(freqs,f0s[:,0])
		scaled_mags = mag_scaler.transform(mags)

		controlfeatures = control_features.control_features(sheet_path,timing_path,stocs.shape[0])

		stocslist.append(stocs)
		additionalinputslist.append(np.concatenate([f0s_dev,freqs,scaled_mags],axis=-1))
		controlfeatureslist.append(controlfeatures)

	scaler = MinMaxScaler((0,1))
	scaler.fit(np.concatenate(stocslist))

	allrs, allcs, allos = [], [], []
	for i in range(len(stocslist)):
		stocs = scaler.transform(stocslist[i])
		additionalinputs = additionalinputslist[i]
		controlfeatures = controlfeatureslist[i]		
		r, c, o = model.window_data(stocs, additionalinputs, controlfeatures)
		allrs.append(r)
		allcs.append(c)
		allos.append(o)
	return np.concatenate(allrs),np.concatenate(allcs),np.concatenate(allos),scaler

def create_stoc_model():
	return modified_wavenet.ModifiedWavenet(control_input_dim=control_features.control_feature_dim, **stoc_config)

def train_stoc_model(data_paths, mag_scaler):
	epochs=stoc_config['epochs']
	model = create_stoc_model()
	X, Xcontrols, Y = [], [], []
	start = time.time()
	X, Xcontrols, Y, scaler = preprocess_stoc_data(model, data_paths, mag_scaler)
	Xcontrols = np.array(Xcontrols)
	Y = np.array(Y)
	end = time.time()
	print('preprocess',end-start)
	start = time.time()
	model.compile(**get_config_for(stoc_config,compile_config_keys))
	if epochs>0:
		model.fit([np.array(X),np.array(Xcontrols)],Y,
			callbacks=[EarlyStopping(**get_config_for(stoc_config,earlystopping_config_keys))],
			**get_config_for(stoc_config,fit_config_keys))
	end = time.time()
	print('training',end-start)
	return model, scaler
	
def predict_f0(model, prev_outs, controlfeatures):
	X = np.expand_dims(prev_outs,axis=0)
	Y = model.predict([X,np.array([controlfeatures])])
	return Y

def predict_freq(model, prev_outs, controlfeatures):
	X = np.expand_dims(prev_outs,axis=0)
	Y = model.predict([X,np.array([controlfeatures])])
	return Y

def predict_mag(model, prev_outs, controlfeatures):
	X = np.expand_dims(prev_outs,axis=0)
	Y = model.predict([X,np.array([controlfeatures])])
	return Y

def predict_stoc(model, prev_outs, controlfeatures):
	X = np.expand_dims(prev_outs,axis=0)
	Y = model.predict([X,np.array([controlfeatures])])
	return Y

# def create_post_scaler(pre_scaler,)

def generate(f0_model,freq_model, mag_model, stoc_model, mag_scaler, stoc_scaler, sheet_path, timing_path, num, outputFile, only=None, existing_data_path=None):

	if existing_data_path:
		existing_hps = np.load(existing_data_path)
		num=len(existing_hps)
	
	written_note_pitches = control_features.written_note_pitches(sheet_path,timing_path,num)
	num=len(written_note_pitches)

	if only in {'f0',None}:
		print('generating f0s...')
		f0_control_features = control_features.control_features(sheet_path, timing_path, num)
		f0_model_outputs = f0_model.generate(None,f0_control_features)
	elif only in {'freq','mag','stoc'}:
		f0_model_outputs = m_to_deviation(np.array([freq_to_m(x[0][:1]) for x in existing_hps]),written_note_pitches)
	else:
		f0_model_outputs = np.zeros((num,f0_config['feature_dim']))

	if only in {'freq',None}:
		print('generating freqs...')
		freq_control_features = control_features.control_features(sheet_path, timing_path, num)
		freq_model_outputs = freq_model.generate(f0_model_outputs,freq_control_features)
	elif only in {'mag','stoc'}:
		f0s = np.array([freq_to_m(x[0][:1]) for x in existing_hps])
		freqs = np.array([freq_to_m(x[0]) for x in existing_hps])
		freq_model_outputs = m_to_deviation(freqs,f0s[:,0])
	else:
		freq_model_outputs = np.zeros((num,freq_config['feature_dim']))

	if only in {'mag',None}:
		print('generating mags...')
		mag_control_features = control_features.control_features(sheet_path, timing_path, num)
		mag_model_outputs = mag_model.generate(np.concatenate([f0_model_outputs,freq_model_outputs],axis=-1),mag_control_features)
	elif only == 'stoc':
		mags = np.array([x[1] for x in existing_hps])
		mag_model_outputs = mag_scaler.transform(mags)
	else:
		mag_model_outputs = np.zeros((num,mag_config['feature_dim']))

	if only in {'stoc',None}:
		stoc_control_features = control_features.control_features(sheet_path, timing_path, num)
		stoc_model_outputs = stoc_model.generate(np.concatenate([f0_model_outputs,freq_model_outputs,mag_model_outputs],axis=-1),stoc_control_features)
	elif existing_data_path:
		stocs = np.array([x[2] for x in existing_hps])
		stoc_model_outputs = stoc_scaler.transform(stocs)
	else:
		stoc_model_outputs = np.zeros((num,stoc_config['feature_dim']))

	mag_model_outputs = mag_scaler.inverse_transform(mag_model_outputs)
	stoc_model_outputs = stoc_scaler.inverse_transform(stoc_model_outputs)

	outputs = np.array(list(zip(*[m_to_freq(deviation_to_m(freq_model_outputs,deviation_to_m(f0_model_outputs,written_note_pitches)[:,0])),mag_model_outputs,stoc_model_outputs])))
	print('saving...')
	np.save(outputFile,outputs)