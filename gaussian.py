from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import *
import tensorflow as tf
import math

ExpandDim = Lambda(lambda x: K.expand_dims(x, axis=1))

def GaussianRandomPredictor(layer_input, feature_dim, **kwargs):
	trainout = GaussianTrainOut(layer_input, feature_dim)
	return Lambda(lambda x: K.map_fn(lambda x: x[1] + tf.random_normal(x[1].shape,0,1)*K.abs(x[0]),x))(trainout)

def GaussianMeanPredictor(layer_input, feature_dim, **kwargs):
	trainout = GaussianTrainOut(layer_input, feature_dim)
	return Lambda(lambda x: x[:,1])(trainout)

def GaussianTrainOut(layer_input, feature_dim, **kwargs):
	outputs = []
	for i in range(2):
		outputs.append(ExpandDim(Conv1D(filters=feature_dim,kernel_size=1)(layer_input)))
	return concatenate(outputs, axis=1)

def GaussianParams(a, epsilon=1e-07):
	sigma = K.abs(a[:,0])
	mu = a[:,1]
	
	return sigma, mu

def GaussianLikelihood(x, sigma, mu, epsilon=1e-07):
	n = tf.to_float(K.shape(x)[0])

	return (2*math.pi*sigma**2)**(-n/2)*K.exp(-1/(2*sigma**2)*K.sum((x-mu)**2,axis=0))

def GaussianLogLikelihood(x, sigma, mu, epsilon=1e-07):
	n = tf.to_float(K.shape(x)[0])
	return -n/2*K.log(2*math.pi)-(n/2)*K.log(sigma**2+epsilon)-1/(2*sigma**2+epsilon)*K.sum((x-mu)**2,axis=0)

def GaussianLogLikelihoodWithoutSigma(x, sigma, mu, epsilon=1e-07):
	n = tf.to_float(K.shape(x)[0])
	return -K.sum((x-mu)**2,axis=0)-K.sum((0-sigma)**2,axis=0)

def GaussianLossWithoutSigma(epsilon=1e-07):
	def f(yTrue, yPred):
		x = yTrue[:,0]
		sigma, mu = GaussianParams(yPred,epsilon)
		return -GaussianLogLikelihoodWithoutSigma(x, sigma, mu,epsilon)
	return f
	
def GaussianLoss(epsilon=1e-07, **kwargs):
	def f(yTrue, yPred):
		x = yTrue[:,0]
		sigma, mu = GaussianParams(yPred,epsilon)
		return -GaussianLogLikelihood(x,sigma,mu,epsilon)
	return f
	
import numpy as np
def GaussianFit(model, x, y, **kwargs):
	plcholdr = np.zeros(y.shape)
	y_feed = np.concatenate([np.expand_dims(y,axis=1),np.expand_dims(plcholdr,axis=1)],axis=1)
	return model.fit(x, y_feed, **kwargs)