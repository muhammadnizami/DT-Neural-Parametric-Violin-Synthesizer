from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import *
import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as tfd
import tensorflow as tf
import math

ExpandDim = Lambda(lambda x: K.expand_dims(x, axis=1))

def CGMRandom(a, num_gaussians=4, gamma_u = 1.6, gamma_s = 1.1, gamma_w = 1/1.75, temperature=0.5, epsilon=1e-07, **kwargs):
	sigma, mu, w = CGMParams(a, num_gaussians, gamma_u, gamma_s, gamma_w, epsilon)
	sigma, mu, w = temperature_control(sigma, mu, w, temperature)
	w_flattened = tf.reshape(w,(-1,num_gaussians))
	chosen_mixtures = tf.reshape(tf.random.multinomial(w_flattened,1,output_dtype=tf.int32),(-1,))
	indices = tf.range(0,tf.shape(chosen_mixtures)[0],dtype=chosen_mixtures.dtype)
	gt_indices = tf.stack([chosen_mixtures,indices],axis=1)
	mu_flattened = tf.reshape(mu,(4,-1))
	chosen_mu = tf.reshape(tf.gather_nd(mu_flattened,gt_indices),tf.shape(mu)[1:])
	sigma_flattened = tf.reshape(sigma,(4,-1))
	chosen_sigma = tf.reshape(tf.gather_nd(sigma_flattened,gt_indices),tf.shape(sigma)[1:])
	output = chosen_mu + tf.random.normal(tf.shape(chosen_sigma))*chosen_sigma
	return output

def CGMRandomPredictor(layer_input, feature_dim, num_gaussians=4, gamma_u = 1.6, gamma_s = 1.1, gamma_w = 1/1.75, temperature=0.5, epsilon=1e-07, **kwargs):
	trainout = CGMTrainOut(layer_input, feature_dim)
	return Lambda(lambda x: CGMRandom(x,num_gaussians,gamma_u,gamma_s, gamma_w, temperature, epsilon))(trainout)

def CGMTrainOut(layer_input, feature_dim, **kwargs):
	outputs = []
	for i in range(4):
		outputs.append(ExpandDim(Conv1D(filters=feature_dim,kernel_size=1)(layer_input)))
	return concatenate(outputs, axis=1)

def abs(a,epsilon):
	return K.sqrt(K.square(a)+K.square(epsilon))
	
def CGMParams(a, num_gaussians=4, gamma_u = 1.6, gamma_s = 1.1, gamma_w = 1/1.75, epsilon=1e-07):
	Xi = 2 * K.sigmoid(a[:,0])-1
	Omega = 2/255 * K.exp((4*K.sigmoid(a[:,1])))
	alpha = 2 * K.sigmoid(a[:,2])-1
	beta = 2 * K.sigmoid(a[:,3])

	sigma = [None] * num_gaussians
	mu = [None] * num_gaussians
	w = [None] * num_gaussians
	for k in range(num_gaussians):
		sigma[k] = Omega * K.exp(((abs(alpha,epsilon))*gamma_s-1)*k)
		mu[k] = Xi + K.sum(sigma[:k])*gamma_u*alpha
		w[k] = ((alpha+epsilon) ** (2*k) * (beta+epsilon) ** k * gamma_w ** k)
	sigma = abs(K.concatenate([K.expand_dims(cmpt,axis=0) for cmpt in sigma],axis=0),epsilon)
	mu = K.concatenate([K.expand_dims(cmpt,axis=0) for cmpt in mu],axis=0)
	w = K.concatenate([K.expand_dims(cmpt,axis=-1) for cmpt in w],axis=-1)+epsilon
	w = w/K.expand_dims(K.sum(w,axis=-1),axis=-1)

	return sigma, mu, w

def temperature_control(sigma, mu, w, temperature):
	num_gaussians = sigma.shape[0]
	temperature = tf.constant(temperature)
	global_mu = K.sum([tf.gather(w,k,axis=-1)*mu[k] for k in range(num_gaussians)],axis=0)
	mu = mu + (global_mu-mu) * (1-temperature)
	sigma = sigma * K.sqrt(temperature)
	return sigma, mu, w

def CGMLoss(num_gaussians=4, gamma_u = 1.6, gamma_s = 1.1, gamma_w = 1/1.75, epsilon=1e-07, **kwargs):

	def _minusloglikelihood(x, sigma, mu, w):

		num_gaussians=sigma.shape[0]
		assert mu.shape[0]==num_gaussians
		assert w.shape[-1]==num_gaussians

		cat = tfd.Categorical(probs = w)
		dist = [None] * num_gaussians
		for k in range(num_gaussians):
			dist[k] = tfd.Normal(loc = mu[k], scale = sigma[k])
		mixdist = tfd.Mixture(cat = cat, components = dist)

		return -mixdist.log_prob(x)

	def f(yTrue, yPred):
		x = yTrue[:,0]
		sigma, mu, w = CGMParams(yPred,num_gaussians, gamma_u, gamma_s, gamma_w, epsilon)
		minusloglikelihood = _minusloglikelihood(x, sigma, mu, w)

		return K.mean(minusloglikelihood)
	return f

import numpy as np
def CGMFit(model, x, y, **kwargs):
	plcholdr = np.expand_dims(np.zeros(y.shape),axis=1)
	y_feed = np.concatenate([np.expand_dims(y,axis=1),plcholdr,plcholdr,plcholdr],axis=1)
	return model.fit(x, y_feed, **kwargs)