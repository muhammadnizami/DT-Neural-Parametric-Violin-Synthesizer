from tensorflow.python.keras.layers import *
from tensorflow.python.keras.activations import *
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import Adam
from noise import *
from collections import deque
import numpy as np
import gaussian
import CGM
import progressbar

class ModifiedWavenet:
    def __init__(self, feature_dim, additional_input_dim = 0,
    control_input_dim = 1, input_noise_level = 0.4, initial_causal_conv_size = 10,
    residual_channels = 130, dilated_conv_size = 2, dilation_factors = [1,2,4,1,2],
    skip_channels = 240, fragment_length = 1, corruption=0,  
    output_dist='linear', window_hop = None, generation_algorithm='naive', **params):

        #save all the configs
        self.feature_dim = feature_dim
        self.additional_input_dim = additional_input_dim
        self.control_input_dim = control_input_dim
        self.input_noise_level = input_noise_level
        self.initial_causal_conv_size = initial_causal_conv_size
        self.residual_channels = residual_channels
        self.dilated_conv_size = dilated_conv_size
        self.dilation_factors = dilation_factors
        self.skip_channels = skip_channels
        self.fragment_length = fragment_length
        self.corruption = corruption
        self.params = params
        self.output_dist = output_dist
        self.window_hop = window_hop
        self.generation_algorithm = generation_algorithm

        #build the model
        self.training_model = self.build_training_model()
        if self.generation_algorithm=='naive':
            self.naive_generation_model = self.build_naive_generation_model()
        else:
            self.initial_iteration_generation_model = self.build_initial_iteration_generation_model()
            self.iteration_generation_model = self.build_iteration_generation_model()

    def compile(self,**params):
        new_params = {**params}
        if "optimizer" not in new_params:
            if "optimizer" in self.params:
                new_params["optimizer"] = self.params["optimizer"]
            else:
                new_params["optimizer"] = Adam(**{key: params[key] for key in params if key in {'lr', 'decay'}})
                new_params.pop('lr', None)
                new_params.pop('decay', None)
        if self.output_dist.lower()=="gaussian":
            new_params['loss']=gaussian.GaussianLoss(**new_params)
        elif self.output_dist.lower()=="cgm":
            new_params['loss']=CGM.CGMLoss(**new_params)
        else:
            if "loss" not in new_params:
                if "loss" in params:
                    new_params["loss"]=params["loss"]
                else:
                    new_params["loss"] = "mean_squared_error"
        return self.training_model.compile(**new_params)

    def fit(self,X,Y,**params):
        if self.output_dist.lower()=="gaussian":
            a = gaussian.GaussianFit(self.training_model,X,Y,**params)
        elif self.output_dist.lower()=="cgm":
            a = CGM.CGMFit(self.training_model,X,Y,**params)
        else:
            a = self.training_model.fit(X,Y,**params)
        if self.generation_algorithm=='naive':
            self.naive_generation_model.set_weights(self.training_model.get_weights())
        else:
            self.initial_iteration_generation_model.set_weights(self.training_model.get_weights())
            self.iteration_generation_model.set_weights(self.training_model.get_weights())
        return a

    def calculate_input_length(self):
        return calculate_input_length(self.initial_causal_conv_size, self.dilation_factors, self.fragment_length)

    def build_training_model(self):
  
        input_length = calculate_input_length(self.initial_causal_conv_size,self.dilation_factors,self.fragment_length)
        acoustic_input = Input((input_length, self.feature_dim + self.additional_input_dim), 
            name='input_acoustic')

        gauss_acoustic_input = GaussianNoise(self.corruption)(acoustic_input)
        control_input = Input((input_length, self.control_input_dim),
            name='input_control')

        residual = Conv1D(filters=self.residual_channels,
            kernel_size=self.initial_causal_conv_size, dilation_rate=1, padding='causal',
            name='causalconv1d_initial')(gauss_acoustic_input)

        skips = []
        dilation_depth = len(self.dilation_factors)
        for i in range(dilation_depth):
            tanh_dilated_in = Conv1D(filters=self.residual_channels, kernel_size = self.dilated_conv_size,
                dilation_rate=self.dilation_factors[i], padding='causal',
                name='dilated_conv1d_tanh_{}'.format(i))(residual)
            sigm_dilated_in = Conv1D(filters=self.residual_channels, kernel_size = self.dilated_conv_size,
                dilation_rate=self.dilation_factors[i], padding='causal',
                name='dilated_conv1d_sigm_{}'.format(i))(residual)
            tanh_control_in = Conv1D(filters = self.residual_channels, kernel_size = 1, 
                name='control_conv1d_tanh_{}'.format(i))(control_input)
            sigm_control_in = Conv1D(filters = self.residual_channels, kernel_size = 1, 
                name='control_conv1d_sigm_{}'.format(i))(control_input)

            tanh_out = Activation('tanh')(add([tanh_dilated_in,tanh_control_in]))
            sigm_out = Activation('sigmoid')(add([sigm_dilated_in,sigm_control_in]))

            gated_out = multiply([tanh_out,sigm_out])

            residual_addition = Conv1D(filters=self.residual_channels, kernel_size = 1, 
                name='residual_addition_{}'.format(i))(gated_out)
            residual = add([residual,residual_addition])
            skips.append(Conv1D(filters=self.skip_channels, kernel_size=1,
                name='skip_{}'.format(i))(gated_out))

        skips.append(Conv1D(filters=self.skip_channels, kernel_size=1, name='skip_control')(control_input))

        valid_dilation_output = Lambda(lambda x: x[:,-self.fragment_length:,:],
            name='valid_dilation_output')(add(skips) if len(skips)>1 else skips[0] if len(skips)==1 else residual)
        before_before_output=Activation('tanh',name='before_final_tanh')(valid_dilation_output)
        if self.output_dist.lower()=='gaussian':
            output = gaussian.GaussianTrainOut(concatenate([valid_dilation_output,before_before_output]),self.feature_dim)
        elif self.output_dist.lower()=='cgm':
            output = CGM.CGMTrainOut(before_before_output,self.feature_dim,**self.params)
        else:
            before_output=Activation('tanh',name='final_tanh')(concatenate([valid_dilation_output,before_before_output]))
            output = Conv1D(filters=self.feature_dim,kernel_size=1,name='output')(concatenate([valid_dilation_output,before_before_output,before_output]))

        return Model([acoustic_input,control_input],output)

    def build_naive_generation_model(self):

        input_length = calculate_input_length(self.initial_causal_conv_size,self.dilation_factors,1)
        acoustic_input = Input((input_length, self.feature_dim + self.additional_input_dim), 
            name='input_acoustic')

        control_input = Input((input_length, self.control_input_dim),
            name='input_control')

        residual = Conv1D(filters=self.residual_channels,
            kernel_size=self.initial_causal_conv_size, dilation_rate=1, padding='causal',
            name='causalconv1d_initial')(acoustic_input)

        skips = []
        dilation_depth = len(self.dilation_factors)
        for i in range(dilation_depth):
            tanh_dilated_in = Conv1D(filters=self.residual_channels, kernel_size = self.dilated_conv_size,
                dilation_rate=self.dilation_factors[i], padding='causal',
                name='dilated_conv1d_tanh_{}'.format(i))(residual)
            sigm_dilated_in = Conv1D(filters=self.residual_channels, kernel_size = self.dilated_conv_size,
                dilation_rate=self.dilation_factors[i], padding='causal',
                name='dilated_conv1d_sigm_{}'.format(i))(residual)
            tanh_control_in = Conv1D(filters = self.residual_channels, kernel_size = 1, 
                name='control_conv1d_tanh_{}'.format(i))(control_input)
            sigm_control_in = Conv1D(filters = self.residual_channels, kernel_size = 1, 
                name='control_conv1d_sigm_{}'.format(i))(control_input)

            tanh_out = Activation('tanh')(add([tanh_dilated_in,tanh_control_in]))
            sigm_out = Activation('sigmoid')(add([sigm_dilated_in,sigm_control_in]))

            gated_out = multiply([tanh_out,sigm_out])

            residual_addition = Conv1D(filters=self.residual_channels, kernel_size = 1, 
                name='residual_addition_{}'.format(i))(gated_out)
            residual = add([residual,residual_addition])
            skips.append(Conv1D(filters=self.skip_channels, kernel_size=1,
                name='skip_{}'.format(i))(gated_out))

        skips.append(Conv1D(filters=self.skip_channels, kernel_size=1, name='skip_control')(control_input))

        valid_dilation_output = Lambda(lambda x: x[:,-1:,:],
            name='valid_dilation_output')(add(skips) if len(skips)>1 else skips[0] if len(skips)==1 else residual)
        before_before_output=Activation('tanh',name='before_final_tanh')(valid_dilation_output)
        if self.output_dist.lower()=='gaussian':
            output = gaussian.GaussianRandomPredictor(concatenate([valid_dilation_output,before_before_output]),self.feature_dim)
        elif self.output_dist.lower()=='cgm':
            output = CGM.CGMRandomPredictor(before_before_output,self.feature_dim,**self.params)
        else:
            before_output=Activation('tanh',name='final_tanh')(concatenate([valid_dilation_output,before_before_output]))
            output = Conv1D(filters=self.feature_dim,kernel_size=1,name='output')(concatenate([valid_dilation_output,before_before_output,before_output]))

        return Model([acoustic_input,control_input],output)

    def build_initial_iteration_generation_model(self):
  
        input_length = calculate_input_length(self.initial_causal_conv_size,self.dilation_factors,1)
        acoustic_input = Input((input_length, self.feature_dim + self.additional_input_dim), 
            name='input_acoustic')
        control_input = Input((input_length, self.control_input_dim),
            name='input_control')

        residual = Conv1D(filters=self.residual_channels,
            kernel_size=self.initial_causal_conv_size, dilation_rate=1, padding='causal',
            name='causalconv1d_initial')(acoustic_input)

        skips = []
        next_cached_residuals = []
        dilation_depth = len(self.dilation_factors)
        for i in range(dilation_depth):
            next_cached_residuals.append(Lambda(lambda x: x[:,-(self.dilated_conv_size-1)*self.dilation_factors[i]:,:])(residual))
            tanh_dilated_in = Conv1D(filters=self.residual_channels, kernel_size = self.dilated_conv_size,
                dilation_rate=self.dilation_factors[i], padding='causal',
                name='dilated_conv1d_tanh_{}'.format(i))(residual)
            sigm_dilated_in = Conv1D(filters=self.residual_channels, kernel_size = self.dilated_conv_size,
                dilation_rate=self.dilation_factors[i], padding='causal',
                name='dilated_conv1d_sigm_{}'.format(i))(residual)
            tanh_control_in = Conv1D(filters = self.residual_channels, kernel_size = 1, 
                name='control_conv1d_tanh_{}'.format(i))(control_input)
            sigm_control_in = Conv1D(filters = self.residual_channels, kernel_size = 1, 
                name='control_conv1d_sigm_{}'.format(i))(control_input)

            tanh_out = Activation('tanh')(add([tanh_dilated_in,tanh_control_in]))
            sigm_out = Activation('sigmoid')(add([sigm_dilated_in,sigm_control_in]))

            gated_out = multiply([tanh_out,sigm_out])

            residual_addition = Conv1D(filters=self.residual_channels, kernel_size = 1, 
                name='residual_addition_{}'.format(i))(gated_out)
            residual = add([residual,residual_addition])
            skips.append(Conv1D(filters=self.skip_channels, kernel_size=1,
                name='skip_{}'.format(i))(gated_out))

        skips.append(Conv1D(filters=self.skip_channels, kernel_size=1, name='skip_control')(control_input))

        valid_dilation_output = Lambda(lambda x: x[:,-1:,:],
            name='valid_dilation_output')(add(skips) if len(skips)>1 else skips[0] if len(skips)==1 else residual)
        before_before_output=Activation('tanh',name='before_final_tanh')(valid_dilation_output)
        if self.output_dist.lower()=='gaussian':
            output = gaussian.GaussianRandomPredictor(concatenate([valid_dilation_output,before_before_output]),self.feature_dim)
        elif self.output_dist.lower()=='cgm':
            output = CGM.CGMRandomPredictor(before_before_output,self.feature_dim,**self.params)
        else:
            before_output=Activation('tanh',name='final_tanh')(concatenate([valid_dilation_output,before_before_output]))
            output = Conv1D(filters=self.feature_dim,kernel_size=1,name='output')(concatenate([valid_dilation_output,before_before_output,before_output]))

        return Model([acoustic_input,control_input],next_cached_residuals+[output])

    def build_iteration_generation_model(self):
        input_length = calculate_input_length(self.initial_causal_conv_size,self.dilation_factors,1)

        acoustic_input = Input((self.initial_causal_conv_size, self.feature_dim + self.additional_input_dim), 
            name='input_acoustic')
        control_input = Input((1, self.control_input_dim),
            name='input_control')

        residual = Conv1D(filters=self.residual_channels,
            kernel_size=self.initial_causal_conv_size, dilation_rate=1, padding='valid',
            name='causalconv1d_initial')(acoustic_input)

        skips = []
        prev_cached_residuals = []
        next_cached_residuals = []
        dilation_depth = len(self.dilation_factors)
        for i in range(dilation_depth):
            prev_cached_residual = Input(((self.dilated_conv_size-1)*1,self.residual_channels))
            residual = concatenate([prev_cached_residual, Lambda(lambda x: x[:,-1:,:])(residual)],axis=1)
            prev_cached_residuals.append(prev_cached_residual)
            next_cached_residuals.append(Lambda(lambda x: x[:,-(self.dilated_conv_size-1)*1:,:])(residual))
            tanh_dilated_in = Conv1D(filters=self.residual_channels, kernel_size = self.dilated_conv_size,
                dilation_rate=1, padding='valid',
                name='dilated_conv1d_tanh_{}'.format(i))(residual)
            sigm_dilated_in = Conv1D(filters=self.residual_channels, kernel_size = self.dilated_conv_size,
                dilation_rate=1, padding='valid',
                name='dilated_conv1d_sigm_{}'.format(i))(residual)
            tanh_control_in = Conv1D(filters = self.residual_channels, kernel_size = 1, 
            name='control_conv1d_tanh_{}'.format(i))(control_input)
            sigm_control_in = Conv1D(filters = self.residual_channels, kernel_size = 1, 
                name='control_conv1d_sigm_{}'.format(i))(control_input)

            tanh_out = Activation('tanh')(add([tanh_dilated_in,tanh_control_in]))
            sigm_out = Activation('sigmoid')(add([sigm_dilated_in,sigm_control_in]))

            gated_out = multiply([tanh_out,sigm_out])

            residual_addition = Conv1D(filters=self.residual_channels, kernel_size = 1, 
                name='residual_addition_{}'.format(i))(gated_out)
            residual = add([Lambda(lambda x: x[:,-1:,:])(residual),residual_addition])
            skips.append(Conv1D(filters=self.skip_channels, kernel_size=1,
                name='skip_{}'.format(i))(gated_out))

        skips.append(Conv1D(filters=self.skip_channels, kernel_size=1, name='skip_control')(control_input))

        if len(skips)>1:
            valid_dilation_output = add([Lambda(lambda x: x[:,-1:,:])(skip) for skip in skips])
        else:
            valid_dilation_output = Lambda(lambda x: x[:,-1:,:],
                name='valid_dilation_output')(skips[0] if len(skips)==1 else residual)
        before_before_output=Activation('tanh',name='before_final_tanh')(valid_dilation_output)
        if self.output_dist.lower()=='gaussian':
            output = gaussian.GaussianRandomPredictor(concatenate([valid_dilation_output,before_before_output]),self.feature_dim)
        elif self.output_dist.lower()=='cgm':
            output = CGM.CGMRandomPredictor(before_before_output,self.feature_dim,**self.params)
        else:
            before_output=Activation('tanh',name='final_tanh')(concatenate([valid_dilation_output,before_before_output]))
            output = Conv1D(filters=self.feature_dim,kernel_size=1,name='output')(concatenate([valid_dilation_output,before_before_output,before_output]))

        return Model([acoustic_input,control_input]+prev_cached_residuals,next_cached_residuals+[output])


    def load_weights(self,filepath, **params):
        a = self.training_model.load_weights(filepath, **params)
        if self.generation_algorithm=='naive':
            self.naive_generation_model.set_weights(self.training_model.get_weights())
        else:
            self.initial_iteration_generation_model.set_weights(self.training_model.get_weights())
            self.iteration_generation_model.set_weights(self.training_model.get_weights())
        return a

    def save_weights(self, filepath, **params):
        return self.training_model.save_weights(filepath,**params)

    def get_weights(self, **params):
        return self.training_model.get_weights(**params)

    def set_weights(self, a, **params):
        a = self.training_model.set_weights(a, **params)
        if self.generation_algorithm=='naive':
            self.naive_generation_model.set_weights(self.training_model.get_weights())
        else:
            self.initial_iteration_generation_model.set_weights(self.training_model.get_weights())
            self.iteration_generation_model.set_weights(self.training_model.get_weights())
        return a

    def generate_cached(self,additional_input, control_inputs):
  
        # TODO check bug: additional_input, residual_channels and skip_channels
        if additional_input is None:
            additional_input_shape = list(control_inputs.shape)
            additional_input_shape[-1]=0
            additional_input = np.zeros(additional_input_shape)

        num_of_prev_initial = calculate_input_length(self.initial_causal_conv_size,self.dilation_factors,1)

        initial_control_inputs = np.pad(control_inputs[:1],((num_of_prev_initial-1,0),(0,0)),'constant')
        acoustic_inputs = np.zeros((self.initial_iteration_generation_model.input_shape[0][1] + control_inputs.shape[0],self.feature_dim))
        additional_input = np.pad(additional_input,((num_of_prev_initial-1,0),(0,0)),'constant')

        initial_out = self.initial_iteration_generation_model.predict([[
            np.concatenate([acoustic_inputs[:num_of_prev_initial],additional_input[:num_of_prev_initial]],axis=-1)],
            [initial_control_inputs]])
        cached_residuals = []
        for i in range(len(initial_out)-1):
            cached_residuals.append(deque(initial_out[i][0]))
        acoustic_inputs[num_of_prev_initial] = initial_out[-1][0]
        
        num_of_prev_iterative = self.iteration_generation_model.input_shape[0][1]
        for i in progressbar.progressbar(range(1,len(control_inputs))):
            cur_cached_residuals = [ None ] * len(cached_residuals)
            for j in range(len(cached_residuals)):
                cur_cached_residuals[j] = np.array([[cached_residuals[j].popleft()]])
            out = self.iteration_generation_model.predict(
                [[np.concatenate([acoustic_inputs[i+num_of_prev_initial-num_of_prev_iterative:i+num_of_prev_initial], 
                    additional_input[i+num_of_prev_initial-num_of_prev_iterative:i+num_of_prev_initial]],
                    axis=-1)],
                [[control_inputs[i]]]] + cur_cached_residuals)
            for j in range(len(out)-1):
                cached_residuals[j].append(out[j][0][0])
            acoustic_inputs[num_of_prev_initial+i]=out[-1][0]

        return acoustic_inputs[num_of_prev_initial:]

    def generate_naive(self, additional_input, control_features):
        if additional_input is None:
            additional_input_shape = list(control_features.shape)
            additional_input_shape[-1]=0
            additional_input = np.zeros(additional_input_shape)

        num_of_prev = calculate_input_length(self.initial_causal_conv_size,self.dilation_factors,1)
        num = len(control_features)
        model_outputs = np.zeros((num_of_prev+num, self.naive_generation_model.output_shape[2]))
        control_features = np.pad(control_features,((num_of_prev-1,0),(0,0)),'constant')
        additional_input = np.pad(additional_input,((num_of_prev-1,0),(0,0)),'constant')
        for i in progressbar.progressbar(range(num)):
            prev_outs = model_outputs[i:i+num_of_prev]
            prev_additional = additional_input[i:i+num_of_prev]
            controlfeatures = control_features[i:i+num_of_prev]
            X = np.expand_dims(np.concatenate([prev_outs,prev_additional],axis=-1),axis=0)
            unselectedout = self.naive_generation_model.predict([X,np.array([controlfeatures])])
            newout = unselectedout[0][-1:]
            model_outputs[i+num_of_prev] = newout
        model_outputs = model_outputs[num_of_prev:]
        return model_outputs

    def generate(self, additional_input, control_features):
        if self.generation_algorithm=='naive':
            return self.generate_naive(additional_input, control_features)
        else:
            return self.generate_cached(additional_input, control_features)

    def window_data(self, features, additional_input, control_features):
        return window_data(features, additional_input, control_features, self.initial_causal_conv_size, self.dilation_factors, self.fragment_length, self.window_hop, **self.params)

#TODO add dilated_conv_size to calculate_input_length
def calculate_input_length(initial_causal_conv_size = 10, dilation_factors= [1,2,4,1,2], fragment_length=1, **params):
    return sum(dilation_factors) + initial_causal_conv_size + (fragment_length - 1)

def build_modified_wavenet_model(feature_dim, additional_input_dim = 0,
    control_input_dim = 1, input_noise_level = 0.4, initial_causal_conv_size = 10,
    residual_channels = 130, dilated_conv_size = 2, dilation_factors = [1,2,4,1,2],
    skip_channels = 240, fragment_length = 1, corruption=0,  **params):
  
    input_length = calculate_input_length(initial_causal_conv_size,dilation_factors,fragment_length)
    acoustic_input = Input((input_length, feature_dim + additional_input_dim), 
        name='input_acoustic')

    gauss_acoustic_input = GaussianNoise(corruption)(acoustic_input)
    control_input = Input((input_length, control_input_dim),
        name='input_control')

    residual = Conv1D(filters=residual_channels,
        kernel_size=initial_causal_conv_size, dilation_rate=1, padding='causal',
        name='causalconv1d_initial')(gauss_acoustic_input)

    skips = []
    dilation_depth = len(dilation_factors)
    for i in range(dilation_depth):
        tanh_dilated_in = Conv1D(filters=residual_channels, kernel_size = dilated_conv_size,
            dilation_rate=dilation_factors[i], padding='causal',
            name='dilated_conv1d_tanh_{}'.format(i))(residual)
        sigm_dilated_in = Conv1D(filters=residual_channels, kernel_size = dilated_conv_size,
            dilation_rate=dilation_factors[i], padding='causal',
            name='dilated_conv1d_sigm_{}'.format(i))(residual)
        tanh_control_in = Conv1D(filters = residual_channels, kernel_size = 1, 
            name='control_conv1d_tanh_{}'.format(i))(control_input)
        sigm_control_in = Conv1D(filters = residual_channels, kernel_size = 1, 
            name='control_conv1d_sigm_{}'.format(i))(control_input)

        tanh_out = Activation('tanh')(add([tanh_dilated_in,tanh_control_in]))
        sigm_out = Activation('sigmoid')(add([sigm_dilated_in,sigm_control_in]))

        gated_out = multiply([tanh_out,sigm_out])

        residual_addition = Conv1D(filters=residual_channels, kernel_size = 1, 
            name='residual_addition_{}'.format(i))(gated_out)
        residual = add([residual,residual_addition])
        skips.append(Conv1D(filters=skip_channels, kernel_size=1,
            name='skip_{}'.format(i))(gated_out))

    skips.append(Conv1D(filters=skip_channels, kernel_size=1, name='skip_control')(control_input))

    valid_dilation_output = Lambda(lambda x: x[:,-fragment_length:,:],
        name='valid_dilation_output')(add(skips) if len(skips)>1 else skips[0] if len(skips)==1 else residual)
    before_before_output=Activation('tanh',name='before_final_tanh')(valid_dilation_output)
    before_output=Activation('tanh',name='final_tanh')(concatenate([valid_dilation_output,before_before_output]))
    output = Conv1D(filters=feature_dim,kernel_size=1,name='output')(concatenate([valid_dilation_output,before_before_output,before_output]))

    return Model([acoustic_input,control_input],output)

#length of control_features must be equal to features
#TODO add dilated_conv_size to calculate_input_length
sentinel = object()
def window_data(features, additional_input, control_features, initial_causal_conv_size = 10, dilation_factors= [1,2,4,1,2], fragment_length=1, window_hop=sentinel, **params):
    if additional_input is None:
        additional_input_shape = list(control_features.shape)
        additional_input_shape[-1]=0
        additional_input = np.zeros(additional_input_shape)
    
    if window_hop is sentinel:
        window_hop=fragment_length

    num_of_prev = calculate_input_length(initial_causal_conv_size,dilation_factors,fragment_length)

    input_features = np.concatenate([features, additional_input],axis=-1)
    features_left_pad = num_of_prev-fragment_length+1
    features_right_pad = window_hop-((len(features)-fragment_length)%window_hop)
    padded_features = np.pad(features,[(features_left_pad,features_right_pad),(0,0)],'constant')
    padded_input_features = np.pad(input_features,[(features_left_pad,features_right_pad),(0,0)],'constant')
    padded_control_features = np.pad(control_features,[(features_left_pad,features_right_pad),(0,0)],'constant')
    windowed_regression_features = np.array([padded_input_features[i-num_of_prev:i] for i in range(num_of_prev,len(padded_input_features)-1,window_hop)])
    windowed_output_fragments = np.array([padded_features[i-fragment_length:i] for i in range(num_of_prev+1,len(padded_features),window_hop)])
    windowed_control_features = np.array([padded_control_features[i-num_of_prev:i] for i in range(num_of_prev+1,len(padded_features),window_hop)])
    return windowed_regression_features, windowed_control_features, windowed_output_fragments

def predict(model, additional_input, control_features, num):
    if additional_input is None:
        additional_input_shape = list(control_features.shape)
        additional_input_shape[-1]=0
        additional_input = np.zeros(additional_input_shape)

    if len(control_features) != num:
        raise ValueError("length of control_features must be equal to num")

    if len(additional_input) != num:
        raise ValueError("length of additional_input must be equal to num")

    outs = np.zeros(model.input_shape[0][1:])

    control_input_length = model.input_shape[1][1]
    padded_control_features = np.pad(control_features,[(control_input_length-1,0),(0,0)],'constant')

    additional_input_length = model.input_shape[0][1]
    padded_additional_input = np.pad(additional_input,[(additional_input_length-1,0),(0,0)],'constant')

    prev_input_length = model.input_shape[0][1]
    for i in range(num):
        ctrlin = np.array([padded_control_features[i:i+control_input_length]])
        ftrin = np.concatenate([np.array([outs[-prev_input_length:]]),np.array([padded_additional_input[i:i+additional_input_length]])],axis=-1)
        x = [ftrin,ctrlin]
        newout=model.predict(x)[0][-1]
        outs = np.concatenate([outs,[newout]])

    return outs[prev_input_length:]