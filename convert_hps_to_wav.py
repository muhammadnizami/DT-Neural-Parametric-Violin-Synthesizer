
import numpy as np
from scipy.signal import get_window
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tools/sms-tools/software/models/'))
import utilFunctions as UF
import hpsModel as HPS
import sineModel as SM

def main(inputFile, outputFile, fs=44100, window='hamming', M=2001, N=2048, t=-80, minSineDur=0.02,
          maxnSines=50, freqDevOffset=10, freqDevSlope=0.001, ignore_stoc=False, scale=None, 
          mag_max=None, asymptotic_alpha=None):

    inputData = np.load(inputFile)
    hfreq, hmag, stocEnv = [np.array(a) for a in zip(*inputData)]

    if asymptotic_alpha is not None:
        assert asymptotic_alpha>0, "asymptotic alpha must be >0"
        if mag_max is None:
            mag_max=0
        hmag = -1/asymptotic_alpha * np.log(np.exp(-asymptotic_alpha*hmag)+np.exp(-asymptotic_alpha*mag_max)) 
        stocEnv = -1/asymptotic_alpha * np.log(np.exp(-asymptotic_alpha*stocEnv)+np.exp(-asymptotic_alpha*mag_max)) 
    if mag_max is not None:
        shift_amount = np.max(hmag,axis=0)
        hmag = np.where(np.max(hmag,axis=0)>mag_max,hmag-shift_amount,hmag)

    # size of fft used in synthesis
    Ns = 512
    # hop size (has to be 1/4 of Ns)
    H = 128

    # synthesize the output sound from the sinusoidal representation
    # synthesize a sound from the harmonic plus stochastic representation
    if not ignore_stoc:
        y, yh, yst = HPS.hpsModelSynth(hfreq, hmag, np.array([]), stocEnv, Ns, H, fs)
    else:
        y = SM.sineModelSynth(hfreq, hmag, np.array([]), Ns, H, fs)

    if scale and scale!='no':
        divisor = np.std(y)
        if scale=='2std':
            divisor = divisor * 2
        y = y/divisor

    #clipping for safety
    y = np.clip(y,-1,1)

    UF.wavwrite(y, fs, outputFile)

import argparse
import ntpath
import csv
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path',required=True)
    parser.add_argument('--output-path',required=True)
    parser.add_argument('--ignore-stoc',action='store_true')
    parser.add_argument('--scale',choices=['std','2std','no'])
    parser.add_argument('--mag-max',type=float)
    parser.add_argument('--asymptotic-alpha',type=float)
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path

    if os.path.isdir(input_path) and os.path.isdir(output_path):
        files = os.listdir(input_path)
        for f in files:
            if f[-4:]=='.npy':
                of = ntpath.join(output_path,ntpath.basename(f)[:-4]+'.wav')
                print('converting',f,'to',of)
                main(os.path.join(input_path,f),of,ignore_stoc=args.ignore_stoc,scale=args.scale, mag_max=args.mag_max)
            
    elif os.path.isfile(input_path) and input_path[-4:]=='.npy':
        if os.path.isdir(output_path):
            output_path = ntpath.join(output_path,ntpath.basename(f)[:-4]+'.wav')
        main(input_path, output_path,ignore_stoc=args.ignore_stoc,scale=args.scale, mag_max=args.mag_max)

    else:
        print('paths must be both existing directory or input path is \'.npy\' file')