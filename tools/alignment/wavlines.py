import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../sms-tools/software/models/'))
import utilFunctions as UF
import numpy as np

def txttowavlines(inputFile, outFile, fs):
	edges = np.loadtxt(inputFile)
	edges_i = (edges[np.isfinite(edges)] * fs).astype(int)
	x = np.zeros(max(edges_i)+1)
	x[edges_i] = 1.0
	UF.wavwrite(x, fs, outFile)

def wavlinestotxt(inputFile, outFile, fs):
	fs, x = UF.wavread(inputFile)
	i = np.where(np.array(x)>0.5)
	i = i[0]
	i.sort()
	times = i.astype(float)/fs
	np.savetxt(outFile,np.expand_dims(times,axis=1),fmt='%.10f')
