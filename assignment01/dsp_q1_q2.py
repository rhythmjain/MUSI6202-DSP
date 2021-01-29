import numpy as np
import matplotlib.pyplot as plt 
import scipy
from scipy import signal
from scipy.io.wavfile import read

path1 = "snare.wav"
path2 = "drum_loop.wav"
q1_image = 'results/01-correlation.png'
q2_file_path = 'results/02-snareLocation.txt'

#Function loadSoundFile(path) to load sound files stored within path and return the sample as an array.
#    input: path to an audio file
#    output: a numpy array

def loadSoundFile(path):
    s, data = scipy.io.wavfile.read(path)
    return np.array(data[:,0], dtype=float)

#Function crossCorr(a, b) to find correlation between sound and return the sample as an array.
#    input: a, b as numpy arrays
#    output: an array
def crossCorr(a, b):
    return scipy.signal.correlate(a, b)


#Function findSnarePosition(p1,p2) to find snare position in the drum loop
#    input: paths to the audio files
#    output: an array
def findSnarePosition(p1,p2):
	x = loadSoundFile(p1)
	y = loadSoundFile(p2)
	z = crossCorr(x, y)
	peaks, _ = scipy.signal.find_peaks(z, height = 1e+11)
	return peaks

if __name__ == '__main__':
	x = loadSoundFile(path1)
	y = loadSoundFile(path2)
	z = crossCorr(x, y)
	
	#Generating the plot for q1
	fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
	
	ax0.plot(x)
	ax0.set_title('Snare', color='green')
	ax0.set_xlabel('Sample Number')
	ax0.set_ylabel('Amplitude')

	ax1.plot(y)
	ax1.set_title('Drum Loop', color='green')
	ax1.set_xlabel('Sample Number')
	ax1.set_ylabel('Amplitude')

	ax2.plot(z)
	ax2.set_title('Correlated Result', color='green')
	ax2.set_xlabel('Sample Number')
	ax2.set_ylabel('Amplitude')

	plt.subplots_adjust(hspace=2)
	plt.savefig(q1_image)

	pos = findSnarePosition(path1, path2)

	#Generating the text file for q2
	np.savetxt(q2_file_path, pos)
