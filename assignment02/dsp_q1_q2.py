import numpy as np
import scipy
from scipy import * 
from matplotlib import pyplot as plt
import time
from scipy.io.wavfile import read
import numpy as np        

path1 = "piano.wav"
path2 = "impulse-response.wav"
q1_image = 'results/y_time.png'
q2_file_path = "results/res.txt"

def loadSoundFile(path):
	s, data = scipy.io.wavfile.read(path)
	return np.array(data, dtype=float)
	
#Q: If the length of 'x' is 200 and the length of 'h' is 100, what is the length of 'y' ?
#A: 299

def myTimeConv(x,h):
	x_len = len(x)
	h_len = len(h) 
	
	y_len = x_len + h_len-1
	y = np.zeros(y_len)
	
	m = y_len - x_len
	n = y_len - h_len
	
	x =np.pad(x,(0,m),'constant')
	h =np.pad(h,(0,n),'constant')
	
	for n in range (y_len):
		for k in range (y_len):
			if n >= k:
				y[n] = y[n]+x[n-k]*h[k]  
	return y

def plotfn(func):
	print(func, len(func))
	plt.figure()
	plt.plot(func)

def CompareConv(x, h):
	time1 = time.perf_counter()
	y1 = myTimeConv(x,h)
	time2 = time.perf_counter()
	y2 = np.convolve(x,h)
	time3 = time.perf_counter()
	
	timeArr = [(time2-time1), (time3-time2)]
	
	diff = y2-y1
	absDiff = np.abs(diff)
	m = np.sum(diff)/(y1.size+ y2.size)
	mabs = np.sum(absDiff)/(y1.size+ y2.size)
	stddev = np.std(diff)
	
	return [m, mabs, stddev, timeArr]

def main():
	
	x_q1 = np.ones(200)
	h = np.arange(0,1,0.04)
	h2 = np.zeros(1)
	h=np.append(h, np.arange(1,0,-0.04))
	h_q1=np.append(h,h2)
	y_q1 =  myTimeConv(x_q1,h_q1)
	
	#Generating the plot for q1
	fig, (ax0) = plt.subplots(nrows=1)

	ax0.plot(y_q1)
	ax0.set_title('question1', color='green')
	ax0.set_xlabel('Time')
	ax0.set_ylabel('Amplitude')

	plt.subplots_adjust(hspace=2)
	plt.savefig(q1_image)
  
	x_q2 = loadSoundFile(path1)
	h_q2 = loadSoundFile(path2)
	
	#As this function takes long to evaluate - since myTimeConv is evaluating in time domain, 
	#here I am evaluating 10000 samples, to just create the file res.txt Otherwise the program takes a long time to evaluate

	arr = CompareConv(x_q2[0:10000],h_q2[0:10000]) 
	print(arr)

	f=open(q2_file_path,'a')
	a = np.array(arr[0:2])
	b = np.array(arr[3])
	np.savetxt(f, a, fmt='%1.3f', newline=", ")
	f.write("\n")
	np.savetxt(f, b, fmt='%1.3f', newline=", ")
	f.write("\n")
	f.close()
	
if __name__ == "__main__":
	main() 