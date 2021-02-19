import matplotlib.pylab as plt
import numpy as np 
import scipy.signal

def generateSinusoidal(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):
    t = np.arange(0, length_secs,1.0/(sampling_rate_Hz))
    x = amplitude * np.sin(2*np.pi*frequency_Hz*t+ phase_radians)
    return t,x

def plotfn(x_plot, y_plot, x_label, y_label):
    plt.figure()
    plt.plot(x_plot, y_plot)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def generateSquare(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians, num_sin):
    sq = 0
    for i in range(1, num_sin*2, 2):
        sq += (generateSinusoidal(amplitude, sampling_rate_Hz, i*frequency_Hz, length_secs, phase_radians)[1])/i
    sq_final = (4/np.pi) * sq

    t = generateSinusoidal(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians)[0]    
    return t, sq_final

def computeSpectrum(x, sample_rate_Hz):
    half_N = int(len(x)//2) + 1
    ft = np.fft.fft(x)
    XAbs = np.abs(ft[:half_N])

    XIm = np.imag(ft[:half_N])
    XRe = np.real(ft[:half_N])
    
    XPhase = np.angle(ft[:half_N])

    f = np.linspace(0, sample_rate_Hz, half_N)
    return f, XAbs, XPhase, XRe, XIm

def stride_trick(a, stride_length, stride_step):
    nrows = ((a.size - stride_length) // stride_step) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, stride_length), strides=(stride_step*n, n))

def generateBlocks(x, sample_rate_Hz, block_size, hop_size):
    n = len(x)
    
    if block_size < hop_size:
        print("Error in block and hop sizes")
        return
    
    zero_padding = block_size-hop_size
    rest_samples = np.abs(n - zero_padding) % np.abs(block_size - zero_padding)
    pad_signal = np.append(x, np.array([0] * int(zero_padding) * int(rest_samples != 0.)))
    
    t = []
    for i in range(0, n-hop_size, hop_size):
        t.append(i/44100)
    
    frames = stride_trick(pad_signal, block_size, hop_size)
    print(frames.shape)
    return t,frames

def mySpecgram(x, block_size, hop_size, sample_rate_Hz, window_type):

    t, sig = generateBlocks(x, sample_rate_Hz, block_size, hop_size)
    fftDone = []
    freq_vector = []
    
    if window_type=='rect':
        for i in sig:
            dat = computeSpectrum(i*scipy.signal.boxcar(block_size), 44100)
            fftDone.append(dat[1])
    elif window_type == 'hann':
        for i in sig:
            dat = computeSpectrum(i*np.hanning(block_size), 44100)
            fftDone.append(dat[1])

    else:
        print("Wrong window specified")
        
    print(len(x))
    freq_vector = np.linspace(0, len(x), block_size)
        
    return t, freq_vector, fftDone

def plotSpecgram(freq_vector, time_vector, magnitude_spectrogram, title):
    if len(freq_vector) < 2 or len(time_vector) < 2:
        return

    Z = 20. * np.log10(magnitude_spectrogram)
    Z = np.flipud(Z)
  
    pad_xextent = (time_vector[1] - time_vector[0]) / 2
    xmin = np.min(time_vector) - pad_xextent
    xmax = np.max(time_vector) + pad_xextent
    extent = xmin, xmax, freq_vector[0], freq_vector[-1]
  
    im = plt.imshow(Z, None, extent=extent, origin='upper')
    plt.axis('auto')
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    #q1
    A = 1.0
    fs = 44100
    f0 = 400
    len_secs = 0.5
    phase_radians = np.pi/2    
    (t1,x1) = generateSinusoidal(A, fs, f0, len_secs, phase_radians)
    plotfn(t1[:200],x1[:200],'Time(sec)', 'Amplitude')

    #q2
    phase_radians=0
    num_sin=10
    (t2,x2) = generateSquare(A, fs, f0, len_secs, phase_radians, num_sin)
    plotfn(t2[:200],x2[:200],'Time(sec)', 'Amplitude')
    

    #q3
    (f1,XAbs1,XPhase1,XRe1,XIm1) = computeSpectrum(x1, fs)
    (f2,XAbs2,XPhase2,XRe2,XIm2) = computeSpectrum(x2, fs)
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('Magnitude Spectrum')
    plt.plot(f1,XAbs1)
    
    plt.subplot(2,1,2)
    plt.xlabel('Frequency')
    plt.ylabel('Phase')
    plt.title('Phase Spectrum')
    plt.plot(f1, XPhase1)
    
    plt.show()
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('Magnitude Spectrum')
    plt.plot(f2,XAbs2)

    plt.subplot(2,1,2)
    plt.xlabel('Frequency')
    plt.ylabel('Phase')
    plt.title('Phase Spectrum')
    plt.plot(f2, XPhase2)
    
    plt.show()
    
    #q4
    block_size = 2048
    hop_size = 1024
    timeArr_hann, freq_vector_hann, fftDone_hann = mySpecgram(x2, block_size, hop_size, fs, "hann")
    fftDone_hann = np.transpose(fftDone_hann)
    plotSpecgram(freq_vector_hann, timeArr_hann, fftDone_hann, "Frequency spectrum for Square wave with hanning window")

    timeArr_rect, freq_vector_rect, fftDone_rect = mySpecgram(x2, block_size, hop_size, fs, "rect")
    fftDone_rect = np.transpose(fftDone_rect)
    plotSpecgram(freq_vector_rect, timeArr_rect, fftDone_rect, "Frequency spectrum for Square wave with rectangular window")
