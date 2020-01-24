import numpy as np
import pylab as pl

from get_data import import_data_to_denoise, reconstruct_data

linear_0, non_linear_0, sigma, type_of_data, time_para, test_0, linear, non_linear, method, test = import_data_to_denoise()

to_test = 6	#78	#6	#151
i = 2

i_0 = np.where(test_0 == to_test)[0][0]
index = np.where(test == to_test)[0]

ln_p, lppl, lppl_0 = reconstruct_data(linear_0[i_0], non_linear_0[i_0], sigma[i_0], type_of_data[i_0], time_para[i_0], linear[index][i], non_linear[index][i])

fft_p = np.fft.fft(ln_p)
fft_lppl = np.fft.fft(lppl)
freq = np.fft.fftfreq(len(lppl))

fft_noise = np.abs(fft_p)**2 - np.abs(fft_lppl)**2
print np.where(fft_noise > np.abs(fft_lppl)**2, 1, 0).mean()
if lppl_0 is not None:
	print (np.max(ln_p) - np.min(ln_p))/(np.max(lppl_0) - np.min(lppl_0)+sigma[i_0])
noise = np.abs(np.fft.ifft(fft_noise))

ln_p_f = np.abs(np.fft.ifft(np.where(np.abs(freq)<0.2, fft_p, 0)))
lppl_p_f = np.abs(np.fft.ifft(np.where(np.abs(freq)<0.2, fft_lppl, 0)))

fig = pl.figure()
pl.plot(ln_p, label = 'ln(p)')
pl.plot(np.abs(np.fft.ifft(np.where(np.abs(freq)<0.2, fft_p, 0))))
if lppl_0 is not None:
	pl.plot(lppl_0, label = 'lppl$_0$')
pl.plot(lppl, label = 'lppl')
pl.plot()
pl.plot(noise-noise[0]+ln_p[0], label = 'noise')
pl.legend()

fig= pl.figure()
pl.plot(freq, np.abs(fft_p))
pl.plot(freq, np.abs(fft_lppl))
pl.show()
