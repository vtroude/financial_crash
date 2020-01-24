import numpy as np
from model import LPPLS

def generate_BM(time):
	N = len(time)
	X = np.zeros(N)
	for n in range(1,N):
		X[n] = X[n-1] + np.random.normal(0., np.sqrt(time[n]-time[n-1]))

	return X

def generate_data(time, sigma=0.01, p1 = 0.05, p2 = 0.1, p3=0.15, p4=0.2, p5=0.55, p6=0.8, p7=0.9):
	p = np.random.rand()
	model = LPPLS()
	N = len(time)
	
	if p<p1:
		data, linear_0, non_linear_0 = model.generate_model(time, show_parameter=True)
		return data, linear_0, non_linear_0, 'true lppls'
	elif p<p2:
		data, linear_0, non_linear_0 = model.generate_model(time, show_parameter=True)
		data = data + np.random.normal(0., sigma, N)
		return data, linear_0, non_linear_0, 'true lppls with gaussian noise'
	elif p<p3:
		data, linear_0, non_linear_0 = model.generate_model(time, show_parameter=True)
		data = data + sigma*generate_BM(time)
		return data, linear_0, non_linear_0, 'true lppls with brownian noise'
	elif p<p4:
		data, linear_0, non_linear_0 = model.generate_qualified_model(time, show_parameter=True)
		return data, linear_0, non_linear_0, 'qualified lppls'
	elif p<p5:
		data, linear_0, non_linear_0 = model.generate_qualified_model(time, show_parameter=True)
		data = data +  np.random.normal(0., sigma, N)
		return data, linear_0, non_linear_0, 'qualified lppls with gaussian noise'
	elif p<p6:
		data, linear_0, non_linear_0 = model.generate_qualified_model(time, show_parameter=True)
		data = data +  sigma*generate_BM(time)
		return data, linear_0, non_linear_0, 'qualified lppls with brownian noise'
	elif p<p7:
		return np.random.normal(0., sigma, N), [None, None, None, None], [None, None, None], 'pure gaussian noise'
	else:
		return sigma*generate_BM(time), [None, None, None, None], [None, None, None], 'pure brownian noise'
