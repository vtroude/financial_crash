import numpy as np
import time as tm

from model import LPPLS
from generate_data import generate_data as gd
from lppl_fit import lppl_fit_standard as lft
import save_data as sd

def test(N = 1000, sigma = 0.01, method = ['Jan', 'Nelder-Mead', 'Newton-CG']):
	model = LPPLS()
	time = np.arange(N)
	tc0, tc1 = model.tc_bounds(time)
	bnds = ((0., 1.), (tc0, tc1), (4., 25.))
	
	# Generate data and parameters
	ln_p, linear_0, non_linear_0, type_of_data = gd(time, sigma, 0., 0., 0., 0.2, 0.65, 1.)
	model.print_parameter(linear_0, non_linear_0)
	qual_0 = model.qual(time, linear_0, non_linear_0)
	# Save parameters and data type
	test_n = sd.save_parameters(type_of_data, N, time[0], time[-1], sigma, linear_0, non_linear_0, qual_0)
	# Launch test for different methods
	for m in method:
		if m == 'Jan':
			t = tm.time()
			pars, sse = lft(time, ln_p)
			t = tm.time() - t
			linear = np.array(pars[:4])
			non_linear = np.array([pars[5], pars[4], pars[6]])
			qual = model.qual(time, linear, non_linear)
			Break = not model.in_bounds(non_linear, bnds)
			sd.save_test(test_n, linear, non_linear, sse, qual, t, 1, Break, m)
		else:
			linear, non_linear, sse, Break, t = model.test_fit(ln_p, time, bnds, m)
			rep_max = len(linear)
			for n in range(rep_max):
				qual = model.qual(time, linear[n], non_linear[n])
				if n < rep_max-1:
					sd.save_test(test_n, linear[n], non_linear[n], sse[n], qual, t[n], n+1, False, m)
				else:
					sd.save_test(test_n, linear[n], non_linear[n], sse[n], qual, t[n], n+1, Break, m)

def generic_test(M = 10, n = 100):
	sigma = np.logspace(-8, 1, 10000)
	for k in range(n):
		N = np.random.randint(50, 1001)
		s = sigma[np.random.randint(10000)]
		for m in range(M):
			test(N, s)
#test()
generic_test()
			
			
