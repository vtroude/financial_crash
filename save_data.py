import pandas as pd
import numpy as np
from os import path

def save_parameters(type_of_data, N, t1, t2, sigma, linear, non_linear, qual):

	exist = path.exists('test_parameters.csv')
	if exist:
		data = pd.read_csv('test_parameters.csv')
		test_n = data['test'][data.index[-1]]+1
	else:
		test_n = 1

	raw_data = {'test': [test_n], 'N': [N], 't1': [t1], 't2': [t2], 'sigma': [sigma], 'A': [linear[0]], 'B': [linear[1]], 'C1': [linear[2]], 'C2': [linear[3]], 'm': [non_linear[0]], 'tc': [non_linear[1]], 'w': [non_linear[2]], 'type': [type_of_data], 'qual': [qual]}

	if exist:
		with open('test_parameters.csv', 'a') as f:
			pd.DataFrame(raw_data).to_csv(f, header=False)
	else:
		pd.DataFrame(raw_data).to_csv('test_parameters.csv')

	return test_n

def save_test(test_n, linear, non_linear, sse, qual, time, rep, Break, method):
	raw_data = {'test': test_n, 'A': [linear[0]], 'B': [linear[1]], 'C1': [linear[2]], 'C2': [linear[3]], 'm': [non_linear[0]], 'tc': [non_linear[1]], 'w': [non_linear[2]], 'SSE': [sse], 'qual': [qual], 'time': [time], 'rep': [rep], 'break': [Break], 'method': [method]}
	exist = path.exists('test.csv')
	if not exist:
		pd.DataFrame(raw_data).to_csv('test.csv')
	else:
		with open('test.csv', 'a') as f:
			pd.DataFrame(raw_data).to_csv(f, header=False)
	
