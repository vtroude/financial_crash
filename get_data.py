import numpy as np
import pandas as pd

from model import LPPLS
from generate_data import generate_BM

# Define dictionnary
type_of_data = ['true lppls', 'true lppls with gaussian noise', 'true lppls with brownian noise', 'qualified lppls', 'qualified lppls with gaussian noise', 'qualified lppls with brownian noise', 'pure gaussian noise', 'pure brownian noise']
method = ['Jan', 'Nelder-Mead', 'Newton-CG']
rep = range(1, 11)
name = ['test_parameters.csv', 'test.csv']

def get_parameters():
	data = pd.read_csv(name[0])
	qual_0 = data['qual'].to_numpy()
	test_0 = data['test'].to_numpy()
	type_0 = data['type'].to_numpy()
	tc_0 = data['tc'].to_numpy()
	type_index = []
	for td in type_of_data:
		type_index.append(np.where(type_0 == td)[0])

	return qual_0, test_0, tc_0, type_index

def get_data():
	data = pd.read_csv(name[1])
	qual = data['qual'].to_numpy()
	time = data['time'].to_numpy()
	sse = data['SSE'].to_numpy()
	test = data['test'].to_numpy()
	method = data['method'].to_numpy()
	tc = data['tc'].to_numpy()
	method_index = []
	for md in method:
		method_index.append(np.where(method == md)[0])
	rep_data = data['rep'].to_numpy()
	rep_index = []
	for r in rep:
		rep_index.append(np.where(rep_data == r)[0])

	return qual, test, time, sse, tc, method_index, rep_index

def import_qual_data():
	qual_0, test_0, _, type_index = get_parameters()
	qual, test, _, _, _, method_index, rep_index = get_data()

	return qual_0, test_0, type_index, qual, test, method_index, rep_index, type_of_data, method, rep, name

def import_data():
	qual_0, test_0, tc_0, type_index = get_parameters()
	qual, test, time, sse, tc, method_index, rep_index = get_data()

	return qual_0, test_0, tc_0, type_index, qual, test, time, sse, tc, method_index, rep_index, type_of_data, method, rep, name

def import_data_to_denoise():
	para = pd.read_csv(name[0])
	linear_0 = para[['A', 'B', 'C1', 'C2']].to_numpy()
	non_linear_0 = para[['m', 'tc', 'w']].to_numpy()
	sigma = para['sigma'].to_numpy()
	type_of_data = para['type'].to_numpy()
	time_para = para[['t1', 't2', 'N']].to_numpy()
	test_0 = para['test'].to_numpy()
	data = pd.read_csv(name[1])
	linear = data[['A', 'B', 'C1', 'C2']].to_numpy()
	non_linear = data[['m', 'tc', 'w']].to_numpy()
	method = data['method'].to_numpy()
	test = data['test'].to_numpy()

	return linear_0, non_linear_0, sigma, type_of_data, time_para, test_0, linear, non_linear, method, test

def match_data(method_index, rep_index, test_0, test, obs, obs_0):
	i = np.intersect1d(method_index, rep_index)
	g = np.where(np.isin(test_0, test[i])==True)[0]

	return obs[i][g], obs_0[g], i, g

def reconstruct_data(linear_0, non_linear_0, sigma, type_of_data, time_para, linear, non_linear):
	time = np.linspace(time_para[0], time_para[1], time_para[2])	
	if None in linear_0 or None in non_linear_0 or np.isnan(linear_0).any() or np.isnan(non_linear_0).any():
		ln_p = 0.
		lppl_0 = None
	else:
		ln_p = LPPLS().model(time, linear_0, non_linear_0)
		lppl_0 = ln_p
	if 'gaussian' in type_of_data:
		ln_p = ln_p + np.random.normal(0., sigma, len(time))
	elif 'brownian' in type_of_data:
		ln_p = ln_p + sigma*generate_BM(time)

	lppl = LPPLS().model(time, linear, non_linear)

	return ln_p, lppl, lppl_0











