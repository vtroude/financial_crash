import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime
import pylab as pl
import matplotlib.gridspec as gridspec
import os.path
from os import path
from lppls import LPPLS

class Analyse:
	''' A Dynamic Fit Routine '''

	def __init__(self): pass

	def save_parameter_routine(self, t1, t2, M, Tc, W, A, B, C1, C2, SSE, qual, name):
		raw_data = {'t1': t1, 't2': t2, 'm': M, 'tc': Tc, 'w': W, 'A': A, 'B': B, 'C1': C1, 'C2': C2, 'SSE': SSE, 'qual': qual}
		df = pd.DataFrame(raw_data)
		df.to_csv(name+'.csv')

	def save_parameter(self, M, Tc, W, A, B, C1, C2, SSE, qual, name):
		raw_data = {'m': M, 'tc': Tc, 'w': W, 'A': A, 'B': B, 'C1': C1, 'C2': C2, 'SSE': SSE, 'qual': qual}
		pd.DataFrame(raw_data).to_csv(name+'.csv')

	def qual_type(self, b):
		if b>0:
			return -1
		elif b<0:
			return 1

	def qual(self, m, tc, w, sse, b, tc0, tc1):
		a = (m>0 and m<1)
		a = (a and tc>tc0 and tc<tc1)
		a = (a and w>2 and w<25)
		a = (a and sse is not 'nan')
		if a:
			return self.qual_type(b)
		else:
			return 0

	def bound_tc(self, N, t2):
		tc0 = np.maximum(-60, -N*0.5)+t2
		tc1 = np.minimum(252, N*0.5)+t2

		return tc0, tc1

	def routine(self, price, time, N0, N1, name):
		I = range(N1+1, len(time)+1, 20)
		N = range(N0, N1, 20)
		model = LPPLS()
		t1, t2 = [], []
		M, Tc, W = [], [], []
		A, B, C1, C2 = [], [], [], []
		SSE, qual = [], []
		for i in I:
			for n in N:
				print '\t t in [',time[i-n],',',time[i-1],']'
				print
				tc0, tc1 = self.bound_tc(n, time[i-1])
				tc_guess = (tc1-tc0+1)*0.5+tc0
				t1.append(time[i-n])
				t2.append(time[i-1])
				m, tc, w, a, b, c1, c2, sse = model.fit(price[i-n:i], time[i-n:i], tc_guess=tc_guess, tc0=tc0, tc1=tc1, rep=1, descent='steepest', precision=10.**(-16), grid_search=True)
				M.append(m)
				Tc.append(tc)
				W.append(w)
				A.append(a)
				B.append(b)
				C1.append(c1)
				C2.append(c2)
				SSE.append(sse)
				qual.append(self.qual(m, tc, w, sse, b, tc0, tc1))
		self.save_parameter_routine(t1, t2, M, Tc, W, A, B, C1, C2, SSE, qual, name)

	def generate_parameter(self, phi, phi0, phi1):
		if phi is None:
			return np.random.rand()*(phi1-phi0) + phi0
		else:
			return phi

	def generate_lppls(self, N=1000, sigma=1., A=None, B=None, C1=None, C2=None, m=None, tc=None, w=None):
		t = np.linspace(0, N-1, N).astype(float)
		A = self.generate_parameter(A, -1., 1.)
		B = self.generate_parameter(B, -1., 1.)
		C1 = self.generate_parameter(C1, -.1, .1)
		C2 = self.generate_parameter(C2, -.1, .1)
		m = self.generate_parameter(m, 0., 1.)
		tc0, tc1 = self.bound_tc(N, N-1)
		tc = self.generate_parameter(tc, tc0, tc1)
		w = self.generate_parameter(w, 2., 25.)
		y = LPPLS().lppls(t, m, tc, w, A, B, C1, C2)+np.random.normal(0., sigma, N)
		
		return np.exp(y), t, np.array([A, B, C1, C2, m, tc, w])

	def generate_bm(self, N=1000, x0 = 0., delta=1):
		dt = 1./delta
		x = np.zeros(N)
		x[0] = x0
		for n in range(1,N):
			x0 = x[n-1]
			for k in range(delta):
				x0 = x0 + np.random.normal(0., dt**0.5)
			x[n] = x0

		return np.exp(x), np.arange(N)

	def test(self, price, time, N, name, phi=None, sigma=0, precision=10.**(-16), plot=True):
		tc0, tc1 = self.bound_tc(N, time[-1])
		dev = np.zeros((4, 2, 2))
		SSE = np.zeros((4, 2, 2))
		m = np.zeros((4, 2, 2))
		tc = np.zeros((4, 2, 2))
		w = np.zeros((4, 2, 2))
		a = np.zeros((4, 2, 2))
		b = np.zeros((4, 2, 2))
		c1 = np.zeros((4, 2, 2))
		c2 = np.zeros((4, 2, 2))
		qual = np.zeros((4, 2, 2))
		rep = [1,2,3,4]
		descent = ['steepest', 'conjugate']
		grid = [True, False]

		if path.exists(name+'.csv'):
			data = pd.read_csv(name+'.csv')
			test_number = data['test'][data.index[-1]]+1
			if phi is not None:
				raw_data = {'test': [test_number], 't1': [time[0]], 't2': [time[-1]], 'N': [N], 'sigma': [sigma], 'A*': [phi[0]], 'B*': [phi[1]], 'C1*': [phi[2]], 'C3*': [phi[3]], 'm*': [phi[4]], 'tc*': [phi[5]], 'w*': [phi[6]]}
			else:
				raw_data = {'test': [test_number], 't1': [time[0]], 't2': [time[-1]], 'N': [N]}
			with open(name+'_parameters.csv', 'a') as f:
				pd.DataFrame(raw_data).to_csv(f, header=False)
		else:
			test_number = 1
			if phi is not None:
				raw_data = {'test': [test_number],'t1': [time[0]], 't2': [time[-1]], 'N': [N], 'sigma': [sigma], 'A*': [phi[0]], 'B*': [phi[1]], 'C1*': [phi[2]], 'C3*': [phi[3]], 'm*': [phi[4]], 'tc*': [phi[5]], 'w*': [phi[6]]}
			else:
				raw_data = {'test': [test_number],'t1': [time[0]], 't2': [time[-1]], 'N': [N]}
			pd.DataFrame(raw_data).to_csv(name+'_parameters.csv')

		model = LPPLS()
		for k1 in range(4):
			for k2 in range(2):
				for k3 in range(2):
					m[k1,k2,k3], tc[k1,k2,k3], w[k1,k2,k3], a[k1,k2,k3], b[k1,k2,k3], c1[k1,k2,k3], c2[k1,k2,k3], SSE[k1,k2,k3] = model.fit(price, time, tc0=tc0, tc1=tc1, rep=rep[k1], descent=descent[k2], precision=precision, grid_search=grid[k3])
					qual[k1,k2,k3] = self.qual(m[k1,k2,k3], tc[k1,k2,k3], w[k1,k2,k3], SSE[k1,k2,k3], b[k1,k2,k3], tc0, tc1)
					if phi is not None:
						dev[k1,k2,k3] = np.sum((phi-np.array([a[k1,k2,k3],b[k1,k2,k3],c1[k1,k2,k3],c2[k1,k2,k3], m[k1,k2,k3],tc[k1,k2,k3],w[k1,k2,k3]]))**2)
						raw_data = {'test': [test_number], 'descent': [descent[k2]], 'grid': [grid[k3]], 'rep': [rep[k1]], 'A': [a[k1,k2,k3]], 'B': [b[k1,k2,k3]], 'C1': [c1[k1,k2,k3]], 'C2': [c2[k1,k2,k3]], 'm': [m[k1,k2,k3]], 'tc': [tc[k1,k2,k3]], 'w': [w[k1,k2,k3]], 'qual': [qual[k1,k2,k3]], 'SSE': [SSE[k1,k2,k3]], 'dev': [dev[k1,k2,k3]]}
					else:
						raw_data = {'test': [test_number], 'descent': [descent[k2]], 'grid': [grid[k3]], 'rep': [rep[k1]], 'A': [a[k1,k2,k3]], 'B': [b[k1,k2,k3]], 'C1': [c1[k1,k2,k3]], 'C2': [c2[k1,k2,k3]], 'm': [m[k1,k2,k3]], 'tc': [tc[k1,k2,k3]], 'w': [w[k1,k2,k3]], 'qual': [qual[k1,k2,k3]], 'SSE': [SSE[k1,k2,k3]]}
					if test_number == 1 and k3==0 and k2==0 and k1==0:
						pd.DataFrame(raw_data).to_csv(name+'.csv')
					else:
						with open(name+'.csv', 'a') as f:
							pd.DataFrame(raw_data).to_csv(f, header=False)

		if plot:
			col1 = ['--', '-', ':', '-.']
			col2 = ['g', 'r', 'k', 'c']
			fig = pl.figure()
			pl.plot(time, np.log(price), label='price')
			for k1 in range(4):
				for k2 in range(2):
					for k3 in range(2):
						pl.plot(time, model.lppls(time, m[k1,k2,k3], tc[k1,k2,k3], w[k1,k2,k3], a[k1,k2,k3], b[k1,k2,k3], c1[k1,k2,k3], c2[k1,k2,k3]), col1[k1]+col2[k2*2+k3], label='gid='+str(grid[k3])+', '+descent[k2]+', rep='+str(rep[k1]))
			pl.ylabel('ln(price)')
			pl.xlabel('time')
			pl.legend()
			dpi = fig.get_dpi()
			h, w = fig.get_size_inches()
			fig.set_size_inches(h*2, w*2)
			fig.savefig('figure/'+name+'_'+str(test_number)+'.png')
			pl.close()


	def test_lppls(self, N=1000, sigma=1., name=None, precision=10.**(-16), plot=True):
		#model = LPPLS()
		if name is None:
			name = 'test_true_lppls'
		price, time, phi = self.generate_lppls(N, sigma)
		self.test(price, time, N, name, phi, sigma, precision, plot)

	def generic_test_lppls(self, sigma0, sigma1, n=10, N=1000, M=10, name=None, precision=10.**(-16), plot=True):
		sigma = np.linspace(sigma0, sigma1, n)
		for i in range(n):
			for j in range(M):
				self.test_lppls(N, sigma[i], name, precision, plot)

	def test_bm(self, N=1000, name=None, precision=10.**(-16), plot=True):
		if name is None:
			name = 'test_bm'
		price, time = self.generate_bm()
		self.test(price, time, len(time), name, precision=precision, plot=plot)
		
	def generic_test_bm(self, n=10, N=1000, name=None, precision=10.**(-16), plot=True):
		for k in range(n):
			self.test_bm(N, name, precision, plot)

	def mean_std_per_value(self, x, y):
		x_new = np.array(list(set(x)))
		y_mean = np.zeros(len(x_new))
		y_dev = np.zeros(len(x_new))
		for n in range(len(x_new)):
			index = np.where(x==x_new[n])[0]
			y_mean[n] = np.mean(y[index])
			y_dev[n] = np.std(y[index])

		return x_new, y_mean, y_dev

	def barplot_mean(self, y, sub_index, color, label, x_label, y_label, name, n1=2, n2=2, n3=4):
		fig = pl.figure()
		index = np.arange(n1*n2)
		width = 1./(n1*n2+1)
		pos = np.arange(-n1*n2*0.5,n1*n2*0.5)*width+width*0.5
		to_label = True
		for k1 in range(n1):
			for k2 in range(n2):
				for k3 in range(n3):
					if to_label:
						pl.bar(index[k2+k1*n2]+pos[k3], np.mean(y[sub_index[k3+k2*n3+k1*n2*n3]]), width, alpha=0.5, color=color[k3], label='rep='+label[k3])
					else:
						pl.bar(index[k2+k1*n2]+pos[k3], np.mean(y[sub_index[k3+k2*n3+k1*n3*n2]]), width, alpha=0.5, color=color[k3])
				to_label = False
		pl.xticks(index, x_label)
		pl.ylabel(y_label)
		pl.legend()
		dpi = fig.get_dpi()
		h, w = fig.get_size_inches()
		fig.set_size_inches(h*2, w*2)
		fig.savefig('figure/'+name+'.png')

	def plot_mean_std_per_value(self, x, y, sub_index, color, label, size, x_label, y_label, name, yscale='log'):
		fig = pl.figure()
		gs = gridspec.GridSpec(2,2, wspace=-0.5, hspace=0.3)
		ax1 = fig.add_subplot(gs[0,0])
		ax2 = fig.add_subplot(gs[1,0])
		for index,c,l,s in zip(sub_index, color, label, size):
			new_x, mean, std = self.mean_std_per_value(x, y[index])
			ax1.plot(new_x, mean, c, label = l, markersize = s)
			ax2.plot(new_x, std, c, markersize = s)
		ax1.set_yscale(yscale)
		ax2.set_yscale(yscale)
		ax1.set_xlabel(x_label)
		ax2.set_xlabel(x_label)
		ax1.set_ylabel('average '+y_label)
		ax2.set_ylabel('standard deviation '+y_label)
		ax3 = fig.add_subplot(gs[:,1])
		h, l = ax1.get_legend_handles_labels()
		ax3.legend(h,l, borderaxespad=0, loc='center right')
		ax3.axis('off')
		dpi = fig.get_dpi()
		h, w = fig.get_size_inches()
		fig.set_size_inches(h*2, w*2)
		fig.savefig('figure/'+name+'.png')

	def get_index_and_label(self, data):
		descent_index = [np.where(data['descent']=='steepest')]
		descent_index.append(np.where(data['descent']=='conjugate'))
		grid_index = [np.where(data['grid']==True)]
		grid_index.append(np.where(data['grid']==False))
		rep_index = []
		for n in range(1,5):
			rep_index.append(np.where(data['rep']==n))

		descent_color = ['+', '<']
		grid_size = [4, 2]
		rep_color = ['b', 'r', 'g', 'k']

		descent_label = ['steepest', 'conjugate']
		grid_label = ['True', 'False']
		rep_label = ['1', '2', '3', '4']
		
		return descent_index, grid_index, rep_index, descent_color, grid_size, rep_color, descent_label, grid_label, rep_label

	def get_data(self, data):
		sse = data['SSE'].to_numpy()
		qual = data['qual'].to_numpy()

		return sse, qual

	def get_data_true_lppls(self, data):
		dev = data['dev'].to_numpy()
		sse, qual = self.get_data(data)

		return dev, sse, qual

	def initialize_study(self, name=None, true_lppls = True):
		if name is None and true_lppls:
			name = 'test_true_lppls'
		elif name is None:
			name = 'test_bm'
		data = pd.read_csv(name+'.csv')
		descent_index, grid_index, rep_index, descent_color, grid_size, rep_color, descent_label, grid_label, rep_label = self.get_index_and_label(data)
		if true_lppls:
			dev, sse, qual = self.get_data_true_lppls(data)
			sigma = pd.read_csv(name+'_parameters.csv')['sigma'].to_numpy()
		
			return name, dev, sse, qual, sigma, descent_index, grid_index, rep_index, descent_color, grid_size, rep_color, descent_label, grid_label, rep_label
		else :
			sse, qual = self.get_data(data)

			return name, sse, qual, descent_index, grid_index, rep_index, descent_color, grid_size, rep_color, descent_label, grid_label, rep_label

	def get_global_index_and_label(self, descent_index, grid_index, rep_index, descent_color, grid_size, rep_color, descent_label, grid_label, rep_label):			
		sub_index = []
		color = []
		label = []
		size = []
		for k1 in range(len(descent_index)):
			for k2 in range(len(grid_index)):
				for k3 in range(len(rep_index)):
					sub_index.append(np.intersect1d(np.intersect1d(descent_index[k1], grid_index[k2], True), rep_index[k3], True))
					color.append(descent_color[k1]+rep_color[k3])
					label.append('descent='+descent_label[k1]+', grid='+grid_label[k2]+', rep='+rep_label[k3])
					size.append(grid_size[k2])

		return sub_index, color, label, size

	def compare_method(self, name=None, show=True, true_lppls=True):
		if true_lppls:
			name, dev, sse, qual, sigma, descent_index, grid_index, rep_index, descent_color, grid_size, rep_color, descent_label, grid_label, rep_label = self.initialize_study(name, true_lppls)

		else:
			name, sse, qual, descent_index, grid_index, rep_index, descent_color, grid_size, rep_color, descent_label, grid_label, rep_label = self.initialize_study(name, true_lppls)

		sub_index, color, label, size = self.get_global_index_and_label(descent_index, grid_index, rep_index, descent_color, grid_size, rep_color, descent_label, grid_label, rep_label)

		name = name+'_stat'
		
		if true_lppls:
			fig = pl.figure()
			for index,c,l,s in zip(sub_index, color, label, size):
				pl.plot(dev[index], sse[index], c, label = l, markersize = s)
			pl.xscale('log')
			pl.yscale('log')
			pl.xlabel('Deviation')
			pl.ylabel('SSE')
			pl.legend()
			dpi = fig.get_dpi()
			h, w = fig.get_size_inches()
			fig.set_size_inches(h*2, w*2)		
			fig.savefig('figure/'+name+'_dev_sse'+'.png')

			self.barplot_mean(np.abs(qual), sub_index, rep_color, rep_label, ['steepest with grid', 'steepest without grid', 'conjugate with grid', 'conjugate without grid'], 'Average qualification', name+'_qual')
			self.plot_mean_std_per_value(sigma, sse, sub_index, color, label, size, '$\sigma$ (standard deviation of the gaussian noise)', 'SSE', name+'_sse=f(noise)')
			self.plot_mean_std_per_value(sigma, dev, sub_index, color, label, size, '$\sigma$ (standard deviation of the gaussian noise)', 'Deviation', name+'_dev=f(noise)')
			self.plot_mean_std_per_value(sigma, np.abs(qual), sub_index, color, label, size, '$\sigma$ (standard deviation of the gaussian noise)', 'qualifucation', name+'_qual=f(noise)', yscale='linear')
		else:
			fig = pl.figure()
			for index,c,l,s in zip(sub_index, color, label, size):
				pl.plot(sse[index], np.abs(qual[index]), c, label = l, markersize = s)
			pl.xscale('log')
			pl.xlabel('SSE')
			pl.ylabel('|qual|')
			pl.legend()
			dpi = fig.get_dpi()
			h, w = fig.get_size_inches()
			fig.set_size_inches(h*2, w*2)		
			fig.savefig('figure/'+name+'_sse_qual'+'.png')

			self.barplot_mean(np.abs(qual), sub_index, rep_color, rep_label, ['steepest with grid', 'steepest without grid', 'conjugate with grid', 'conjugate without grid'], 'Average qualification', name+'_qual')

		if show: pl.show()
			

	def rolling_volatility(self, p, w):
		sigma = np.zeros(len(p)-w)
		for n in range(len(sigma)):
			sigma[n] = np.std(p[n:n+w])
		
		return sigma

	def epsilon(self, price, time, e0=2.55, w=35, drawup = True):
		log_p = np.log(price)
		#dt = time[1:]-time[:-1]
		r = (log_p[1:]-log_p[:-1])	#/dt
		epsilon = e0*self.rolling_volatility(log_p, w)
		i0 = w
		i1 = []
		N = len(r)
		not_finished = i0<N-1
		while not_finished:
			P_i0 = np.zeros(N-i0)
			for n in range(i0, N):
				P_i0[n-i0] = np.sum(r[i0:n+1])
			i = 0
			for n in range(i0+1, N):
				if drawup:
					delta = np.max(P_i0[:n-i0+1]) - P_i0[n-i0]
				else:
					delta = P_i0[n-i0] - np.min(P_i0[:n-i0+1])
				#if delta>epsilon[n-w-1]:
				if delta>epsilon[i0-w-1]:
					i = n
					break
			if i == 0:
				not_finished = False
				break	
			else:
				if drawup:
					i1.append(np.argmax(P_i0[:i-i0+1])+i0)
				else:
					i1.append(np.argmin(P_i0[:i-i0+1])+i0)
				i0 = i1[-1]+1
				if i0>=N-1:
					not_finished = False
					break
		if len(i1)==0:
			return None
		else:
			return time[np.asarray(i1)+1]

	def rolling_qual(self, name, error=2):
		data = pd.read_csv(name+'.csv')
		qual = []
		SSE = []
		for b, sse in zip(data['B'], data['SSE']):
			if ((sse is not 'nan') and (sse<error)):
				qual.append(self.qual_type(b))
			else:
				qual.append(0)
			SSE.append(sse)

		qual = np.asarray(qual)
		SSE = np.asarray(SSE)
		Cl = np.zeros(len(qual))
		for n in range(len(qual)):
			Cl[n] = float(np.sum(qual[:n+1]))/(n+1)

		fig = pl.figure()
		ax1 = fig.add_subplot(211)
		ax1.plot(qual, '.b', label='qual')
		ax1.set_ylabel('qual')
		ax1.legend()
		ax2 = ax1.twinx()
		ax2.plot(SSE, '.r', label='SSE')
		ax2.set_ylabel('SSE', color='r')
		for tl in ax2.get_yticklabels():
			tl.set_color('r')
		ax2.legend()
		ax3 = fig.add_subplot(212)
		ax3.plot(Cl, '.b', label='Cl')
		ax3.set_ylabel('Cl')
		ax3.legend()

	def peak(self, peak, y_min, y_max):
		pl.plot([peak, peak+10**(-8)], [y_min, y_max], 'r-')

	def plot_peak(self, price, time, t_peaks):
		y = np.log(price)
		std = np.std(y)
		y_max = np.max(y)+std
		y_min = np.min(y)-std
		pl.plot(time, np.log(price), label = 'price')
		for peak in t_peaks:
			self.peak(peak, y_min, y_max)
		pl.ylim([y_min, y_max])
		pl.legend()

	def plot_qual(self, name, price = None, time = None):
		data = pd.read_csv(name+'.csv')
		t2 = data['t2'].to_numpy()
		qual = data['qual'].to_numpy()
		t2, cl, std = self.mean_std_per_value(t2, qual)
		arg = np.argsort(t2)
		t2 = t2[arg]
		cl = cl[arg]
		std = std[arg]
		arg = np.where(time<=t2[0])[0][-1]
		fig = pl.figure()
		if price is None:
			ax1 = fig.add_subplot(111)
			ax1.plot(t2, cl, 'k', label = 'Cl($t_2$)')
			ax1.set_xlabel('time []')
			ax1.set_ylabel('Cl($t_2$)')
			ax1.set_ylim([-1.001,1.001])
			ax1.legend(loc='upper left')
			ax2 = ax1.twinx()
			ax2.plot(t2, std, 'b', label='$\sigma_{Cl}$(t2)')
			ax2.set_ylabel('$\sigma_{Cl}$(t2)')
			ax2.spines['right'].set_color('b')
			ax2.tick_params(axis='y', colors='b')
			ax2.yaxis.label.set_color('b')
			ax2.title.set_color('b')
			ax2.legend(loc='upper right')
		else:
			gs = gridspec.GridSpec(2, 1)
			ax1 = fig.add_subplot(gs[0,0])			
			ax1.plot(time[arg:], np.log(price[arg:]), 'k', label='price')
			ax1.set_xlabel('time []')
			ax1.set_ylabel('ln(price) []')
			ax1.legend(loc='upper left')
			ax2 = ax1.twinx()
			ax2.plot(t2, cl, 'r', label = 'Cl($t_2$)')
			ax2.set_xlabel('time []')
			ax2.set_ylabel('Cl($t_2$)')
			ax2.set_ylim([-1.001,1.001])
			ax2.spines['right'].set_color('r')
			ax2.tick_params(axis='y', colors='r')
			ax2.yaxis.label.set_color('r')
			ax2.title.set_color('r')
			ax2.legend(loc='upper right')
			ax3 = fig.add_subplot(gs[1,0])
			ax3.plot(t2, std, 'b', label='$\sigma_{Cl}$(t2)')
			ax3.set_xlabel('time []')
			ax3.set_ylabel('$\sigma_{Cl}$(t2)')
			ax3.legend()
		dpi = fig.get_dpi()
		h, w = fig.get_size_inches()
		fig.set_size_inches(h*2, w*2)
		pl.show()
		

################################################################
''' Test routine '''

def get_data():
	nasdaq = pdr.get_data_yahoo('^IXIC', start=datetime.datetime(2016,01,1), end=datetime.datetime(2019, 06,1))

	N = len(nasdaq['Close'])
	time = np.array((nasdaq['Close'].index-nasdaq['Close'].index[0]).days).reshape(-1)
	price = np.zeros(N)
	for n in range(N): price[n]=nasdaq['Close'][n]

	return price, time

def test_routine():
	price, time = get_data()

	routine = Analyse()
	routine.routine(price, time, 20, 504, 'nasdaq_01-01-16_01-06-19')

def test_rolling_qual():
	analyse = Analyse()
	analyse.rolling_qual("nasdaq_01-01-16_01-06-19")
	pl.show()

def test_plot_qual():
	price, time = get_data()
	Analyse().plot_qual('nasdaq_01-01-16_01-06-19', price, time)

def test_epsilon(drawup=True):
	price, time = get_data()
	
	analyse = Analyse()
	t_peaks = analyse.epsilon(price, time, e0=2.55, w=35, drawup=drawup)
	if t_peaks is not None:
		print 'Peak Time'
		print t_peaks
		analyse.plot_peak(price, time, t_peaks)
		pl.show()
	else:
		print 'No peak found'	
	

#test_routine()
#test_rolling_qual()
#test_plot_qual()
#test_epsilon(False)
#Analyse().test_lppls()
#Analyse().generic_test_lppls(0., 10.)
#Analyse().generic_test_bm(n=20)
Analyse().compare_method(true_lppls=False)
















