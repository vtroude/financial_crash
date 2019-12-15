import numpy as np
from lppls import LPPLS
import pandas as pd
import pandas_datareader as pdr
import datetime
import pylab as pl

class Analyse:
	''' A Dynamic Fit Routine '''

	def __init__(self): pass

	def qual_type(self, b):
		if b>0:
			return -1
		elif b<0:
			return 1

	def qual(self, m, tc, w, sse, b, tc0, tc1):
		a = (m>0 and m<1)
		a = (a and tc>tc0 and tc<tc1)
		a = (a and w>4 and w<25)
		a = (a and sse is not 'nan')
		if a:
			return self.qual_type(b)
		else:
			return 0

	def routine(self, price, time, N0, N1, name):
		I = range(N1+1, len(time)+1, 100)
		N = range(N0, N1, 100)
		model = LPPLS()
		t1, t2 = [], []
		M, Tc, W = [], [], []
		A, B, C1, C2 = [], [], [], []
		SSE, qual = [], []
		for i in I:
			for n in N:
				print '\t t in [',time[i-n],',',time[i-1],']'
				print
				tc0 = np.maximum(-60, -n*0.5)+time[i-1]
				tc1 = np.minimum(252, n*0.5)+time[i-1]
				tc_guess = (tc1-tc0+1)*0.5+tc0
				t1.append(time[i-n])
				t2.append(time[i-1])
				m, tc, w, a, b, c1, c2, sse = model.fit(price[i-n:i], time[i-n:i], tc_guess=tc_guess, tc0=tc0, tc1=tc1, rep=2, descent='steepest', precision=10.**(-16))
				M.append(m)
				Tc.append(tc)
				W.append(w)
				A.append(a)
				B.append(b)
				C1.append(c1)
				C2.append(c2)
				SSE.append(sse)
				qual.append(self.qual(m, tc, w, sse, b, tc0, tc1))
		raw_data = {'t1': t1, 't2': t2, 'm': M, 'tc': Tc, 'w': W, 'A': A, 'B': B, 'C1': C1, 'C2': C2, 'SSE': SSE, 'qual': qual}
		columns = ['t1', 't2', 'm', 'tc', 'w', 'A', 'B', 'C1', 'C2', 'SSE', 'qual']
		df = pd.DataFrame(raw_data)
		df.to_csv(name+'.csv')

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
#test_epsilon(False)


















