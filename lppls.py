import numpy as np
import pylab as pl
import pandas_datareader as pdr
import datetime

class LPPLS:
	''' LPPLS model using the steepest descent as an optimization procedure '''

	def __init__(self): pass

	def in_bound(self, m, tc, w, m0, m1, tc0, tc1, w0, w1):
		a = (m>m0 and m<m1)
		a = (a and tc>tc0 and tc<tc1)
		a = (a and w>w0 and w<w1)
		
		return a

####################################################################"
	''' Methods wich define the model and the cost function '''

	def lppls(self, t,m,tc,w,a,b,c,d):		# LPPLS formula
		return a+np.abs(tc-t)**m*(b+c*np.cos(w*np.log(np.abs(tc-t)))+d*np.sin(w*np.log(np.abs(tc-t))))

	def delta(self, y, t, a, b, c, d, m, tc, w):		# Error = (delta**2)/2
		return y-self.lppls(t,m,tc,w,a,b,c,d)

	def error(self, y, t, a, b, c, d, m, tc, w):
		delta = self.delta(y, t, a, b, c, d, m, tc, w)
		return (delta.dot(delta))*0.5/len(y)

	def log_likelihood(self, y, t, a, b, c, d, m, tc, w):
		return len(y)*0.5*np.log(self.error(y, t, a, b, c, d, m, tc, w))

########################################################################
	''' Setting methods '''

	def set_parameters(self, t, m_guess, tc_guess, w_guess):
		if tc_guess is None: tc_guess = float(t[-1]-t[0]+1)/2

		return m_guess, tc_guess, w_guess

	def set_time_constraint(self, t, tc0, tc1):
		if tc0 is None: tc0 = t[0]
		if tc1 is None: tc1 = t[-1]

		return tc0, tc1

	def initialize_non_linear_opt(self, t, m_guess, tc_guess, w_guess, m0, m1, tc0, tc1, w0, w1):

		m, tc, w = self.set_parameters(t, m_guess, tc_guess, w_guess)

		para = np.array([m, tc, w])
		lagrange_1, lagrange_2 = np.ones(3), np.ones(3)
		
		tc0, tc1 = self.set_time_constraint(t, tc0, tc1)

		cons_0 = np.array([m0, tc0, w0])
		cons_1 = np.array([m1, tc1, w1])

		V, S = np.zeros(3), np.zeros(3)
		Vl1, Sl1 = np.zeros(3), np.zeros(3)
		Vl2, Sl2 = np.zeros(3), np.zeros(3)

		beta_1 = 0.9
		beta_2 = 0.999
		epsilon = 0.00000001

		return para, lagrange_1, lagrange_2, cons_0, cons_1, V, S, Vl1, Sl1, Vl2, Sl2, beta_1, beta_2, epsilon

##########################################################################
	''' Methods to obtain the required gradients '''

	def grad_Error(self, y, t, a, b, c, d, m, tc, w):
		delta = -self.delta(y, t, a, b, c, d, m, tc, w)
		delta_m = np.log(np.abs(tc-t))*self.lppls(t, m, tc, w, 0, b, c, d)
		delta_tc = np.abs(tc-t)**m/(tc-t)*self.lppls(t, m, tc, w, 0, m*b, d*w+c*m, d*m-c*w)
		delta_w = np.log(np.abs(tc-t))*self.lppls(t, m, tc, w, 0, 0, d, -c)

		return np.array([delta.dot(delta_m), delta.dot(delta_tc), delta.dot(delta_w)])

	def grad_lambda(self, phi, phi0, phi1):
		return -(phi-phi0)/(phi1-phi0)

	def grad_damping(self, para, damping_0):
		return -(para[0]-damping_0*para[-1])

	def add_constraint(self, grad, lambda_1, lambda_2, phi0, phi1):
		return grad+(lambda_2-lambda_1)/(phi1-phi0)

	def get_gradients(self, y, t, a, b, c, d, para, lambda_1, lambda_2, phi0, phi1):
		N = len(y)

		error = self.error(y, t, a, b, c, d, para[0], para[1], para[2])*N

		grad = self.grad_Error(y, t, a, b, c, d, para[0], para[1], para[2])
		
		grad = self.add_constraint(grad, lambda_1, lambda_2, phi0, phi1)/error
		
		grad_l1 = self.grad_lambda(para, phi0, phi1)/error
		grad_l2 = self.grad_lambda(para, phi1, phi0)/error

		return grad, grad_l1, grad_l2

##############################################################################
	''' Methods to obtain the Adam optimization procedure '''

	def update_adam(self, V, S, beta_1, beta_2, grad):
		return (beta_1*V+(1-beta_1)*grad), (beta_2*S+(1-beta_2)*grad*grad)
	
	def correct_adam(self, V, S, beta_1, beta_2, epsilon):
		return (V/(1.-beta_1))/np.sqrt(S/(1.-beta_2)+epsilon)

	def adam(self, grad, grad_l1, grad_l2, V, S, Vl1, Sl1, Vl2, Sl2, beta_1, beta_2, epsilon):
		V, S = self.update_adam(V, S, beta_1, beta_2, grad)
		Vl1, Sl1 = self.update_adam(Vl1, Sl1, beta_1, beta_2, grad_l1)
		Vl2, Sl2 = self.update_adam(Vl2, Sl2, beta_1, beta_2, grad_l2)
		
		grad = self.correct_adam(V, S, beta_1, beta_2, epsilon)
		grad_l1 = self.correct_adam(Vl1, Sl1, beta_1, beta_2, epsilon)
		grad_l2 = self.correct_adam(Vl2, Sl2, beta_1, beta_2, epsilon)

		return grad, grad_l1, grad_l2, V, S, Vl1, Sl1, Vl2, Sl2

#################################################################################
	''' Linear optimization part '''

	def build_linear_system(self, y, f, g, h):	# Build the Linear system for the linear 
		N = len(y)				# optimization part
		a1 = np.sum(f)
		a2 = np.sum(g)
		a3 = np.sum(h)
		a4 = f.dot(f)
		a5 = f.dot(g)
		a6 = f.dot(h)
		a7 = g.dot(g)
		a8 = g.dot(h)
		a9 = h.dot(h)
		
		A = np.array([[N, a1, a2, a3], [a1, a4, a5, a6], [a2, a5, a7, a8], [a3, a6, a8, a9]])
		b = np.array([[np.sum(y)], [y.dot(f)], [y.dot(g)], [y.dot(h)]])
		
		return A, b

	def linear_optimization(self, y, f, g, h):
		A, b = self.build_linear_system(y, f, g, h)
		
		return np.linalg.solve(A,b).reshape(-1)

	def define_grid(self, t, m0=0., m1=1., tc0=None, tc1=None, w0=2., w1=25., step=10):
		eps = 10.**(-8)
		dm = (w1-w0)/step
		m = np.linspace(m0+eps, m1-eps, step)
		tc0, tc1 = self.set_time_constraint(t, tc0, tc1)
		dtc = (tc1-tc0)/step
		tc = np.linspace(tc0+eps, tc1-eps, step)
		dw = (w1-w0)/step
		w = np.linspace(w0+eps, w1-eps, step)

		return m, tc, w

	def define_non_linear(self, t, m, tc, w):
		f = np.abs(tc-t)**m
		g = f*np.cos(w*np.log(np.abs(tc-t)))
		h = f*np.sin(w*np.log(np.abs(tc-t)))
		
		return f, g, h

	def grid_search(self, y, t,  m0=0., m1=1., tc0=None, tc1=None, w0=2., w1=25., step=10):
		m, tc, w = self.define_grid(t, m0, m1, tc0, tc1, w0, w1, step)
		Error = np.zeros(step**3)
		phi = []
		l = range(step)
		for k in l:
			for j in l:
				for i in l:
					f, g, h = self.define_non_linear(t, m[i], tc[j], w[k])
					phi.append(self.linear_optimization(y, f, g, h))
					Error[i+j*step+k*step*step] = self.error(y, t, phi[-1][0], phi[-1][1], phi[-1][2], phi[-1][3], m[i], tc[j], w[k])
		arg = np.argmin(Error)
		i = arg%step
		a = arg/step
		j = a%step
		k = a/step
		
		return phi[arg], m[i], tc[j], w[k]
		

###############################################################################
	''' Non-linear optimization procedure '''

	def update(self, para, lagrange_1, lagrange_2, grad, grad_l1, grad_l2, rate, cons_0, cons_1, epsilon):
		para = para - rate*grad
		lagrange_1 = lagrange_1 - rate*grad_l1
		lagrange_2 = lagrange_2 - rate*grad_l2

		#para = np.where(para<=cons_0, cons_0+epsilon, para)
		#para = np.where(para>=cons_1, cons_1-epsilon, para)

		return para, lagrange_1, lagrange_2


	def steepest_descent(self, y, t, a, b, c, d, para, lagrange_1, lagrange_2, cons_0, cons_1, V, S, Vl1, Sl1, Vl2, Sl2, beta_1, beta_2, epsilon, rate):
		
		grad, grad_l1, grad_l2 = self.get_gradients(y, t, a, b, c, d, para, lagrange_1, lagrange_2, cons_0, cons_1)

		Grad, Grad_l1, Grad_l2, V, S, Vl1, Sl1, Vl2, Sl2 = self.adam(grad, grad_l1, grad_l2, V, S, Vl1, Sl1, Vl2, Sl2, beta_1, beta_2, epsilon)

		para, lagrange_1, lagrange_2 = self.update(para, lagrange_1, lagrange_2, Grad, Grad_l1, Grad_l2, rate, cons_0, cons_1, epsilon)

		return para, lagrange_1, lagrange_2, V, S, Vl1, Sl1, Vl2, Sl2, grad, grad_l1, grad_l2


	def update_conjugate(self, grad, grad_0, D):
		#return grad - grad.dot(grad)/grad_0.dot(grad_0)*D
		return grad - np.maximum(0., grad.dot(grad-grad_0)/grad_0.dot(grad_0))*D
		#return grad + grad.dot(grad-grad_0)/D.dot(grad-grad_0)*D
		#return grad + grad.dot(grad)/D.dot(grad-grad_0)*D


	def conjugate_gradient(self, y, t, a, b, c, d, para, lagrange_1, lagrange_2, cons_0, cons_1, V, S, Vl1, Sl1, Vl2, Sl2, beta_1, beta_2, epsilon, rate, grad_0, grad_l1_0, grad_l2_0, D, Dl1, Dl2):
		
		grad, grad_l1, grad_l2 = self.get_gradients(y, t, a, b, c, d, para, lagrange_1, lagrange_2, cons_0, cons_1)
		
		D = self.update_conjugate(grad, grad_0, D)
		Dl1 = self.update_conjugate(grad_l1, grad_l1_0, Dl1)
		Dl2 = self.update_conjugate(grad_l2, grad_l2_0, Dl2)

		Grad, Grad_l1, Grad_l2, V, S, Vl1, Sl1, Vl2, Sl2 = self.adam(D, Dl1, Dl2, V, S, Vl1, Sl1, Vl2, Sl2, beta_1, beta_2, epsilon)

		para, lagrange_1, lagrange_2 = self.update(para, lagrange_1, lagrange_2, Grad, Grad_l1, Grad_l2, rate, cons_0, cons_1, epsilon)

		return para, lagrange_1, lagrange_2, V, S, Vl1, Sl1, Vl2, Sl2, grad, grad_l1, grad_l2, D, Dl1, Dl2


	def gradient_descent(self, y, t, a, b, c, d, para, lagrange_1, lagrange_2, cons_0, cons_1, V, S, Vl1, Sl1, Vl2, Sl2, beta_1, beta_2, epsilon, rate, max_iteration, precision):
		
		for n in range(max_iteration):
			step = para

			para, lagrange_1, lagrange_2, V, S, Vl1, Sl1, Vl2, Sl2, _, _, _ = self.steepest_descent(y, t, a, b, c, d, para, lagrange_1, lagrange_2, cons_0, cons_1, V, S, Vl1, Sl1, Vl2, Sl2, beta_1, beta_2, epsilon, rate)

			step = step - para
			if step.dot(step) <= precision: 
				print 'stop at ', n
				break

		return para[0], para[1], para[2]


	def conjugate_descent(self, y, t, a, b, c, d, para, lagrange_1, lagrange_2, cons_0, cons_1, V, S, Vl1, Sl1, Vl2, Sl2, beta_1, beta_2, epsilon, rate, max_iteration, precision):
		
		para, lagrange_1, lagrange_2, V, S, Vl1, Sl1, Vl2, Sl2, grad, grad_l1, grad_l2 = self.steepest_descent(y, t, a, b, c, d, para, lagrange_1, lagrange_2, cons_0, cons_1, V, S, Vl1, Sl1, Vl2, Sl2, beta_1, beta_2, epsilon, rate)

		D = grad
		Dl1 = grad_l1
		Dl2 = grad_l2

		for n in range(max_iteration-1):
			step = para

			para, lagrange_1, lagrange_2, V, S, Vl1, Sl1, Vl2, Sl2, grad, grad_l1, grad_l2, D, Dl1, Dl2 = self.conjugate_gradient(y, t, a, b, c, d, para, lagrange_1, lagrange_2, cons_0, cons_1, V, S, Vl1, Sl1, Vl2, Sl2, beta_1, beta_2, epsilon, rate, grad, grad_l1, grad_l2, D, Dl1, Dl2)

			step = step - para
			if step.dot(step) <= precision: 
				print 'stop at ', n
				break

		return para[0], para[1], para[2]


	def non_linear_optimization(self, y, t, a, b, c, d, m_guess=0.5, tc_guess=None, w_guess=15., rate=0.1, precision=10.**(-6), max_iteration=10000, m0=0., m1=1., tc0=None, tc1=None, w0=2., w1=25., descent='steepest'):

		para, lagrange_1, lagrange_2, cons_0, cons_1, V, S, Vl1, Sl1, Vl2, Sl2, beta_1, beta_2, epsilon = self.initialize_non_linear_opt(t, m_guess, tc_guess, w_guess, m0, m1, tc0, tc1, w0, w1)

		if descent == 'steepest':
			return self.gradient_descent(y, t, a, b, c, d, para, lagrange_1, lagrange_2, cons_0, cons_1, V, S, Vl1, Sl1, Vl2, Sl2, beta_1, beta_2, epsilon, rate, max_iteration, precision)
		elif descent == 'conjugate':
			return self.conjugate_descent(y, t, a, b, c, d, para, lagrange_1, lagrange_2, cons_0, cons_1, V, S, Vl1, Sl1, Vl2, Sl2, beta_1, beta_2, epsilon, rate, max_iteration, precision)

#########################################################################"
	''' Fitting, plot and print methods '''

	def fit(self, price, t, m_guess=0.5, tc_guess=None, w_guess=15., rep=2, m0=0., m1=1., tc0=None, tc1=None, w0=2., w1=25., descent='steepest', precision=10.**(-6), grid_search=True):

		m_guess, tc_guess, w_guess = self.set_parameters(t, m_guess, tc_guess, w_guess)
		y = np.log(price)		
	
		for n in range(rep):
			if grid_search and n==0:
				phi, m_guess, tc_guess, w_guess = self.grid_search(y, t, m0, m1, tc0, tc1, w0, w1)
			else:
				if tc_guess in t:
					tc_guess = tc_guess+10.**(-8)
				f, g, h = self.define_non_linear(t, m_guess, tc_guess, w_guess)
				phi = self.linear_optimization(y, f, g, h)
			m, tc, w = self.non_linear_optimization(y, t, phi[0], phi[1], phi[2], phi[3], m_guess=m_guess, tc_guess=tc_guess, w_guess=w_guess, m0=m0, m1=m1, tc0=tc0, tc1=tc1, w0=w0, w1=w1, descent=descent, precision=precision)
			if not self.in_bound(m, tc, w, m0, m1, tc0, tc1, w0, w1):
				print '\t--> Break'
				break
			m_guess, tc_guess, w_guess = m, tc, w
		
		self.print_para(y, t, m, tc, w, phi)

		return m, tc, w, phi[0], phi[1], phi[2], phi[3], self.error(y, t, phi[0], phi[1], phi[2], phi[3], m, tc, w)

	def plot(self, price,t,m,tc,w,a,b,c,d, name):
		pl.title('Lppls fit to ' + name)
		pl.plot(t, np.log(price), label=name)
		pl.plot(t, self.lppls(t,m,tc,w,a,b,c,d), label='LPPLS')
		pl.ylabel('ln(price) []')
		pl.xlabel('time [days]')
		pl.legend()
		pl.show()

	def print_para(self, y, t, m, tc, w, phi):
		print 'A = ', phi[0], '\tB = ', phi[1], '\nC1 = ', phi[2], '\tC2 = ', phi[3]
		print 'm = ', m, '\ttc = ', tc, '\tw = ', w
		print 'SSE = ', self.error(y, t, phi[0], phi[1], phi[2], phi[3], m, tc, w)
		print


#######################################################
''' Test LPPLS '''

def test():	
	nasdaq = pdr.get_data_yahoo('^IXIC',start=datetime.datetime(2016,01,1),end=datetime.datetime(2019, 06,1))

	N = len(nasdaq['Close'])
	time = np.array((nasdaq['Close'].index-nasdaq['Close'].index[0]).days).reshape(-1)
	price = np.zeros(N)
	for n in range(N): price[n]=nasdaq['Close'][n]

	model = LPPLS()
	m, tc, w, a, b, c, d, sse = model.fit(price, time, rep=2, descent='conjugate', precision=10.**(-16))
	model.plot(price, time, m, tc, w, a, b, c, d, 'nasdaq')

#test()
























		
		
		
