from abc import ABCMeta, abstractmethod
from scipy.optimize import minimize
import numpy as np
import time as tm

class Model:
	''' Abstract class: Define a model and an optimization procedure '''

	__metaclass__ = ABCMeta

	def in_bounds(self, para, bounds):
		for n in range(len(para)):
			if para[n]<bounds[n][0] or para[n]>bounds[n][1]:
				return False

		return True

##############################################################################
	''' Methods which define the model and the cost function '''

	@abstractmethod
	def model(self, time, linear=None, non_linear=None): pass
	
	def delta(self, objective, time, linear=None, non_linear=None):
		return objective-self.model(time, linear, non_linear)

	def error(self, objective, time, linear=None, non_linear=None):
	# Define the error of the model with respects to the objective 
	# SSE = ((objective-model)**2)/(2*N), N: number of data point
		delta = self.delta(objective, time, linear, non_linear)

		return delta.dot(delta)*0.5/len(objective)

	def log_likelihood(self, objective, time, linear=None, non_linear=None):
	# Log-likelyhood: LL~(1/2)*ln(SSE)
		return  np.log(self.error(objective, time, linear, non_linear))*0.5

	def to_minimize(self, non_linear, objective, time, linear=None):
	# Function to minimize to fit to the model
		return self.log_likelihood(objective, time, linear, non_linear)

##################################################################################
	''' Linear optimization methods '''

	@abstractmethod
	def build_linear_system(self, objective, time, non_linear=None): pass

	def linear_optimization(self, objective, time, non_linear=None):
	# Return the solution of the linear system
		A, b = self.build_linear_system(objective, time, non_linear)

		return np.linalg.solve(A,b).reshape(-1)

#################################################################################
	''' Grid search for linear optimization '''

	@abstractmethod
	def get_grid(self, non_linear_bounds, step=10): pass

	def grid_search(self, objective, time, non_linear_bounds, step=10):
	# Define a grid search on the non-linear parameters to optimize the linear ones
		non_linear = self.get_grid(non_linear_bounds, step)
		N_point = len(non_linear)
		linear = []
		SSE = np.zeros(N_point)
		for n in range(N_point):
			linear.append(self.linear_optimization(objective, time, non_linear[n]))
			SSE[n] = self.error(objective, time, linear[-1], non_linear[n])
		argmin = np.argmin(SSE)

		return linear[argmin], non_linear[argmin]

#################################################################################
	''' Methods which compute the gradient of the function to minimize '''

	@abstractmethod
	def grad_model(self, time, linear=None, non_linear=None): pass

	def grad_error(self, objective, time, linear=None, non_linear=None):
	# Return the gradient of the SSE
		delta = self.delta(objective, time, linear, non_linear)
		N = len(objective)
		
		return -delta.dot(self.grad_model(time, linear, non_linear))/N

	def grad_ll(self, objective, time, linear=None, non_linear=None):
	# Return the gradient of the log-likelihood
		SSE_2 = 2*self.error(objective, time, linear, non_linear)

		return self.grad_error(objective, time, linear, non_linear)/SSE_2

	def jac(self, non_linear, objective, time, linear=None):
	# Jacobian of the function to minimize
		return self.grad_ll(objective, time, linear, non_linear)

################################################################################
	''' Methods which compute the hessian matrix of the function to minimize '''

	@abstractmethod
	def hess_model(self, time, linear=None, non_linear=None): pass

	def hess_error(self, objective, time, linear=None, non_linear=None):
	# Return hessian matrix of the SSE
		jac_model = self.grad_model(time, linear, non_linear)
		hess_model = self.hess_model(time, linear, non_linear)
		delta = self.delta(objective, time, linear, non_linear)
		N = len(objective)

		return np.dot(jac_model.T, jac_model)/N-hess_model.dot(delta)/N

	def hess_ll(self, objective, time, linear=None, non_linear=None):
	# Return hessian matrix of the log-likelihood
		SSE = self.error(objective, time, linear, non_linear)
		grad_SSE = self.grad_error(objective, time, linear, non_linear)
		hess_SSE = self.hess_error(objective, time, linear, non_linear)
		
		return hess_SSE/SSE/2 - grad_SSE.dot(grad_SSE)/2/SSE/SSE

	def hess(self, non_linear, objective, time, linear=None):
	# Hessian matrix of the function to minimize
		return self.hess_ll(objective, time, linear, non_linear)

################################################################################
	''' Non-linear optimization methods '''

	def non_linear_optimization(self, objective, time, linear, non_linear_guess, bounds, method='Nelder-Mead', tol=None, options=None):

		return minimize(self.to_minimize, non_linear_guess, args=(objective, time, linear), method=method, jac=self.jac, hess=self.hess, bounds=bounds, tol=tol, options=options).x

###############################################################################
	''' Optimization procedure '''

	def fit(self, objective, time, non_linear_bounds=None, method='Nelder-Mead', tol=None, options=None):

		if non_linear_bounds is None:
			linear = self.linear_optimization(objective, time)

			sse = self.error(objective, time, linear)
			self.print_parameter(linear, None, sse)

			return linear, None, self.error(objective, time, linear)
		else:
			linear, non_linear_guess = self.grid_search(objective, time, non_linear_bounds)
			non_linear = self.non_linear_optimization(objective, time, linear, non_linear_guess, non_linear_bounds, method, tol, options)
			linear = self.linear_optimization(objective, time, non_linear)

			if not self.in_bounds(non_linear, non_linear_bounds):
				print '\t---> Out of bounds'

			sse = self.error(objective, time, linear, non_linear)
			self.print_parameter(linear, non_linear, sse)

			return linear, non_linear, self.error(objective, time, linear, non_linear)

	def test_fit(self, objective, time, non_linear_bounds=None, method='Nelder-Mead', rep = 10):
		Linear, Non_Linear, SSE = [], [], []
		t = []
		t.append(tm.time())
		linear, non_linear, sse = self.fit(objective, time, non_linear_bounds, method)
		t[-1] = tm.time() - t[-1]
		Linear.append(linear)
		Non_Linear.append(non_linear)
		SSE.append(sse)
		if True:	#not self.in_bounds(non_linear, non_linear_bounds):
			return Linear, Non_Linear, SSE, True, t
		else:
			for n in range(1, rep):
				t.append(tm.time())
				Non_Linear.append(self.non_linear_optimization(objective, time, Linear[-1], Non_Linear[-1], non_linear_bounds, method))
				Linear.append(self.linear_optimization(objective, time, Non_Linear[-1]))
				t[-1] = t[-2] + tm.time() - t[-1]
				SSE.append(self.error(objective, time, Linear[-1], Non_Linear[-1]))
				
				if not self.in_bounds(Non_Linear[-1], non_linear_bounds):
					print '\t---> Out of bounds'
					self.print_parameter(Linear[-1], Non_Linear[-1], SSE[-1])

					return Linear, Non_Linear, SSE, True, t

		return Linear, Non_linear, SSE, False, t

####################################################################################
	''' Print methods '''

	@abstractmethod
	def print_parameter(self, linear = None, non_linear = None, sse = None): pass











######################################################################################################







class LPPLS(Model):
	''' Define LPPLS model '''

	def __init_(self): pass

########################################################################################
	''' Personalized methods '''

	def power_law(self, t, m, tc):
		return np.abs(t-tc)**(-m)

	def log_angle(self, t, tc, w):
		return w*np.log(np.abs(t-tc))

	def C(self, linear):
		return np.sqrt(linear[2]*linear[2] + linear[3]*linear[3])

	def damping(self, linear, non_linear):
		return non_linear[0]/non_linear[2]*np.abs(linear[1]/self.C(linear))

	def oscillation(self, time, non_linear):
		return non_linear[2]/2/np.pi*np.log(np.abs((non_linear[1]-time[0])/(non_linear[1]-time[-1])))

	def tc_bounds(self, t):
		N = len(t)
		tc0 = np.maximum(-60, -N*0.5)+t[-1]
		tc1 = np.minimum(252, N*0.5)+t[-1]

		return tc0, tc1

	def generate_model(self, time, non_linear_bounds = None, show_parameter = False):
		A = np.random.rand()*2.-1.
		B = np.random.rand()*2.-1.
		C1 = np.random.rand()*0.2-0.1
		C2 = np.random.rand()*0.2-0.1
		if non_linear_bounds is None:
			m = np.random.rand()
			tc0, tc1 = self.tc_bounds(time)
			tc = np.random.rand()*(tc1-tc0) + tc0
			w = np.random.rand()*(25-4) + 4
		else:
			m = np.random.rand()*(non_linear_bounds[0][1]-non_linear_bounds[0][0]) + non_linear_bounds[0][0]
			tc = np.random.rand()*(non_linear_bounds[1][1]-non_linear_bounds[1][0]) + non_linear_bounds[1][0]
			w = np.random.rand()*(non_linear_bounds[2][1]-non_linear_bounds[2][0]) + non_linear_bounds[2][0]

		if show_parameter:
			return self.model(time, [A, B, C1, C2], [m, tc, w]), np.array([A, B, C1, C2]), np.array([m, tc, w])
		else:
			return self.model(time, [A, B, C1, C2], [m, tc, w])

	def generate_qualified_model(self, time, show_parameter = False):
		A = np.random.rand()*2.-1.
		B = np.random.rand()*2.-1.
		m = np.random.rand()
		w = np.random.rand()*(25-4) + 4
		C = np.sqrt(np.random.rand()*4.*m*m/w/w*B*B)
		C1 = np.random.rand()*2.*C-C
		C2 = (np.random.randint(2)*2-1)*np.sqrt(C*C-C1*C1)
		tc0, tc1 = self.tc_bounds(time)
		if np.abs(C/B)>=0.05:
			Omega = np.exp(2*np.pi/w)
			tc0 = np.maximum(tc0, (Omega*time[-1]-time[0])/(Omega-1))
			tc1 = np.minimum(tc1, (Omega*time[-1]+time[0])/(Omega+1))
		tc = np.random.rand()*(tc1-tc0) + tc0

		if show_parameter:
			return self.model(time, [A, B, C1, C2], [m, tc, w]), np.array([A, B, C1, C2]), np.array([m, tc, w])
		else:
			return self.model(time, [A, B, C1, C2], [m, tc, w])

	def qual(self, time, linear, non_linear):
		if None in linear or None in non_linear or 'nan' in linear or 'nan' in non_linear:
			return 0
		else: 
			m = (non_linear[0]>=0. and non_linear[0]<=1.)
			tc0, tc1 = self.tc_bounds(time)
			tc = (non_linear[1]>=tc0 and non_linear[1]<=tc1)
			w = (non_linear[2]>=4. and non_linear[2]<+25.)
			D = self.damping(linear, non_linear)
			D = (D>=0.5)
			C = self.C(linear)
			if np.abs(C/linear[1])>=0.05:
				O = self.oscillation(time, non_linear)
				O = (O>=2.5)
			else:
				O = True
		
			qual = m and tc and w and D and O
			if qual:
				if linear[1]>0:
					return -1
				else:
					return 1
			else:
				return 0

########################################################################################
	''' Override Model abstract methods '''

	def model(self, time, linear=[0., 1., 0., 0.], non_linear=[1., 0., 0.]):
	# Define LPPLS model with linear = [A, B, C1, C2] and non-linear=[m, tc, w]
	# LPPLS = A + |tc-t|**(-m)*(B + C1*cos(w*ln|tc-t|) + C2*sin(w*ln|tc-t|))
		T = self.power_law(time, non_linear[0], non_linear[1])
		phi = self.log_angle(time, non_linear[1], non_linear[2])

		return linear[0] + T*(linear[1] + linear[2]*np.cos(phi) + linear[3]*np.sin(phi))

	def build_linear_system(self, objective, time, non_linear=[1., 0., 0.]):
	# Build a linear system to optimize the SSE with respects to: A+B*f+C1*g+C2*h
		phi = self.log_angle(time, non_linear[1], non_linear[2])
		# Define the gradiant [N, f, g, h] with respact to A, B, C1, C2
		N = len(objective)
		f = self.power_law(time, non_linear[0], non_linear[1])
		g = f*np.cos(phi)
		h = f*np.sin(phi)
		# Define matrix element
		a1 = np.sum(f)
		a2 = np.sum(g)
		a3 = np.sum(h)
		a4 = f.dot(f)
		a5 = f.dot(g)
		a6 = f.dot(h)
		a7 = g.dot(g)
		a8 = g.dot(h)
		a9 = h.dot(h)
		# Define linear system Ax=b where x = [A, B, C1, C2]
		A = np.array([[N, a1, a2, a3], [a1, a4, a5, a6], [a2, a5, a7, a8], [a3, a6, a8, a9]])
		b = np.array([[np.sum(objective)], [objective.dot(f)], [objective.dot(g)], [objective.dot(h)]])

		return A, b
		
	def get_grid(self, non_linear_bounds, step=10):
	# Define a grid on the non_linear parameters for the grid search
	# non_linear_bounds = ((m0, m1), (tc0, tc1), (w0, w1))
		eps = 10.**(-6)
		m = np.linspace(non_linear_bounds[0][0]+eps, non_linear_bounds[0][1]-eps, step)
		tc = np.linspace(non_linear_bounds[1][0]+eps, non_linear_bounds[1][1]-eps, step)
		w = np.linspace(non_linear_bounds[2][0]+eps, non_linear_bounds[2][1]-eps, step)

		return np.array(np.meshgrid(m, tc, w)).T.reshape(-1, 3)

	def grad_model(self, time, linear=[0., 1., 0., 0.], non_linear=[1., 0., 0.], give_all=False):
		f1 = self.model(time, [0., linear[1], linear[2], linear[3]], non_linear).reshape(-1, 1)	
		f2 = self.model(time, [0., 0., linear[3], -linear[2]], non_linear).reshape(-1, 1)
		T = (non_linear[1]-time).reshape(-1, 1)
		log = np.log(np.abs(T)).reshape(-1, 1)

		dm = -log*f1
		dtc = -non_linear[0]*f1/T + non_linear[2]*f2/T
		dw = log*f2
		
		if give_all:
			return np.concatenate((dm, dtc, dw), axis=1), f1, f2, T, log
		else:
			return np.concatenate((dm, dtc, dw), axis=1)

	def hess_model(self, time, linear=[0., 1., 0., 0.], non_linear=[1., 0., 0.]):
		jac, f1, f2, T, log = self.grad_model(time, linear, non_linear, True)
		f3 = self.model(time, [0., 0., linear[2], linear[3]], non_linear).reshape(-1, 1)

		d2m = (log*jac[:,0].reshape(-1,1)).reshape(-1)
		dmdtc = (-log*jac[:,1].reshape(-1,1) + f1/T).reshape(-1)
		dmdw = (-log*jac[:,2].reshape(-1,1)).reshape(-1)
		d2tc = (-(1+non_linear[0])*jac[:,1].reshape(-1,1)/T - non_linear[2]*f3/T/T).reshape(-1)
		dtcdw = (-non_linear[0]*jac[:,2].reshape(-1,1)/T - non_linear[2]*log*f3/T + f2/T).reshape(-1)
		d2w = (-log*log*f3).reshape(-1)
		
		return np.array([[d2m, dmdtc, dmdw], [dmdtc, d2tc, dtcdw], [dmdw, dtcdw, d2w]])

	def print_parameter(self, linear=None, non_linear=None, sse = None):
		print 'A = ', linear[0], '\tB = ', linear[1]
		print 'C1 = ', linear[2], '\tC2 = ', linear[3]
		print 'm = ', non_linear[0], '\ttc = ', non_linear[1], '\tw = ', non_linear[2]
		if sse is not None:
			print 'SSE = ', sse
		print

		



















		




