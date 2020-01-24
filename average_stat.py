import numpy as np
import pylab as pl

from get_data import import_data, match_data

qual_0, test_0, tc_0, type_index, qual, test, time, sse, tc, method_index, rep_index, type_of_data, method, rep, name = import_data()

def categorization(result, target, time):
	good = np.where(result == target)[0]
	bad = np.where(result != target)[0]
	positive = np.where(target != 0)[0]
	negative = np.where(target == 0)[0]

	gp = np.intersect1d(good, positive)
	gn = np.intersect1d(good, negative)
	bp = np.intersect1d(bad, positive)
	bn = np.intersect1d(bad, negative)

	time_gp = time[gp].mean()
	time_gn = time[gn].mean()
	time_bp = time[bp].mean()
	time_bn = time[bn].mean()

	sig_gp = time[gp].std()
	sig_gn = time[gn].std()
	sig_bp = time[bp].std()
	sig_bn = time[bn].std()
	
	return time_gp, time_gn, time_bp, time_bn, sig_gp, sig_gn, sig_bp, sig_bn

def dev_tc(result, target, T):
	good = np.where(result == target)[0]
	bad = np.where(result != target)[0]
	positive = np.where(target != 0)[0]

	gp = np.intersect1d(good, positive)
	bp = np.intersect1d(bad, positive)

	T_gp = T[gp].mean()
	T_bp = T[bp].mean()

	std_gp = T[gp].std()
	std_bp = T[bp].std()

	return T_gp, T_bp, std_gp, std_bp

def plot_average_per_method(obs, obs_name, obs_unity = ''):
	fig, ax = pl.subplots()
	pl.title('Average ' + obs_name + ' per methods')
	index = np.arange(len(method))
	width = 1./5
	pos = np.arange(-5*0.5,5*0.5)*width + width
	to_label = True
	for n in range(len(method)):
		q, q_0, i, g = match_data(method_index[n], rep_index[0], test_0, test, qual, qual_0)
		obs_gp, obs_gn, obs_bp, obs_bn, sig_gp, sig_gn, sig_bp, sig_bn = categorization(q, q_0, obs[i][g])
		if to_label:
			pl.bar(index[n]+pos[0], obs_gp, width, yerr=sig_gp, alpha=1., color = 'r', label = 'good positive')
			pl.bar(index[n]+pos[1], obs_gn, width, yerr=sig_gn, alpha=1., color = 'k', label = 'good negative')
			pl.bar(index[n]+pos[2], obs_bp, width, yerr=sig_bp, alpha=1., color = 'b', label = 'bad positive')
			pl.bar(index[n]+pos[3], obs_bn, width, yerr=sig_bn, alpha=1., color = 'g', label = 'bad negative')
		else:
			pl.bar(index[n]+pos[0], obs_gp, width, yerr=sig_gp, alpha=1., color = 'r')
			pl.bar(index[n]+pos[1], obs_gn, width, yerr=sig_gn, alpha=1., color = 'k')
			pl.bar(index[n]+pos[2], obs_bp, width, yerr=sig_bp, alpha=1., color = 'b')
			pl.bar(index[n]+pos[3], obs_bn, width, yerr=sig_bn, alpha=1., color = 'g')
		pl.text(index[n]+pos[0], obs_gp+sig_gp, str(round(obs_gp,1))+'$\pm$'+str(round(sig_gp, 1))+obs_unity, ha='center', va='bottom')
		pl.text(index[n]+pos[1], obs_gn+sig_gn, str(round(obs_gn,1))+'$\pm$'+str(round(sig_gn, 1))+obs_unity, ha='center', va='bottom')
		pl.text(index[n]+pos[2], obs_bp+sig_bp, str(round(obs_bp,1))+'$\pm$'+str(round(sig_bp, 1))+obs_unity, ha='center', va='bottom')
		pl.text(index[n]+pos[3], obs_bn+sig_bn, str(round(obs_bn,1))+'$\pm$'+str(round(sig_bn, 1))+obs_unity, ha='center', va='bottom')
		to_label = False
	pl.xticks(index, method)
	pl.ylabel('<' + obs_name + '> ' + obs_unity)
	pl.legend()
	ax.minorticks_on()
	pl.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.5)
	pl.grid(which='minor', linestyle='-', linewidth='0.25', color='black', alpha=0.5)
	dpi = fig.get_dpi()
	h, w = fig.get_size_inches()
	fig.set_size_inches(h*2, w*2)
	fig.savefig('figure/'+ obs_name +'_per_method.png')

def plot_average_tc_deviation_per_method():
	fig, ax = pl.subplots()
	pl.title('Average t$_c$ - t$_{c0}$ per methods')
	index = np.arange(len(method))
	width = 1./3
	pos = np.arange(-3*0.5,3*0.5)*width + width
	to_label = True
	for n in range(len(method)):
		q, q_0, i, g = match_data(method_index[n], rep_index[0], test_0, test, qual, qual_0)
		T_gp, T_bp, std_gp, std_bp = dev_tc(q, q_0, tc[i][g] - tc_0[g])
		if to_label:
			pl.bar(index[n]+pos[0], T_gp, width, yerr=std_gp, alpha=1., color = 'b', label = 'good positive')
			pl.bar(index[n]+pos[1], T_bp, width, yerr=std_bp, alpha=1., color = 'g', label = 'bad_positive')
		else:
			pl.bar(index[n]+pos[0], T_gp, width, yerr=std_gp, alpha=1., color = 'b')
			pl.bar(index[n]+pos[1], T_bp, width, yerr=std_bp, alpha=1., color = 'g')
		pl.text(index[n]+pos[0], T_gp+std_gp, str(round(T_gp,1))+'$\pm$'+str(round(std_gp, 1)), ha='center', va='bottom')
		pl.text(index[n]+pos[1], T_bp+std_bp, str(round(T_bp,1))+'$\pm$'+str(round(std_bp, 1)), ha='center', va='bottom')
		to_label = False
	pl.xticks(index, method)
	pl.ylabel('<t$_c$-t$_{c0}$>')
	pl.legend()
	ax.minorticks_on()
	pl.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.5)
	pl.grid(which='minor', linestyle='-', linewidth='0.25', color='black', alpha=0.5)
	dpi = fig.get_dpi()
	h, w = fig.get_size_inches()
	fig.set_size_inches(h*2, w*2)
	fig.savefig('figure/tc_per_method.png')

plot_average_per_method(time, 'time', 's')
plot_average_per_method(sse, 'SSE')
plot_average_tc_deviation_per_method()
pl.show()



