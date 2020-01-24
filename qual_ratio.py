import numpy as np
import  matplotlib.pyplot as pl

from get_data import import_qual_data, match_data

qual_0, test_0, type_index, qual, test, method_index, rep_index, type_of_data, method, rep, name = import_qual_data()

def ratio(result, target):
	good = np.where(target == result, 1, 0)
	bad = np.where(target != result, 1, 0)
	positive = np.where(result != 0, 1, 0)
	negative = np.where(result == 0, 1, 0)
	true_positive = good & positive
	true_negative = good & negative
	false_positive = bad & positive
	false_negative = bad & negative
	print np.mean(np.where(target!=0, 1, 0))
	
	return np.mean(true_positive), np.mean(true_negative), np.mean(false_positive), np.mean(false_negative)

def plot_qual_ratio_per_method(method_index=method_index, rep_index=rep_index, test_0=test_0, test=test, qual=qual, qual_0=qual_0, title = 'Categorization ratio per methods', name = 'ratio_per_method'):
	fig, ax = pl.subplots()
	pl.title(title)
	index = np.arange(len(method))
	width = 1./5
	pos = np.arange(-5*0.5,5*0.5)*width + width
	to_label = True
	for n in range(len(method)):
		q, q_0, _, _ = match_data(method_index[n], rep_index[0], test_0, test, qual, qual_0)
		tp, tn, fp, fn = ratio(q, q_0)
		if to_label:
			pl.bar(index[n]+pos[0], 100*tp, width, alpha=1., color = 'b', label = 'true positive')
			pl.bar(index[n]+pos[1], 100*tn, width, alpha=1., color = 'g', label = 'true negative')
			pl.bar(index[n]+pos[2], 100*fp, width, alpha=1., color = 'r', label = 'false positive')
			pl.bar(index[n]+pos[3], 100*fn, width, alpha=1., color = 'k', label = 'false negative')
		else:
			pl.bar(index[n]+pos[0], 100*tp, width, alpha=1., color = 'b')
			pl.bar(index[n]+pos[1], 100*tn, width, alpha=1., color = 'g')
			pl.bar(index[n]+pos[2], 100*fp, width, alpha=1., color = 'r')
			pl.bar(index[n]+pos[3], 100*fn, width, alpha=1., color = 'k')
		pl.text(index[n]+pos[0], 100*tp, str(round(100*tp,2))+'%', ha='center', va='bottom')
		pl.text(index[n]+pos[1], 100*tn, str(round(100*tn,2))+'%', ha='center', va='bottom')
		pl.text(index[n]+pos[2], 100*fp, str(round(100*fp,2))+'%', ha='center', va='bottom')
		pl.text(index[n]+pos[3], 100*fn, str(round(100*fn,2))+'%', ha='center', va='bottom')
		to_label = False
	pl.xticks(index, method)
	pl.ylabel('ratio %')
	pl.legend()
	ax.minorticks_on()
	pl.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.5)
	pl.grid(which='minor', linestyle='-', linewidth='0.25', color='black', alpha=0.5)
	dpi = fig.get_dpi()
	h, w = fig.get_size_inches()
	fig.set_size_inches(h*2, w*2)
	fig.savefig('figure/'+name+'.png')

def plot_qual_ratio_per_type_of_data():
	for n in range(len(type_of_data)):
		plot_qual_ratio_per_method(method_index, rep_index, test_0[type_index[n]], test, qual, qual_0[type_index[n]], 'Categorization for '+ type_of_data[n] + ' (' + str(round(100.*len(test_0[type_index[n]])/len(test_0),1)) + '% of the test)', type_of_data[n].replace(' ', '_'))
	

plot_qual_ratio_per_method()
plot_qual_ratio_per_type_of_data()
pl.show()











