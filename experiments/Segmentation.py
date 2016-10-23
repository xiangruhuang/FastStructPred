import sys
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from setting import *
from utils import *

if (len(sys.argv) < 3):
	exit_with_help()

output_fig, labels, filelist = parse_args(sys.argv)

#fin_ = [open(sys.argv[i], 'r') for i in sys.argc)]

ref_y = -252.391
ref_y = ref_y + abs(ref_y)*eps
fig, ax = plt.subplots()
for i in range(len(filelist)):
	data = numpy.array(readData(filelist[i]))
	print data[-1][1]
	data[:][:, 1] = [(ref_y - x)/abs(ref_y) for x in data[:][:, 1]]
	data = process(data)
	print labels[i], 'color=', colors[labels[i]], 'linestyle=', linestyles[labels[i]]
	algolabel = labels[i]
	if (algolabel == "Soft-BCFW"):
		algolabel = r"SoftBCFW-$\lambda$=1"
	if (algolabel == "smoothMSD"):
		algolabel = r"SmoothMSD-$\gamma=10^{-4}$"
	ax.plot(data[:][:, 0], data[:][:, 1], color=colors[labels[i]], label=algolabel, linestyle=linestyles[labels[i]], linewidth=linewidths[labels[i]], marker=markers[labels[i]])# dashes=dash[labels[i]])

#ax.plot(numpy.linspace(0, 1e5, 1000), [ref_y for i in range(1000)], 'k--', linewidth='0.5', label='OPT')

legend = ax.legend(loc=(0.65, 0.5), shadow=True, fontsize=11)
#frame = legend.get_frame()
#frame.set_facecolor('0.90')
plt.yscale('log')
plt.xscale('log')
plt.title('Segmentation', fontsize=title_fontsize)
plt.xlabel('Time', fontsize=25);
plt.ylabel('Relative Primal Gap (decoded)', fontsize=25);
plt.axis([0, 1e3, 0, 1])
plt.savefig(output_fig+'.eps')
