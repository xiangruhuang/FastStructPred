import sys
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from setting import *
from utils import *
import os.path

if (len(sys.argv) < 3):
	exit_with_help()

output_fig, labels, filelist = parse_args(sys.argv)

#fin_ = [open(sys.argv[i], 'r') for i in sys.argc)]

ref_y = 1711
fig, ax = plt.subplots()
for i in range(len(filelist)):
	if not os.path.exists(filelist[i]):
		continue
	data = numpy.array(readData(filelist[i]))
	print data[-1][1]
	#data[:][:, 1] = [(ref_y - x)/ref_y for x in data[:][:, 1]]
	print labels[i], 'color=', colors[labels[i]], 'linestyle=', linestyles[labels[i]]
	algolabel=labels[i]
	if (algolabel == "Soft-BCFW"):
		algolabel = r"Soft-BCFW-$\rho$=1"
	ax.plot(data[:][:, 0], data[:][:, 1], color=colors[labels[i]], label=algolabel, linestyle=linestyles[labels[i]], linewidth=linewidths[labels[i]])

plt.plot([0, 1e5], [1711, 1711], 'k--', linewidth=0.01)
legend = ax.legend(loc='upper right', shadow=True, fontsize=16)
#frame = legend.get_frame()
#frame.set_facecolor('0.90')
#plt.yscale('')
plt.xscale('log')
plt.title('107network', fontsize=40)
plt.xlabel('Time', fontsize=25);
plt.ylabel('Relative P-Obj Gap(decoded)', fontsize=25)
plt.axis([0, 1e5, 1650, 1750])
plt.savefig(output_fig+'.eps')
