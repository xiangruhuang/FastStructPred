import re
import math
def readData(filename):
	fin = open(filename, 'r')
	data = []
	for l in fin.readlines():
		num_list = [float(num) for num in re.split(' |\t', l.strip())]
		data.append(num_list)
	return data

def parse_args(args):	
	filenames=args[0:]
	print filenames
	output_fig = args[0].split('.')[0]
	labels = []
	for filename in filenames[1:]:
		label = filename.split('/')[-1].split('.')[0]
		labels.append(label)
		#output_fig = output_fig + '_' + label
	print 'output figure file=',output_fig
	filelist = filenames[1:]
	print 'filelist=', filelist
	return output_fig, labels, filelist

def exit_with_help():
	print "python <dataset name>.py <path1>/<algo1>.log,...,<pathn>/<algon>.log"
	exit()

def process(data):
	#goal less than 10 points
	lastgap = 100
	lastx = 1e-10
	eps=(math.log(data[-1][0]) - math.log(data[0][0]))/15
	indices=[]
	for i in range(len(data)):
		d = data[i]
		if (d[1] < lastgap and (math.log(d[0])-math.log(lastx) > eps or len(data) < 10)):
			indices.append(i)
			lastgap = d[1]
			lastx = d[0]
		elif (i == len(data)-1):
			indices.append(i)
	print indices
	return data[indices]
