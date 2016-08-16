import sys

fin = open(sys.argv[1])
lines = fin.readlines()
x = float(lines[0].split()[1])
y = float(lines[-1].split()[1])
if abs(x-y) > 1e-3:
	print sys.argv[1]

