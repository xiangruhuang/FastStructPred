import matplotlib.pyplot as plt
import sys
import numpy

fin = open(sys.argv[1], 'r')
w = int(sys.argv[2])
h = int(sys.argv[3])
lines = fin.readlines()
tokens = lines[1].strip().split(' ')
im = [int(token) for token in tokens]
im = numpy.asarray(im)
print type(im)
print im.shape
im = im.reshape(h, w)

plt.gray()
plt.imshow(im)
plt.show()
