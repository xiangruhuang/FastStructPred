import sys;
import os;
import random;
from random import shuffle;

try:
    edgeFname = sys.argv[1];
    numSample = int(sys.argv[2]);
except IndexError:
    print('python3 subsample.py [edge_file] [#nodes_sampled]');
    print('OUTPUT: [edge_file].sub');
    exit();

nodeSet = set();
edgeSet = set();
with open(edgeFname) as f:
    for line in f:
        str_pair = list(line.strip().split(sep=' '));
        nodeSet.add(str_pair[0]);
        nodeSet.add(str_pair[1]);
        edgeSet.add(tuple(str_pair));

numNodes = len(nodeSet);
nodeList = list(nodeSet);

shuffle(nodeList);

subNodeList = nodeList[0:numSample];
subNodeSet = set(subNodeList);
subEdgeList = list([e for e in edgeSet if (e[0] in subNodeSet) and (e[1] in subNodeSet)]);

print('before:');
print('#node='+str(len(nodeList))+'');
print('#edge='+str(len(edgeSet))+'');

print('after:');
print('#node='+str(len(subNodeList))+'');
print('#edge='+str(len(subEdgeList))+'');

subFname = edgeFname + '.sub';
with open(subFname,'w') as f:
    for e in subEdgeList:
        f.write(e[0]+' '+e[1]+'\n');
