from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
colors = {'TRWS':'c', 'AD3':'g', 'PSDD':'b', 'MPLP':'y', 'GDMM':'r', 'Soft-BCFW':'m', 'Soft-BCFW-acc':'m', 'LPsparse':'k', 'smoothMSD':'brown'}
linestyles = {'TRWS':'-', 'AD3':'-', 'PSDD':'-', 'MPLP':'-', 'GDMM':'-', 'Soft-BCFW':'--', 'Soft-BCFW-acc':':', 'LPsparse':'-', 'smoothMSD':'-'}
linewidths = {'TRWS':1, 'AD3':1.5, 'PSDD':1, 'MPLP':1, 'GDMM':2, 'Soft-BCFW':1, 'Soft-BCFW-acc':1, 'LPsparse':1, 'smoothMSD':0.5}
markers = {'TRWS':'*', 'AD3':'o', 'PSDD':'>', 'MPLP':'^', 'GDMM':'', 'Soft-BCFW':'', 'Soft-BCFW-acc':'', 'LPsparse':'d', 'smoothMSD':'v'}
#dash = {'TRWS':[4, 2, 1, 2], 'AD3':[1, 1, 1, 1], 'PSDD':[4, 2, 4, 2, 1, 2], 'MPLP':[1, 2, 1, 2], 'GDMM':[], 'Soft-BCFW':[4, 2, 4, 2], 'Soft-BCFW-acc':[2, 4, 2, 4], 'LPsparse':[4, 2, 1, 2, 1, 2], 'smoothMSD':[]}

eps=1e-5
title_fontsize=38
