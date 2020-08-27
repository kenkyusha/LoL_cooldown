# colorplot

import matplotlib
import matplotlib.pyplot as plt
import colorsys
import numpy as np
import matplotlib.colors as mcolors
#import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
# from draw_utils import calc_metrics



import pdb
'''
colors = (
		(0,1,0),
		(1,1,0),
		(1,0,0)
	)

asd = mcolors.LinearSegmentedColormap.from_list('Jamaica', colors)
'''
def calc_metrics(pts):
# 	# pts contains RNG POS AND CUR_POS
#     # FORMAT: (rng_pos, cur_pos)
#     # FORMAT: (TRUEPOS, CUR_POS)
# 	#rng_pos = np.array([list(y) for y in [x[0] for x in pts]]) # #cur_pos = np.array([list(y) for y in [x[1] for x in pts]])
# 	# pdb.set_trace()
#     if 0:
# 	    rng_pos = [list(y) for y in [x[0] for x in pts]]
# 	    cur_pos = [list(y) for y in [x[1] for x in pts]]
#     else:
#         rng_pos = [list(y) for y in pts[0]]
#         cur_pos = [list(y) for y in pts[1]]
#     if 1:
#         # REMOVE 0
#         indices = [sum(item) for item in cur_pos if sum(item) == 0]
#         for item in reversed(indices):
#             rng_pos.pop(item)
#             cur_pos.pop(item)
	
#     rng_pos = np.array(pts[0])
#     cur_pos = np.array(pts[1])
    true = np.array(pts[0]).reshape(len(pts[0]),1)
    pred = np.array(pts[1]).reshape(len(pts[1]),1)
    ER = np.sqrt(np.sum(np.square(true-pred), axis=1))
    MAE = np.mean(ER)
    norm_ER = ER/np.max(ER)
    sort_ER = np.sort(ER)
    CEP_ind = int(sort_ER.shape[0]*0.5)
    CEP95_ind = int(sort_ER.shape[0]*0.95)
    if 0: 
        print('MAE = ', MAE)
        print('CEP = ', sort_ER[CEP_ind])
        print('CEP95 = ', sort_ER[CEP95_ind])
    return ER, MAE, sort_ER[CEP_ind], sort_ER[CEP95_ind]

def pretty_plot(pts, str_name, savepath=None):
    # rng_pos = true pos
    # cur_pos = predicted pos
    colors = (
        (0,1,0),
        (1,1,0),
        (1,0,0)
    )

    asd = mcolors.LinearSegmentedColormap.from_list('Jamaica', colors)
    #rng_pos = np.array(pts[0])#.reshape(len(pts[0]),1)
    #cur_pos = np.array(pts[1])#.reshape(len(pts[1]),1)
    

    ER, MAE, CEP, CE95 = calc_metrics(pts)
    if 0:
        rng_pos = np.array([np.array([x,x]) for x in pts[0]])
        cur_pos = np.array([np.array([x,x]) for x in pts[1]])
    rng_pos = np.array(pts[0]).reshape(len(pts[0]),1)
    cur_pos = np.array(pts[1]).reshape(len(pts[1]),1)
    
#     pdb.set_trace()

    fig1 = plt.figure()
    # 	str_name = name #.split('/')[1].split('.')[0]
    fig1.suptitle('Scenario: '+ str_name)
    ax0 = fig1.add_subplot(111)
#     ax0.plot(rng_pos[:,0], rng_pos[:,1], 'bo', zorder = 1, mfc='none', label='target')
    ax0.plot(rng_pos, 'bo', zorder = 1, mfc='none', label='target')
#     im0 = ax0.scatter(cur_pos[:,0], cur_pos[:,1], s= 10, c=ER, cmap = asd, label = 'pred', zorder = 3)
#     im0 = ax0.scatter(cur_pos, s= 10, c=ER, cmap = asd, label = 'pred', zorder = 3)
    #im0 = ax0.plot(cur_pos, c=ER, cmap = asd, label = 'pred', zorder = 3)
    im0 = ax0.plot(cur_pos, 'ro', mfc='none', label='pred', zorder = 3)
    if 0:
        im0.set_clim(0.0, 200.0)
    title = 'MAE= %.3f' % MAE, 'CEP= %.3f' % CEP, 'CE95= %.3f' % CE95
    ax0.set_title(title)
#     ax0.set_xlabel('W [px]')
#     ax0.set_ylabel('H [px]')
#     ax0.set_ylim(0,900)
#     ax0.set_xlim(0,1600)
    if 0:
        points = rng_pos.reshape(-1,1,2)
        refs = cur_pos.reshape(-1,1,2)
        seg = np.concatenate([points, refs], axis = 1)
        fig1.gca().add_collection(LineCollection(seg, linewidths=0.5, color='0.7', linestyle = '--'))
    if 0:
        cbar = fig1.colorbar(im0, ax = ax0)
        cbar.set_label('Error [px]')
    fig1.gca().invert_yaxis()
    ax0.grid()
    if savepath:
        plt.savefig(savepath+str_name+'_XY'+'.pdf')
    else:
        plt.savefig(str_name+'.pdf')
    plt.close()
