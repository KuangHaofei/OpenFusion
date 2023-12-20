import numpy as np

from openfusion.utils import show_pc

root_dir = '/home/ipbhk/OpenFusion/results/office_ipblab1/completion'
points = np.load(root_dir + '/rgb_points.npy')
colors = np.load(root_dir + '/rgb_colors.npy')

show_pc(points, colors)
