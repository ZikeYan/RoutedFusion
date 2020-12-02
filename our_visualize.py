import h5py
import numpy as np
from graphics.utils import extract_mesh_marching_cubes
#from graphics.visualization import plot_mesh

file = '/home/yan/Work/opensrc/RoutedFusion/pretrained_models/fusion/shapenet_noise_005/tests/ICL/output/living.volume.hf5'

with h5py.File(file, 'r') as hf:
    tsdf = hf['TSDF'][:]
print(np.sum(tsdf))
mesh = extract_mesh_marching_cubes(tsdf)
mesh.write("routed.ply")
#plot_mesh(mesh)
