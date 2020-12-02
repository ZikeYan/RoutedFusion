import os
import glob
import pyquaternion
import cc3d

import numpy as np

from skimage import io, transform
from torch.utils.data import Dataset
from copy import copy
from scipy.ndimage.morphology import binary_dilation
from utils.data import add_axial_noise, add_random_zeros, add_lateral_noise, add_outliers, add_kinect_noise, add_depth_noise

from graphics import Voxelgrid

from dataset.binvox_utils import read_as_3d_array


class ICL(Dataset):

    def __init__(self, root_dir, scene='0', frame_list=None, resolution=(240, 320), transform=None, truncation=None):

        self.root_dir = root_dir
        self.scene = scene

        self.frame_list = frame_list

        self._load_color()
        self._load_depth()
        self._load_cameras()

        self.resolution = resolution
        self.xscale = resolution[0]/480
        self.yscale = resolution[1]/640
        self.grid_resolution = 256 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.noise_scale = 0.055
        self.outlier_scale = 3
        self.outlier_fraction = 0.99

        self.transform = transform
        self.truncation = truncation

    def _load_depth(self):

        self.depth_images = glob.glob(os.path.join('/home/yan/Dataset/ICL/living1/', 'depth', '*.png'))
        self.depth_images = sorted(self.depth_images, key=lambda x: int(os.path.splitext(x.split('/')[-1])[0]))
        self.noisy_depth_images = glob.glob(os.path.join('/home/yan/Dataset/ICL/living1_noise/', 'depth', '*.png'))
        self.noisy_depth_images = sorted(self.noisy_depth_images, key=lambda x: int(os.path.splitext(x.split('/')[-1])[0]))
        #print(self.noisy_depth_images)

    def _load_color(self):
        self.color_images = glob.glob(os.path.join('/home/yan/Dataset/ICL/living1/', 'rgb', '*.png'))

        self.color_images = sorted(self.color_images, key=lambda x: int(os.path.splitext(x.split('/')[-1])[0]))

    def _load_cameras(self):

        self.cameras = []

        with open(os.path.join('/home/yan/Dataset/ICL/living1/', 'traj0.gt.freiburg')) as file:

            for line in file:
                elems = line.rstrip().split(' ')
                mat = []
                for p in elems:
                    if p == '':
                        continue
                    mat.append(float(p))

                position = np.asarray(mat[1:4])
                rotation = np.asarray(mat[4:])

                M = np.eye(4)
                M[0, 0] = -1.
                M[1, 1] = 1.
                M[2, 2] = 1.

                qw = rotation[3]
                qx = rotation[0]
                qy = rotation[1]
                qz = rotation[2]

                quaternion = pyquaternion.Quaternion(qw, qx, qy, qz)
                rotation = quaternion.rotation_matrix

                extrinsics = np.eye(4)
                extrinsics[:3, :3] = rotation
                extrinsics[:3, 3] = position

                self.cameras.append(np.copy(extrinsics))

    def __len__(self):
        return len(self.color_images)

    def __getitem__(self, item):

        sample = dict()

        sample['frame_id'] = item
        #print("!!!!!!!!!!!!!!!!!!", sample['frame_id'])

        # load image
        file = self.color_images[item]
        image = io.imread(file)
        image = image[:, :, :3]
        image = transform.resize(image, self.resolution)
        sample['image'] = np.asarray(image)

        # load depth map
        file = self.depth_images[item]
        file_noisy = self.noisy_depth_images[item]
        depth = io.imread(file).astype(np.float32)
        depth_noisy = io.imread(file_noisy).astype(np.float32)

        step_x = depth.shape[0] / self.resolution[0]
        step_y = depth.shape[1] / self.resolution[1]
        #print("#######", step_x, step_y)

        index_y = [int(step_y * i) for i in
                   range(0, int(depth.shape[1] / step_y))]
        index_x = [int(step_x * i) for i in
                   range(0, int(depth.shape[0] / step_x))]

        depth = depth[:, index_y]
        depth = depth[index_x, :]
        depth_noisy = depth_noisy[:, index_y]
        depth_noisy = depth_noisy[index_x, :]

        depth /= 5000.
        depth_noisy /= 5000.

        #sample['depth'] = np.asarray(depth)

        mask = copy(depth)
        mask[mask == np.max(depth)] = 0
        mask[mask != 0] = 1
        sample['mask'] = copy(mask)
        gradient_mask = binary_dilation(mask, iterations=5)
        mask = binary_dilation(mask, iterations=8)
        sample['routing_mask'] = mask
        sample['gradient_mask'] = gradient_mask

        depth[mask == 0] = 0
        depth_noisy[mask == 0] = 0

        sample['depth'] = depth
        sample['noisy_depth'] = depth_noisy

        # load extrinsics
        extrinsics = self.cameras[item]
        sample['extrinsics'] = extrinsics
        #print("noisy depth number: ", len(self.noisy_depth_images), "camera number: ", len(self.cameras))
        # load intrinsics
        intrinsics = np.asarray([[481.20, 0., 319.50],
                                 [0., -480.05, 239.50],
                                 [0., 0., 1.]])

        scaling = np.eye(3)
        scaling[0, 0] = self.xscale
        scaling[1, 1] = self.yscale
        sample['intrinsics'] = np.dot(scaling, intrinsics)

        sample['scene_id'] = self.scene

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_grid(self):
        filepath = os.path.join('/home/yan/Dataset/ICL', 'transformed_512.binvox') #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        with open(filepath, 'rb') as file:
            volume = read_as_3d_array(file)

        array = volume.data.astype(np.int)

        # clean occupancy grids from artifacts
        labels_out = cc3d.connected_components(array)  # 26-connected
        N = np.max(labels_out)
        max_label = 0
        max_label_count = 0
        for segid in range(1, N + 1):
            extracted_image = labels_out * (labels_out == segid)
            extracted_image[extracted_image != 0] = 1
            label_count = np.sum(extracted_image)
            if label_count > max_label_count:
                max_label = segid
                max_label_count = label_count
        array[labels_out != max_label] = 0.

        resolution = 1. / self.grid_resolution

        grid = Voxelgrid(0.03) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        bbox = np.zeros((3, 2))
        bbox[:, 0] = volume.translate #[-2.707, -1.405, -6.408]
        bbox[:, 1] = bbox[:, 0] + resolution * volume.dims[0]#[2.974, 1.625, 2.687]
        print(bbox)

        grid.from_array(array, bbox)

        return grid


if __name__ == '__main__':

    from tqdm import tqdm
    from mayavi import mlab

    import matplotlib.pyplot as plt

    dataset = ICL('/home/yan/Dataset/ICL/living1/') #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def pixel_to_camera_coord(point, intrinsics, z):

        camera_coord = np.zeros(3, )

        camera_coord[2] = z
        camera_coord[1] = z * (point[1] - intrinsics[1, 2]) / intrinsics[1, 1]
        camera_coord[0] = z * (point[0] - intrinsics[0, 1] * camera_coord[1] - intrinsics[0, 2]) / intrinsics[0, 0]

        return camera_coord

    frame_counter = 0
    pointcloud = []

    # frames = np.random.choice(np.arange(0, len(dataset), 1), 20)
    frames = np.arange(0, len(dataset), 1)

    for f in tqdm(frames, total=len(frames)):

        frame = dataset[f]
        depth = frame['depth']
        # depth = np.flip(depth, axis=0)
        # plt.imshow(depth)
        # plt.show()

        for i in range(0, depth.shape[0]):
            for j in range(0, depth.shape[1]):

                z = depth[i, j]

                p = np.asarray([j, i, z])
                c = pixel_to_camera_coord([j, i], frame['intrinsics'], z)
                c = np.concatenate([c, np.asarray([1.])])
                w = np.dot(frame['extrinsics'], c)

                pointcloud.append(w)

        frame_counter += 1

        # if (frame_counter + 1) % 5 == 0:
        #     print(frame_counter)
        #     array = np.asarray(pointcloud)
        #     print(np.max(array, axis=0))
        #     print(np.min(array, axis=0))
        #
        #     mlab.points3d(array[:, 0],
        #                   array[:, 1],
        #                   array[:, 2],
        #                   scale_factor=0.05)
        #
        #     mlab.show()
        #     mlab.close(all=True)

    array = np.asarray(pointcloud)
    print(np.max(array, axis=0))
    print(np.min(array, axis=0))

    # array = np.asarray(pointcloud)
    # mlab.points3d(array[:, 0],
    #               array[:, 1],
    #               array[:, 2],
    #               scale_factor=0.05)
    #
    # mlab.show()
