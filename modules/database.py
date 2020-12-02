import os
import h5py

import numpy as np

from torch.utils.data import Dataset
from graphics import Voxelgrid

from utils.metrics import evaluation


class Database(Dataset):

    def __init__(self, dataset, config):

        super(Database, self).__init__()

        self.transform = config.transform
        self.initial_value = config.init_value
        self.scenes_gt = {}
        self.scenes_est = {}
        self.fusion_weights = {}
        grid = dataset.get_grid()

        self.scenes_gt[0] = grid

        init_volume = self.initial_value * np.ones_like(grid.volume)

        self.scenes_est[0] = Voxelgrid(self.scenes_gt[0].resolution)
        self.scenes_est[0].from_array(init_volume, self.scenes_gt[0].bbox)
        print(self.scenes_gt[0].volume.shape)
        self.fusion_weights[0] = np.zeros(self.scenes_gt[0].volume.shape)

    def __getitem__(self, item):

        sample = dict()

        sample['gt'] = self.scenes_gt[0].volume
        sample['current'] = self.scenes_est[0].volume
        sample['origin'] = self.scenes_gt[0].origin
        sample['resolution'] = self.scenes_gt[0].resolution
        sample['weights'] = self.fusion_weights[0]


        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.scenes_gt)

    def filter(self, value=2.):

        for key in self.scenes_est.keys():

            weights = self.fusion_weights[key]
            self.scenes_est[key].volume[weights < value] = self.initial_value
            self.fusion_weights[key][weights < value] = 0

    def save_to_workspace(self, workspace):

        for key in self.scenes_est.keys():

            tsdf_volume = self.scenes_est[key].volume
            weight_volume = self.fusion_weights[key]

            tsdf_file = key.replace('/', '.') + '.tsdf.hf5'
            weight_file = key.replace('/', '.') + '.weights.hf5'

            workspace.save_tsdf_data(tsdf_file, tsdf_volume)
            workspace.save_weigths_data(weight_file, weight_volume)

    def save(self, path, scene_id=None, epoch=None, groundtruth=True):

        if scene_id is None:
            raise NotImplementedError
        else:
            if epoch is not None:
                filename = 'living.volume.hf5'
                weightname = 'living.weights.hf5'
            else:
                filename = 'living.volume.hf5'
                weightname = 'living.weights.hf5'
            print("volume number: ", self.scenes_est[scene_id].volume.shape)
            with h5py.File(os.path.join(path, filename), 'w') as hf:
                hf.create_dataset("TSDF",
                                  shape=self.scenes_est[scene_id].volume.shape,
                                  data=self.scenes_est[scene_id].volume)
            with h5py.File(os.path.join(path, weightname), 'w') as hf:
                hf.create_dataset("weights",
                                  shape=self.fusion_weights[scene_id].shape,
                                  data=self.fusion_weights[scene_id])

            if groundtruth:
                groundtruthname = 'living.gt.hf5'
                with h5py.File(os.path.join(path, groundtruthname), 'w') as hf:
                    hf.create_dataset("TSDF",
                                      shape=self.scenes_gt[
                                          scene_id].volume.shape,
                                      data=self.scenes_gt[scene_id].volume)

    def evaluate(self, mode='train', workspace=None):

        eval_results = {}

        for scene_id in self.scenes_est.keys():

            if workspace is None:
                print('Evaluating ', scene_id, '...')
            else:
                workspace.log('Evaluating {} ...'.format(scene_id),
                              mode)

            weights = self.fusion_weights[scene_id]
            est = self.scenes_est[scene_id].volume
            gt = self.scenes_gt[scene_id].volume

            mask = np.copy(weights)
            mask[mask > 0] = 1.

            eval_results_scene = evaluation(est, gt, mask)

            for key in eval_results_scene.keys():

                if workspace is None:
                    print(key, eval_results_scene[key])
                else:
                    workspace.log('{} {}'.format(key, eval_results_scene[key]),
                                  mode)

                if not eval_results.get(key):
                    eval_results[key] = eval_results_scene[key]
                else:
                    eval_results[key] += eval_results_scene[key]

        # normalizing metrics
        for key in eval_results.keys():
            eval_results[key] /= len(self.scenes_est.keys())

        return eval_results

    def reset(self):
        for scene_id in self.scenes_est.keys():
            self.scenes_est[scene_id].volume = self.initial_value * np.ones(self.scenes_est[scene_id].volume.shape)
            self.fusion_weights[scene_id] = np.zeros(self.scenes_est[scene_id].volume.shape)
