import torch
import argparse
import os

import numpy as np

from utils import loading
from utils import setup
from utils import transform

from modules.extractor import Extractor
from modules.integrator import Integrator
from modules.model import FusionNet
from modules.routing import ConfidenceRouting
from modules.pipeline import Pipeline

from tqdm import tqdm

# from graphics.utils import extract_mesh_marching_cubes
# from graphics.visualization import plot_mesh


def arg_parse():
    parser = argparse.ArgumentParser(description='Script for testing RoutedFusion')

    parser.add_argument('--experiment', required=True)
    parser.add_argument('--test', required=True)

    args = parser.parse_args()

    return vars(args)

def test_fusion(args, model_config, test_config):

    # define output dir
    test_dir = os.path.join(args['experiment'],
                            'tests',
                            test_config.TEST.name)
    print(test_dir)
    output_dir = os.path.join(test_dir, 'output')

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        os.makedirs(output_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("##############", device)
    model_config.MODEL.device = device

    # get test dataset
    data_config = setup.get_data_config(test_config, mode='test')
    #print("data config, ", data_config.DATA.input)
    dataset = setup.get_data(test_config.DATA.dataset, data_config)
    loader = torch.utils.data.DataLoader(dataset)

    # get test database
    database = setup.get_database(dataset, test_config, mode='test')
    print(database[0]['current'].size())
    #print("!!!!!!!!!!!", model_config)
    # setup pipeline
    pipeline = Pipeline(model_config)
    print("input depth format: ", model_config.DATA.input)
    pipeline = pipeline.to(device)

    # loading neural networks
    model_path = os.path.join(args['experiment'], 'model/best.pth.tar')
    print("!!!!!!!!!!!!!!!!",model_path)
    loading.load_pipeline(model_path, pipeline)
    #routing_checkpoint = os.path.join('/home/yan/Work/opensrc/RoutedFusion/pretrained_models/routing/shapenet_noise_005/ori_best.pth.tar')
    routing_checkpoint = os.path.join('/home/yan/Work/opensrc/learning/RoutedFusion/experiments/routing/finetuned_living2/model/best.pth.tar')
    loading.load_model(routing_checkpoint, pipeline._routing_network)

    pipeline.eval()


    for i, batch in tqdm(enumerate(loader), total=len(dataset)):
        # put all data on GPU
        #print("####", i)
        #print(batch.keys())
        batch = transform.to_device(batch, device)
        # fusion pipeline
        pipeline.fuse(batch, database, device)

    database.filter(value=3.)
    #test_eval = database.evaluate()
    #print(test_eval)

    for scene_id in database.scenes_est.keys():
        database.save(output_dir, scene_id)


if __name__ == '__main__':

    # parse commandline arguments
    args = arg_parse()

    # load config
    model_config = loading.load_experiment(args['experiment'])
    test_config = loading.load_config_from_yaml(args['test'])

    test_fusion(args, model_config, test_config)
