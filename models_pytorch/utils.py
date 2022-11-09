import socket


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights, ResNet101_Weights, ResNet50_Weights
from torchsummary import summary

from models_pytorch.dataset import DataHelper


def create_datahelper(dataset_name: str, seed: int):

    hostname = socket.gethostname()

    # adnwsrtx01
    base_dir = '/home/test-dev/projects/adncuba-geolocation-classifier/grid_builder'
    data_dir = '/mnt/store/geolocation_classifier/datadir'

    # hacke vmware
    if hostname.startswith('adnlt903'):
        base_dir = '/home/hacke/projects/adncuba-geolocation-classifier/grid_builder'
        data_dir = '/home/hacke/projects/data/geolocation_classifier'

    data_helper = DataHelper(base_dir=base_dir, dataset_name=dataset_name, data_dir=data_dir, test_fraction=0.8, seed=seed, check_all_images=False)

    return data_helper


def get_model(model_name: str, device, num_classes: int, input_shape):

    if model_name.lower() == 'resnet101':
        model_ft = models.resnet101(weights=ResNet101_Weights.DEFAULT)
    elif model_name.lower() == 'resnet50':
        model_ft = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    elif model_name.lower() == 'resnet18':
        model_ft = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        raise RuntimeError(f'Unsupported model {model_name}')

    num_ftrs = model_ft.fc.in_features

    # set numer of classes
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    model_ft = model_ft.to(device)

    # print a summary
    summary(model_ft, input_shape, device=str(device))
    return model_ft

