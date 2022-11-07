from __future__ import print_function
import argparse
import os.path
import socket
import time
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
from torchvision.models import ResNet18_Weights, ResNet101_Weights
from torchsummary import summary

from grid_builder.LabelBuilder import LabelBuilder
from models_pytorch.dataset import DataHelper, ImageGeolocationDataset
from models_pytorch.utils import create_datahelper, get_model


def prediction(dataset: ImageGeolocationDataset, labelBuilder : LabelBuilder, model, device):

    results = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            data, target = dataset[idx]
            data = torch.reshape(data, (1, data.shape[0], data.shape[1], data.shape[2]))
            target = target.item()
            cellId = labelBuilder.get_cellId(str(target))

            data = data.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            probs = probs[0]

            entry = dataset.get(idx)
            entry['probabilities'] = [i.item() for i in probs]
            entry['cellId'] = cellId

            results.append(entry)


            if idx == 50:
                break

    return results


def main():
    # Training settings

    parser = argparse.ArgumentParser(description='PyTorch Geolocation classifier')

    parser.add_argument('--model', type=str, default='resnet18', metavar='N',
                        help='pretrained model, supported  values: resnet18, resnet101 (default: resnet18)')
    parser.add_argument('--modelfilename', type=str, metavar='N',
                        help='names of the saved model file')
    parser.add_argument('--dataset', type=str, default='flickr_images', metavar='N',
                        help='dataset, supported are geotags_185K or flickr_images (default: flickr_images)')
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                        help='random seed (default: 123)')


    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()


    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    args.modelfilename = 'geolocation_cnn_flickr_images_14_10_2022_2.pt'
    print(f'Commandline args: {args}')
    print(f'Device: {device}')

    data_helper = create_datahelper(args.dataset, args.seed)
    labelBuilder = LabelBuilder()

    test_dataset = ImageGeolocationDataset(data_helper.test_data, augumentation=False)

    num_classes = labelBuilder.get_num_labels()
    data, label = test_dataset[0]
    input_shape = data.shape

    model = get_model(args.model, device, num_classes, input_shape)
    if not os.path.exists(args.modelfilename):
        raise RuntimeError(f'Modelfile {args.modelfilename} not found')

    #model.load_state_dict(torch.load(args.modelfilename, map_location=device))
    model.eval()

    prediction(test_dataset, labelBuilder, model, device)





if __name__ == '__main__':
    main()
