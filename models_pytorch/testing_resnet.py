from __future__ import print_function
import argparse
import io
import os.path
import socket
import time
import datetime

import PIL
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from grid_builder.LabelBuilder import LabelBuilder
from models_pytorch.dataset import DataHelper, ImageGeolocationDataset
from models_pytorch.utils import create_datahelper, get_model


class Predictor:
    def __init__(self, model, device):

        self.model = model
        self.device = device
        # just normalization
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.model.eval()

    def predict_image(self, buff: io.BytesIO):

        with torch.no_grad():
            image = Image.open(buff)
            image = image.convert('RGB')
            image = self.transforms(image)


            images = torch.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
            images = images.to(self.device)

            output = self.model(images)
            probs = F.softmax(output, dim=1)
            probs = [i.item() for i in probs[0]]

            return probs


def predict_dataset(dataset: ImageGeolocationDataset, labelBuilder : LabelBuilder, model, device, max_limit: int=None):

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


            if max_limit and idx == max_limit:
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
