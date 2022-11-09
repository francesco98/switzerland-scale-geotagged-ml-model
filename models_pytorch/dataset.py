from __future__ import print_function, division

import csv
import random
from typing import List

import PIL
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy

from torchvision import transforms
from torchvision.transforms._presets import ImageClassification

from grid_builder.env_helper import get_base_dir, get_data_dir
from grid_builder.utility import read_cvs_file, read_labels_file, read_excluded_file, read_validated_file


class ImageGeolocationDataset(torch.utils.data.Dataset):

    def __init__(self, files: List, preprocessing: ImageClassification, augumentation: bool):
        self.ids = [None] * len(files)
        self.labels = [None] * len(files)
        self.file_names = [None] * len(files)
        self.augumentation = augumentation
        self.preprocessing = preprocessing

        for idx, elem in enumerate(files):
            self.ids[idx] = elem['id']
            self.labels[idx] = int(elem['label'])
            self.file_names[idx] = elem['filename']

        if not self.augumentation:
            # just normalization
            self.transforms = transforms.Compose([
                    transforms.Resize(self.preprocessing.resize_size[0]),
                    transforms.CenterCrop(self.preprocessing.crop_size[0]),
                    transforms.ToTensor(),
                    transforms.Normalize(self.preprocessing.mean, self.preprocessing.std)
            ])
        else:
            # Data augmentation and normalization
            self.transforms = transforms.Compose([
                    transforms.RandomResizedCrop(self.preprocessing.crop_size[0]),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.preprocessing.mean, self.preprocessing.std)
            ])

        print(f'Dataset created: {len(files)} datapoints, augmentations: { "RandomResizedCrop and RandomHorizontalFlip" if self.augumentation else "none"}')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        try:
            if torch.is_tensor(idx):
                idx = idx.tolist()
                images = [PIL.Image.open(self.file_names[i]) for i in idx]
                labels = [self.labels[i] for i in idx]
            else:
                images = PIL.Image.open(self.file_names[idx])
                labels = self.labels[idx]

            images = self.transforms(images)
            labels = torch.tensor(labels)

            return images, labels

        except Exception as e:
            print(f'Execption: {e}')

            if idx is List:
                print(f'Files: {[self.file_names[i] for i in idx]}')
            else:
                print(f'File: {self.file_names[idx]}')

            raise e

    def get(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            return {'ids': [self.ids[i] for i in idx], 'labels': [self.labels[i] for i in idx], 'filenames':  [self.file_names[i] for i in idx]}
        else:
            return {'id': self.ids[idx], 'label': self.labels[idx], 'filename': self.file_names[idx]}


class DataHelper:

    def __init__(self, base_dir: str, dataset_name: str, data_dir: str, test_fraction: float=0.8, seed: int=42, check_all_images=False):
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.images = read_cvs_file(base_dir + '/input/' + dataset_name + '.csv')
        self.labels = read_labels_file(base_dir + '/output/' + dataset_name + '_label.csv')
        self.excluded = read_excluded_file(base_dir + '/input/' + dataset_name + '_excluded.csv')
        self.validated = read_validated_file(base_dir + '/input/' + dataset_name + '_validated.csv')
        self.all_data = []
        self.all_labels = set()

        # check data dir
        if not data_dir.endswith(dataset_name):
            data_dir = data_dir + '/' + dataset_name

        if not os.path.isdir(data_dir):
            raise RuntimeError(f'Data directory {data_dir} not existing')

        num_missing_labels = 0
        num_missing_images = 0
        num_excluded_images = 0

        for id in self.images:

            url = self.images[id]['url']
            name = url.split('/')[-1]
            file_name = data_dir + '/' + name

            if not os.path.isfile(file_name):
                # print(f'No data for file with id {id} downloaded')
                num_missing_images += 1
                continue

            if id in self.excluded:
                num_excluded_images += 1
                continue

            if id not in self.validated:
                msg = f'ERROR: file {file_name} with id {id} not validated'
                print(msg)
                raise RuntimeError(msg)

            if id not in self.labels:
                #print(f'Warning: no label for file with id {id}')
                num_missing_labels += 1
                continue

            if check_all_images:
                try:
                    image = PIL.Image.open(file_name)
                    image = transforms.Resize(256)(image)

                except Exception as e:
                    print(f'Ignoring invalid filename {file_name}')
                    #os.remove(file_name)
                    continue

            label = self.labels[id]
            self.all_labels.add(label)
            self.all_data.append({'id': id, 'filename': file_name, 'label': label})

        # check the labels
        if False:
            for idx in range(len(self.all_labels)):
                label = f'{idx}'
                if label not in self.all_labels:
                    msg = f'Missing datapoint with label {label}, all-labels: {self.all_labels}'
                    print(f'ERROR: {msg}')
                    raise RuntimeError(msg)

        # train test splitt
        random.seed(seed)
        random.shuffle(self.all_data)

        self.training_data = self.all_data[:int(test_fraction * len(self.all_data))]
        self.test_data = self.all_data[int(test_fraction * len(self.all_data)):]

        print('-' * 10)
        print(f'DataHelper {self.dataset_name} basedir {self.base_dir}')
        print(f'Missing: {num_missing_labels} labels and {num_missing_images} downloaded images, excluded {num_excluded_images} images')
        print(f'Found {len(self.all_data)} of {len(self.images) - num_excluded_images} valid image, {len(self.all_labels)} labels, {len(self.training_data)} training data and {len(self.test_data)} test data (seed {seed})')
        print('-'*10)



def check_data_set(base_dir: str, data_set_name: str, data_dir: str, check_all_batches: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    helper = DataHelper(base_dir=base_dir,
                        dataset_name=data_set_name,
                        data_dir=data_dir,
                        test_fraction=0.8, seed=42)

    batch_size = 250
    training_data = ImageGeolocationDataset(helper.training_data, augumentation=False)
    data, label = training_data[1]
    training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

    num_batches = int(len(training_data)/batch_size)
    for batch_idx, (data, target) in enumerate(training_data_loader):
        data = data.to(device)
        target = target.to(device)
        print(f'Batch {batch_idx+1}/{num_batches}')

        if not check_all_batches and batch_idx > 1000:
            print('Stopping batch check')
            break

    print(f'Checked dataset {data_set_name}')



if __name__ == '__main__':
    base_dir = get_base_dir()
    data_dir = get_data_dir()

    #check_data_set(base_dir, 'flickr_images', data_dir, False)
    check_data_set(base_dir, 'flickr_images', data_dir, False)
