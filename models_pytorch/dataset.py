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

from grid_builder.flickr_search_images import read_cvs_file, read_labels_file, read_excluded_file


class ImageGeolocationDataset(torch.utils.data.Dataset):

    def __init__(self, files: List):
        self.ids = [None] * len(files)
        self.labels = [None] * len(files)
        self.file_names = [None] * len(files)

        for idx, elem in enumerate(files):
            self.ids[idx] = elem['id']
            self.labels[idx] = int(elem['label'])
            self.file_names[idx] = elem['filename']

        self.transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

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


class DataHelper:

    def __init__(self, base_dir: str, dataset_name: str, data_dir: str, test_fraction: float=0.8, seed: int=42):
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.images = read_cvs_file(base_dir + '/input/' + dataset_name + '.csv')
        self.labels = read_labels_file(base_dir + '/output/' + dataset_name + '_label.csv')
        self.excluded = read_excluded_file(base_dir + '/output/' + dataset_name + '_excluded.csv')
        self.all_data = []
        self.all_labels = set()

        num_missing_labels = 0
        num_missing_images = 0
        num_excluded_images = 0

        for id in self.images:

            if id in self.excluded:
                num_excluded_images += 1
                continue

            if id not in self.labels:
                #print(f'Warning: no label for file with id {id}')
                num_missing_labels += 1
                continue

            label = self.labels[id]
            url = self.images[id]['url']
            name = url.split('/')[-1]
            file_name = data_dir + '/' + name

            if not os.path.isfile(file_name):
                #print(f'No data for file with id {id} downloaded')
                num_missing_images += 1
                continue

            self.all_data.append({'id': id, 'filename': file_name, 'label': label})
            self.all_labels.add(label)


        # train test splitt
        random.seed(seed)
        random.shuffle(self.all_data)

        self.training_data = self.all_data[:int(test_fraction * len(self.all_data))]
        self.test_data = self.all_data[int(test_fraction * len(self.all_data)):]

        print('-' * 10)
        print(f'DataHelper {self.dataset_name} basedir {self.base_dir}')
        print(f'Missing: {num_missing_labels} labels and {num_missing_images} downloaded images, excluded {num_excluded_images} images')
        print(f'Found {len(self.all_data)} of {len(self.images) - num_excluded_images} valid image, {len(self.all_labels)} labels, {len(self.training_data)} training data and {len(self.test_data)} test data')
        print('-'*10)

    def update_excluded_file(self):

        updated = False

        call_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


        with open(self.base_dir + '/output/' + self.dataset_name + '_excluded.csv', 'a', encoding='UTF8', newline='') as file:
            writer = csv.writer(file)

            for data in self.all_data:
                id = data['id']
                filename = data['filename']
                try:
                    image = PIL.Image.open(filename)
                    image = call_transforms(image)
                except Exception as e:
                    updated = True
                    print(f'Execption: {e} id {id} filename {filename}')
                    url = self.images[id]['url']
                    writer.writerow([id, url])

        return updated


def check_data_set(data_set_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    helper = DataHelper(base_dir='/home/hacke/projects/adncuba-geolocation-classifier/grid_builder',
                        dataset_name=data_set_name,
                        data_dir='/home/hacke/projects/data/geolocation_classifier',
                        test_fraction=0.8, seed=42)

    if helper.update_excluded_file():
        helper = DataHelper(base_dir='/home/hacke/projects/adncuba-geolocation-classifier/grid_builder',
                            dataset_name=data_set_name,
                            data_dir='/home/hacke/projects/data/geolocation_classifier',
                            test_fraction=0.8, seed=42)

    training_data = ImageGeolocationDataset(helper.training_data)
    data, label = training_data[1]
    training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=1, shuffle=True)

    for batch in training_data_loader:
        input = batch[0]
        label = batch[1]
        input = input.to(device)
        label = label.to(device)

    print(f'Checked dataset {data_set_name}')



if __name__ == '__main__':
    check_data_set('flickr_images')

