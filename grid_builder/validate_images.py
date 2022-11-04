import csv
import os

import PIL
from torchvision import transforms

from grid_builder.env_helper import get_base_dir, get_data_dir
from grid_builder.utility import read_cvs_file, read_excluded_file, read_validated_file


def check_images(base_dir: str, dataset_name: str, data_dir: str):

    # check data dir
    if not data_dir.endswith(dataset_name):
        data_dir = data_dir + '/' + dataset_name

    images = read_cvs_file(base_dir + '/input/' + dataset_name + '.csv')
    excluded = read_excluded_file(base_dir + '/input/' + dataset_name + '_excluded.csv')
    validated = read_validated_file(base_dir + '/input/' + dataset_name + '_validated.csv')

    updated = False

    call_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    with open(base_dir + '/input/' + dataset_name + '_excluded.csv', 'a', encoding='UTF8', newline='') as excluded_file, \
            open(base_dir + '/input/' + dataset_name + '_validated.csv', 'a', encoding='UTF8', newline='') as validated_file:

        excluded_writer = csv.writer(excluded_file)
        validated_writer = csv.writer(validated_file)

        for idx, id in enumerate(images):
            url = images[id]['url']
            name = url.split('/')[-1]
            file_name = data_dir + '/' + name

            if not os.path.isfile(file_name):
                # print(f'No data for file with id {id} downloaded')
                continue

            if id in excluded or id in validated:
                continue

            try:
                image = PIL.Image.open(file_name)
                image = call_transforms(image)
                validated_writer.writerow([id, url])
                validated_file.flush()
                print(f'Count {idx+1}/{len(images)} validated id {id} filename {file_name}')

            except Exception as e:
                print(f'Count {idx+1}/{len(images)} Exception: {e} id {id} filename {file_name}')
                excluded_writer.writerow([id, url])
                excluded_file.flush()

            updated = True

    return updated



if __name__ == '__main__':

    base_dir = get_base_dir()
    data_dir = get_data_dir()


    #check_data_set(base_dir, 'flickr_images', data_dir, False)
    check_images(base_dir, 'flickr_images', data_dir)